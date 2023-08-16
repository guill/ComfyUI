import os
import sys
import copy
import json
import threading
import heapq
import traceback
import gc
import time
from enum import Enum

import torch
import nodes

import comfy.model_management
import comfy.graph_utils
from comfy.graph_utils import is_link, ExecutionBlocker, GraphBuilder
import extension_points

class ExecutionResult(Enum):
    SUCCESS = 0
    FAILURE = 1
    SLEEPING = 2

def get_input_info(class_def, input_name):
    valid_inputs = class_def.INPUT_TYPES()
    input_info = None
    input_category = None
    if "required" in valid_inputs and input_name in valid_inputs["required"]:
        input_category = "required"
        input_info = valid_inputs["required"][input_name]
    elif "optional" in valid_inputs and input_name in valid_inputs["optional"]:
        input_category = "optional"
        input_info = valid_inputs["optional"][input_name]
    elif "hidden" in valid_inputs and input_name in valid_inputs["hidden"]:
        input_category = "hidden"
        input_info = valid_inputs["hidden"][input_name]
    if input_info is None:
        return None, None, None
    input_type = input_info[0]
    if len(input_info) > 1:
        extra_info = input_info[1]
    else:
        extra_info = {}
    return input_type, input_category, extra_info

# ExecutionList implements a topological dissolve of the graph. After a node is staged for execution,
# it can still be returned to the graph after having further dependencies added.
class TopologicalSort:
    def __init__(self, dynprompt):
        self.dynprompt = dynprompt
        self.pendingNodes = {}
        self.blockCount = {} # Number of nodes this node is directly blocked by
        self.blocking = {} # Which nodes are blocked by this node

    def get_input_info(self, unique_id, input_name):
        class_type = self.dynprompt.get_node(unique_id)["class_type"]
        class_def = nodes.NODE_CLASS_MAPPINGS[class_type]
        return get_input_info(class_def, input_name)

    def make_input_strong_link(self, to_node_id, to_input):
        inputs = self.dynprompt.get_node(to_node_id)["inputs"]
        if to_input not in inputs:
            raise Exception("Node %s says it needs input %s, but there is no input to that node at all" % (to_node_id, to_input))
        value = inputs[to_input]
        if not is_link(value):
            raise Exception("Node %s says it needs input %s, but that value is a constant" % (to_node_id, to_input))
        from_node_id, from_socket = value
        self.add_strong_link(from_node_id, from_socket, to_node_id)

    def add_strong_link(self, from_node_id, from_socket, to_node_id):
        self.add_node(from_node_id)
        if to_node_id not in self.blocking[from_node_id]:
            self.blocking[from_node_id][to_node_id] = {}
            self.blockCount[to_node_id] += 1
        self.blocking[from_node_id][to_node_id][from_socket] = True

    def add_node(self, unique_id):
        if unique_id in self.pendingNodes:
            return
        self.pendingNodes[unique_id] = True
        self.blockCount[unique_id] = 0
        self.blocking[unique_id] = {}

        inputs = self.dynprompt.get_node(unique_id)["inputs"]
        for input_name in inputs:
            value = inputs[input_name]
            if is_link(value):
                from_node_id, from_socket = value
                input_type, input_category, input_info = self.get_input_info(unique_id, input_name)
                if "lazy" not in input_info or not input_info["lazy"]:
                    self.add_strong_link(from_node_id, from_socket, unique_id)

    def get_ready_nodes(self):
        return [node_id for node_id in self.pendingNodes if self.blockCount[node_id] == 0]

    def pop_node(self, unique_id):
        del self.pendingNodes[unique_id]
        for blocked_node_id in self.blocking[unique_id]:
            self.blockCount[blocked_node_id] -= 1
        del self.blocking[unique_id]

    def is_empty(self):
        return len(self.pendingNodes) == 0

class ExecutionList(TopologicalSort):
    def __init__(self, dynprompt, outputs):
        super().__init__(dynprompt)
        self.outputs = outputs
        self.staged_node_id = None

    def add_strong_link(self, from_node_id, from_socket, to_node_id):
        if from_node_id in self.outputs:
            # Nothing to do
            return
        super().add_strong_link(from_node_id, from_socket, to_node_id)

    def stage_node_execution(self):
        assert self.staged_node_id is None
        if self.is_empty():
            return None
        available = self.get_ready_nodes()
        if len(available) == 0:
            raise Exception("Dependency cycle detected")
        next_node = available[0]
        # If an output node is available, do that first.
        # Technically this has no effect on the overall length of execution, but it feels better as a user
        # for a PreviewImage to display a result as soon as it can
        # Some other heuristics could probably be used here to improve the UX further.
        for node_id in available:
            class_type = self.dynprompt.get_node(node_id)["class_type"]
            class_def = nodes.NODE_CLASS_MAPPINGS[class_type]
            if hasattr(class_def, 'OUTPUT_NODE') and class_def.OUTPUT_NODE == True:
                next_node = node_id
                break
        self.staged_node_id = next_node
        return self.staged_node_id

    def unstage_node_execution(self):
        assert self.staged_node_id is not None
        self.staged_node_id = None

    def complete_node_execution(self):
        node_id = self.staged_node_id
        self.pop_node(node_id)
        self.staged_node_id = None


class DynamicPrompt:
    def __init__(self, original_prompt):
        # The original prompt provided by the user
        self.original_prompt = original_prompt
        # Any extra pieces of the graph created during execution
        self.ephemeral_prompt = {}
        self.ephemeral_parents = {}
        self.ephemeral_display = {}

    def get_node(self, node_id):
        if node_id in self.ephemeral_prompt:
            return self.ephemeral_prompt[node_id]
        if node_id in self.original_prompt:
            return self.original_prompt[node_id]
        return None

    def add_ephemeral_node(self, node_id, node_info, parent_id, display_id):
        self.ephemeral_prompt[node_id] = node_info
        self.ephemeral_parents[node_id] = parent_id
        self.ephemeral_display[node_id] = display_id

    def get_real_node_id(self, node_id):
        while node_id in self.ephemeral_parents:
            node_id = self.ephemeral_parents[node_id]
        return node_id

    def get_parent_node_id(self, node_id):
        return self.ephemeral_parents.get(node_id, None)

    def get_display_node_id(self, node_id):
        while node_id in self.ephemeral_display:
            node_id = self.ephemeral_display[node_id]
        return node_id

def get_input_data(inputs, class_def, context, outputs={}):
    valid_inputs = class_def.INPUT_TYPES()
    input_data_all = {}
    for x in inputs:
        input_data = inputs[x]
        input_type, input_category, input_info = get_input_info(class_def, x)
        if is_link(input_data) and not input_info.get("rawLink", False):
            input_unique_id = input_data[0]
            output_index = input_data[1]
            if input_unique_id not in outputs:
                continue # This might be a lazily-evaluated input
            obj = outputs[input_unique_id][output_index]
            input_data_all[x] = obj
        elif input_category is not None:
            input_data_all[x] = [input_data]

    if "hidden" in valid_inputs:
        h = valid_inputs["hidden"]
        for x in h:
            if h[x] == "PROMPT":
                input_data_all[x] = [context.get_raw_prompt()]
            elif h[x] == "DYNPROMPT":
                input_data_all[x] = [context.get_dynamic_prompt()]
            elif h[x] == "EXTRA_PNGINFO":
                extra_png_info = context.get_png_info()
                if extra_png_info is not None:
                    input_data_all[x] = [extra_png_info]
            elif h[x] == "UNIQUE_ID":
                input_data_all[x] = [context.get_current_node()]
            elif h[x] == "CONTEXT":
                input_data_all[x] = [context]
    return input_data_all

def map_node_over_list(obj, input_data_all, func, allow_interrupt=False, execution_block_cb=None, pre_execute_cb=None):
    # check if node wants the lists
    input_is_list = False
    if hasattr(obj, "INPUT_IS_LIST"):
        input_is_list = obj.INPUT_IS_LIST

    if len(input_data_all) == 0:
        max_len_input = 0
    else:
        max_len_input = max([len(x) for x in input_data_all.values()])
     
    # get a slice of inputs, repeat last input when list isn't long enough
    def slice_dict(d, i):
        d_new = dict()
        for k,v in d.items():
            d_new[k] = v[i if len(v) > i else -1]
        return d_new
    
    results = []
    if input_is_list:
        if allow_interrupt:
            nodes.before_node_execution()
        execution_block = None
        for k, v in input_data_all.items():
            for input in v:
                if isinstance(v, ExecutionBlocker):
                    execution_block = execution_block_cb(v) if execution_block_cb is not None else v
                    break

        if execution_block is None:
            if pre_execute_cb is not None:
                pre_execute_cb(0)
            results.append(getattr(obj, func)(**input_data_all))
        else:
            results.append(execution_block)
    elif max_len_input == 0:
        if allow_interrupt:
            nodes.before_node_execution()
        results.append(getattr(obj, func)())
    else: 
        for i in range(max_len_input):
            if allow_interrupt:
                nodes.before_node_execution()
            input_dict = slice_dict(input_data_all, i)
            execution_block = None
            for k, v in input_dict.items():
                if isinstance(v, ExecutionBlocker):
                    execution_block = execution_block_cb(v) if execution_block_cb is not None else v
                    break
            if execution_block is None:
                if pre_execute_cb is not None:
                    pre_execute_cb(i)
                results.append(getattr(obj, func)(**input_dict))
            else:
                results.append(execution_block)
    return results

def merge_result_data(results, obj):
    # check which outputs need concatenating
    output = []
    output_is_list = [False] * len(results[0])
    if hasattr(obj, "OUTPUT_IS_LIST"):
        output_is_list = obj.OUTPUT_IS_LIST

    # merge node execution results
    for i, is_list in zip(range(len(results[0])), output_is_list):
        if is_list:
            output.append([x for o in results for x in o[i]])
        else:
            output.append([o[i] for o in results])
    return output

def get_output_data(obj, input_data_all, execution_block_cb=None, pre_execute_cb=None):
    
    results = []
    raw_custom_outputs = {}
    subgraph_results = []
    return_values = map_node_over_list(obj, input_data_all, obj.FUNCTION, allow_interrupt=True, execution_block_cb=execution_block_cb, pre_execute_cb=pre_execute_cb)
    has_subgraph = False
    for i in range(len(return_values)):
        r = return_values[i]
        if isinstance(r, dict):
            for key in extension_points.get_custom_outputs():
                if key in r:
                    if key not in raw_custom_outputs:
                        raw_custom_outputs[key] = []
                    raw_custom_outputs[key].append(r[key])
            if 'expand' in r:
                # Perform an expansion, but do not append results
                has_subgraph = True
                new_graph = r['expand']
                result = r.get("result", None)
                if isinstance(result, ExecutionBlocker):
                    result = tuple([result] * len(obj.RETURN_TYPES))
                subgraph_results.append((new_graph, result))
            elif 'result' in r:
                result = r.get("result", None)
                if isinstance(result, ExecutionBlocker):
                    result = tuple([result] * len(obj.RETURN_TYPES))
                results.append(result)
                subgraph_results.append((None, result))
        else:
            if isinstance(r, ExecutionBlocker):
                r = tuple([r] * len(obj.RETURN_TYPES))
            results.append(r)
    
    if has_subgraph:
        output = subgraph_results
    elif len(results) > 0:
        output = merge_result_data(results, obj)
    else:
        output = []
    custom_outputs = {}
    for key, value in raw_custom_outputs.items():
        custom_output = extension_points.get_custom_output(key)
        custom_outputs[key] = custom_output.merge_list_results(value)
    return output, custom_outputs, has_subgraph

def format_value(x):
    if x is None:
        return None
    elif isinstance(x, (int, float, bool, str)):
        return x
    else:
        return str(x)

def non_recursive_execute(server, dynprompt, outputs, current_item, extra_data, executed, prompt_id, custom_outputs, object_storage, execution_list, pending_subgraph_results):
    unique_id = current_item
    real_node_id = dynprompt.get_real_node_id(unique_id)
    display_node_id = dynprompt.get_display_node_id(unique_id)
    inputs = dynprompt.get_node(unique_id)['inputs']
    class_type = dynprompt.get_node(unique_id)['class_type']
    class_def = nodes.NODE_CLASS_MAPPINGS[class_type]
    if unique_id in outputs:
        return (ExecutionResult.SUCCESS, None, None)

    input_data_all = None
    try:
        if unique_id in pending_subgraph_results:
            cached_results = pending_subgraph_results[unique_id]
            resolved_outputs = []
            for is_subgraph, result in cached_results:
                if not is_subgraph:
                    resolved_outputs.append(result)
                else:
                    resolved_output = []
                    for r in result:
                        if is_link(r):
                            source_node, source_output = r[0], r[1]
                            node_output = outputs[source_node][source_output]
                            for o in node_output:
                                resolved_output.append(o)

                        else:
                            resolved_output.append(r)
                    resolved_outputs.append(tuple(resolved_output))
            output_data = merge_result_data(resolved_outputs, class_def)
            custom_node_outputs = custom_outputs.get(unique_id, {})
            has_subgraph = False
        else:
            context = extension_points.InputContext(unique_id, dynprompt=dynprompt, extra_data=extra_data, custom_outputs=custom_outputs)
            input_data_all = get_input_data(inputs, class_def, context, outputs)
            if server.client_id is not None:
                server.last_node_id = display_node_id
                server.send_sync("executing", { "node": display_node_id, "prompt_id": prompt_id }, server.client_id)

            obj = object_storage.get((unique_id, class_type), None)
            if obj is None:
                obj = class_def()
                object_storage[(unique_id, class_type)] = obj

            if hasattr(obj, "check_lazy_status"):
                required_inputs = map_node_over_list(obj, input_data_all, "check_lazy_status", allow_interrupt=True)
                required_inputs = set(sum([r for r in required_inputs if isinstance(r,list)], []))
                required_inputs = [x for x in required_inputs if isinstance(x,str) and x not in input_data_all]
                if len(required_inputs) > 0:
                    for i in required_inputs:
                        execution_list.make_input_strong_link(unique_id, i)
                    return (ExecutionResult.SLEEPING, None, None)

            def execution_block_cb(block):
                if block.message is not None:
                    mes = {
                        "prompt_id": prompt_id,
                        "node_id": unique_id,
                        "node_type": class_type,
                        "executed": list(executed),

                        "exception_message": "Execution Blocked: %s" % block.message,
                        "exception_type": "ExecutionBlocked",
                        "traceback": [],
                        "current_inputs": [],
                        "current_outputs": [],
                    }
                    server.send_sync("execution_error", mes, server.client_id)
                    return ExecutionBlocker(None)
                else:
                    return block
            def pre_execute_cb(call_index):
                GraphBuilder.set_default_prefix(unique_id, call_index, 0)
            output_data, custom_node_outputs, has_subgraph = get_output_data(obj, input_data_all, execution_block_cb=execution_block_cb, pre_execute_cb=pre_execute_cb)
            custom_outputs[unique_id] = custom_node_outputs
        if has_subgraph:
            cached_outputs = []
            for i in range(len(output_data)):
                new_graph, node_outputs = output_data[i]
                if new_graph is None:
                    cached_outputs.append((False, node_outputs))
                else:
                    # Check for conflicts
                    for node_id in new_graph.keys():
                        if dynprompt.get_node(node_id) is not None:
                            raise Exception("Attempt to add duplicate node %s" % node_id)
                            break
                    new_output_ids = []
                    for node_id, node_info in new_graph.items():
                        display_id = node_info.get("override_display_id", unique_id)
                        dynprompt.add_ephemeral_node(node_id, node_info, unique_id, display_id)
                        # Figure out if the newly created node is an output node
                        class_type = node_info["class_type"]
                        class_def = nodes.NODE_CLASS_MAPPINGS[class_type]
                        if hasattr(class_def, 'OUTPUT_NODE') and class_def.OUTPUT_NODE == True:
                            new_output_ids.append(node_id)
                    for node_id in new_output_ids:
                        execution_list.add_node(node_id)
                    for i in range(len(node_outputs)):
                        if is_link(node_outputs[i]):
                            from_node_id, from_socket = node_outputs[i][0], node_outputs[i][1]
                            execution_list.add_strong_link(from_node_id, from_socket, unique_id)
                    cached_outputs.append((True, node_outputs))
            pending_subgraph_results[unique_id] = cached_outputs
            return (ExecutionResult.SLEEPING, None, None)
        outputs[unique_id] = output_data
    except comfy.model_management.InterruptProcessingException as iex:
        print("Processing interrupted")

        # skip formatting inputs/outputs
        error_details = {
            "node_id": real_node_id,
        }

        return (ExecutionResult.FAILURE, error_details, iex)
    except Exception as ex:
        typ, _, tb = sys.exc_info()
        exception_type = full_type_name(typ)
        input_data_formatted = {}
        if input_data_all is not None:
            input_data_formatted = {}
            for name, inputs in input_data_all.items():
                input_data_formatted[name] = [format_value(x) for x in inputs]

        output_data_formatted = {}
        for node_id, node_outputs in outputs.items():
            output_data_formatted[node_id] = [[format_value(x) for x in l] for l in node_outputs]

        print("!!! Exception during processing !!!")
        print(traceback.format_exc())

        error_details = {
            "node_id": real_node_id,
            "exception_message": str(ex),
            "exception_type": exception_type,
            "traceback": traceback.format_tb(tb),
            "current_inputs": input_data_formatted,
            "current_outputs": output_data_formatted
        }
        return (ExecutionResult.FAILURE, error_details, ex)

    executed.add(unique_id)

    return (ExecutionResult.SUCCESS, None, None)

def recursive_output_delete_if_changed(prompt, old_prompt, outputs, current_item):
    unique_id = current_item
    inputs = prompt[unique_id]['inputs']
    class_type = prompt[unique_id]['class_type']
    class_def = nodes.NODE_CLASS_MAPPINGS[class_type]

    is_changed_old = ''
    is_changed = ''
    to_delete = False
    if hasattr(class_def, 'IS_CHANGED'):
        if unique_id in old_prompt and 'is_changed' in old_prompt[unique_id]:
            is_changed_old = old_prompt[unique_id]['is_changed']
        if 'is_changed' not in prompt[unique_id]:
            context = extension_points.InputContext(unique_id, raw_prompt=prompt)
            input_data_all = get_input_data(inputs, class_def, context, outputs)
            if input_data_all is not None:
                try:
                    #is_changed = class_def.IS_CHANGED(**input_data_all)
                    is_changed = map_node_over_list(class_def, input_data_all, "IS_CHANGED")
                    prompt[unique_id]['is_changed'] = [None if isinstance(x, ExecutionBlocker) else x for x in is_changed]
                except:
                    to_delete = True
        else:
            is_changed = prompt[unique_id]['is_changed']

    if unique_id not in outputs:
        return True

    if not to_delete:
        if is_changed != is_changed_old:
            to_delete = True
        elif unique_id not in old_prompt:
            to_delete = True
        elif inputs == old_prompt[unique_id]['inputs']:
            for x in inputs:
                input_data = inputs[x]

                if is_link(input_data):
                    input_unique_id = input_data[0]
                    output_index = input_data[1]
                    if input_unique_id in outputs:
                        to_delete = recursive_output_delete_if_changed(prompt, old_prompt, outputs, input_unique_id)
                    else:
                        to_delete = True
                    if to_delete:
                        break
        else:
            to_delete = True

    if to_delete:
        d = outputs.pop(unique_id)
        del d
    return to_delete

class PromptExecutor:
    def __init__(self, server):
        self.outputs = {}
        self.object_storage = {}
        self.custom_outputs = {}
        self.old_prompt = {}
        self.server = server

    def handle_execution_error(self, prompt_id, prompt, current_outputs, executed, error, ex):
        node_id = error["node_id"]
        class_type = prompt[node_id]["class_type"]

        # First, send back the status to the frontend depending
        # on the exception type
        if isinstance(ex, comfy.model_management.InterruptProcessingException):
            mes = {
                "prompt_id": prompt_id,
                "node_id": node_id,
                "node_type": class_type,
                "executed": list(executed),
            }
            self.server.send_sync("execution_interrupted", mes, self.server.client_id)
        else:
            if self.server.client_id is not None:
                mes = {
                    "prompt_id": prompt_id,
                    "node_id": node_id,
                    "node_type": class_type,
                    "executed": list(executed),

                    "exception_message": error["exception_message"],
                    "exception_type": error["exception_type"],
                    "traceback": error["traceback"],
                    "current_inputs": error["current_inputs"],
                    "current_outputs": error["current_outputs"],
                }
                self.server.send_sync("execution_error", mes, self.server.client_id)

        # Next, remove the subsequent outputs since they will not be executed
        to_delete = []
        for o in self.outputs:
            if (o not in current_outputs) and (o not in executed):
                to_delete += [o]
                if o in self.old_prompt:
                    d = self.old_prompt.pop(o)
                    del d
        for o in to_delete:
            d = self.outputs.pop(o)
            del d

    def execute(self, prompt, prompt_id, extra_data={}, execute_outputs=[]):
        nodes.interrupt_processing(False)

        if "client_id" in extra_data:
            self.server.client_id = extra_data["client_id"]
        else:
            self.server.client_id = None

        if self.server.client_id is not None:
            self.server.send_sync("execution_start", { "prompt_id": prompt_id}, self.server.client_id)

        with torch.inference_mode():
            #delete cached outputs if nodes don't exist for them
            to_delete = []
            for o in self.outputs:
                if o not in prompt:
                    to_delete += [o]
            for o in to_delete:
                d = self.outputs.pop(o)
                del d
            to_delete = []
            for o in self.object_storage:
                if o[0] not in prompt:
                    to_delete += [o]
                else:
                    p = prompt[o[0]]
                    if o[1] != p['class_type']:
                        to_delete += [o]
            for o in to_delete:
                d = self.object_storage.pop(o)
                del d

            for x in prompt:
                recursive_output_delete_if_changed(prompt, self.old_prompt, self.outputs, x)

            current_outputs = set(self.outputs.keys())
            for x in list(self.custom_outputs.keys()):
                if x in current_outputs:
                    cached_outputs = self.custom_outputs[x]
                    for key, value in cached_outputs.items():
                        custom_output = extension_points.get_custom_output(key)
                        if custom_output is not None:
                            custom_output.on_cached_value_used(self.server, prompt_id, x, value)
                else:
                    d = self.custom_outputs.pop(x)
                    del d

            if self.server.client_id is not None:
                self.server.send_sync("execution_cached", { "nodes": list(current_outputs) , "prompt_id": prompt_id}, self.server.client_id)
            pending_subgraph_results = {}
            dynamic_prompt = DynamicPrompt(prompt)
            executed = set()
            execution_list = ExecutionList(dynamic_prompt, self.outputs)
            for node_id in list(execute_outputs):
                execution_list.add_node(node_id)

            while not execution_list.is_empty():
                node_id = execution_list.stage_node_execution()
                result, error, ex = non_recursive_execute(self.server, dynamic_prompt, self.outputs, node_id, extra_data, executed, prompt_id, self.custom_outputs, self.object_storage, execution_list, pending_subgraph_results)
                if result == ExecutionResult.FAILURE:
                    self.handle_execution_error(prompt_id, dynamic_prompt.original_prompt, current_outputs, executed, error, ex)
                    break
                elif result == ExecutionResult.SLEEPING:
                    execution_list.unstage_node_execution()
                else: # result == ExecutionResult.SUCCESS
                    execution_list.complete_node_execution()
                    real_node_id = dynamic_prompt.get_real_node_id(node_id)
                    parent_node_id = dynamic_prompt.get_parent_node_id(node_id)
                    display_node_id = dynamic_prompt.get_display_node_id(node_id)
                    # TODO - Code cleanup
                    if node_id in self.custom_outputs:
                        for key in self.custom_outputs[node_id]:
                            custom_output = extension_points.get_custom_output(key)
                            if real_node_id is not None and real_node_id != node_id:
                                assert real_node_id in self.custom_outputs
                                self.custom_outputs[real_node_id][key] = custom_output.accumulate_results_as_real(self.custom_outputs[real_node_id].get(key, None), self.custom_outputs[node_id][key])
                                if custom_output is not None:
                                    custom_output.on_value_updated(self.server, prompt_id, real_node_id, self.custom_outputs[real_node_id][key])
                            if parent_node_id is not None and parent_node_id != node_id:
                                assert parent_node_id in self.custom_outputs
                                self.custom_outputs[parent_node_id][key] = custom_output.accumulate_results_as_parent(self.custom_outputs[parent_node_id].get(key, None), self.custom_outputs[node_id][key])
                                if custom_output is not None:
                                    custom_output.on_value_updated(self.server, prompt_id, parent_node_id, self.custom_outputs[parent_node_id][key])
                            if display_node_id is not None and display_node_id != node_id:
                                assert display_node_id in self.custom_outputs
                                self.custom_outputs[display_node_id][key] = custom_output.accumulate_results_as_display(self.custom_outputs[display_node_id].get(key, None), self.custom_outputs[node_id][key])
                                if custom_output is not None:
                                    custom_output.on_value_updated(self.server, prompt_id, display_node_id, self.custom_outputs[display_node_id][key])
                    custom_node_outputs = self.custom_outputs.get(node_id, {})
                    for key in custom_node_outputs.keys():
                        custom_output = extension_points.get_custom_output(key)
                        if custom_output is not None:
                            custom_output.on_value_updated(self.server, prompt_id, node_id, custom_node_outputs[key])
                    if self.server.client_id is not None:
                        result_message = { "node": display_node_id, "prompt_id": prompt_id }
                        custom_node_outputs = self.custom_outputs.get(display_node_id, {})
                        for key in custom_node_outputs.keys():
                            custom_output = extension_points.get_custom_output(key)
                            if custom_output is not None and custom_output.include_in_executed:
                                result_message[custom_output.override_client_key] = custom_node_outputs[key]
                        self.server.send_sync("executed", result_message, self.server.client_id)

            for x in executed:
                if x in prompt:
                    self.old_prompt[x] = copy.deepcopy(prompt[x])
            self.server.last_node_id = None



def validate_inputs(prompt, item, validated):
    unique_id = item
    if unique_id in validated:
        return validated[unique_id]

    inputs = prompt[unique_id]['inputs']
    class_type = prompt[unique_id]['class_type']
    obj_class = nodes.NODE_CLASS_MAPPINGS[class_type]

    class_inputs = obj_class.INPUT_TYPES()
    valid_inputs = set(class_inputs.get('required',{})).union(set(class_inputs.get('optional',{})))

    errors = []
    valid = True

    for x in valid_inputs:
        type_input, input_category, extra_info = get_input_info(obj_class, x)
        if x not in inputs:
            if input_category == "required":
                error = {
                    "type": "required_input_missing",
                    "message": "Required input is missing",
                    "details": f"{x}",
                    "extra_info": {
                        "input_name": x
                    }
                }
                errors.append(error)
            continue

        val = inputs[x]
        info = (type_input, extra_info)
        if isinstance(val, list):
            if len(val) != 2:
                error = {
                    "type": "bad_linked_input",
                    "message": "Bad linked input, must be a length-2 list of [node_id, slot_index]",
                    "details": f"{x}",
                    "extra_info": {
                        "input_name": x,
                        "input_config": info,
                        "received_value": val
                    }
                }
                errors.append(error)
                continue

            o_id = val[0]
            o_class_type = prompt[o_id]['class_type']
            r = nodes.NODE_CLASS_MAPPINGS[o_class_type].RETURN_TYPES
            if type_input != "*" and r[val[1]] != "*" and r[val[1]] != type_input:
                received_type = r[val[1]]
                details = f"{x}, {received_type} != {type_input}"
                error = {
                    "type": "return_type_mismatch",
                    "message": "Return type mismatch between linked nodes",
                    "details": details,
                    "extra_info": {
                        "input_name": x,
                        "input_config": info,
                        "received_type": received_type,
                        "linked_node": val
                    }
                }
                errors.append(error)
                continue
            try:
                r = validate_inputs(prompt, o_id, validated)
                if r[0] is False:
                    # `r` will be set in `validated[o_id]` already
                    valid = False
                    continue
            except Exception as ex:
                typ, _, tb = sys.exc_info()
                valid = False
                exception_type = full_type_name(typ)
                reasons = [{
                    "type": "exception_during_inner_validation",
                    "message": "Exception when validating inner node",
                    "details": str(ex),
                    "extra_info": {
                        "input_name": x,
                        "input_config": info,
                        "exception_message": str(ex),
                        "exception_type": exception_type,
                        "traceback": traceback.format_tb(tb),
                        "linked_node": val
                    }
                }]
                validated[o_id] = (False, reasons, o_id)
                continue
        else:
            try:
                if type_input == "INT":
                    val = int(val)
                    inputs[x] = val
                if type_input == "FLOAT":
                    val = float(val)
                    inputs[x] = val
                if type_input == "STRING":
                    val = str(val)
                    inputs[x] = val
                if type_input == "BOOLEAN":
                    val = bool(val)
                    inputs[x] = val
            except Exception as ex:
                error = {
                    "type": "invalid_input_type",
                    "message": f"Failed to convert an input value to a {type_input} value",
                    "details": f"{x}, {val}, {ex}",
                    "extra_info": {
                        "input_name": x,
                        "input_config": info,
                        "received_value": val,
                        "exception_message": str(ex)
                    }
                }
                errors.append(error)
                continue

            if "min" in extra_info and val < extra_info["min"]:
                error = {
                    "type": "value_smaller_than_min",
                    "message": "Value {} smaller than min of {}".format(val, extra_info["min"]),
                    "details": f"{x}",
                    "extra_info": {
                        "input_name": x,
                        "input_config": info,
                        "received_value": val,
                    }
                }
                errors.append(error)
                continue
            if "max" in extra_info and val > extra_info["max"]:
                error = {
                    "type": "value_bigger_than_max",
                    "message": "Value {} bigger than max of {}".format(val, extra_info["max"]),
                    "details": f"{x}",
                    "extra_info": {
                        "input_name": x,
                        "input_config": info,
                        "received_value": val,
                    }
                }
                errors.append(error)
                continue

            if hasattr(obj_class, "VALIDATE_INPUTS"):
                context = extension_points.InputContext(unique_id, raw_prompt=prompt)
                input_data_all = get_input_data(inputs, obj_class, context)
                #ret = obj_class.VALIDATE_INPUTS(**input_data_all)
                ret = map_node_over_list(obj_class, input_data_all, "VALIDATE_INPUTS")
                for i, r in enumerate(ret):
                    if r is not True and not isinstance(r, ExecutionBlocker):
                        details = f"{x}"
                        if r is not False:
                            details += f" - {str(r)}"

                        error = {
                            "type": "custom_validation_failed",
                            "message": "Custom validation failed for node",
                            "details": details,
                            "extra_info": {
                                "input_name": x,
                                "input_config": info,
                                "received_value": val,
                            }
                        }
                        errors.append(error)
                        continue
            else:
                if isinstance(type_input, list):
                    if val not in type_input:
                        input_config = info
                        list_info = ""

                        # Don't send back gigantic lists like if they're lots of
                        # scanned model filepaths
                        if len(type_input) > 20:
                            list_info = f"(list of length {len(type_input)})"
                            input_config = None
                        else:
                            list_info = str(type_input)

                        error = {
                            "type": "value_not_in_list",
                            "message": "Value not in list",
                            "details": f"{x}: '{val}' not in {list_info}",
                            "extra_info": {
                                "input_name": x,
                                "input_config": input_config,
                                "received_value": val,
                            }
                        }
                        errors.append(error)
                        continue

    if len(errors) > 0 or valid is not True:
        ret = (False, errors, unique_id)
    else:
        ret = (True, [], unique_id)

    validated[unique_id] = ret
    return ret

def full_type_name(klass):
    module = klass.__module__
    if module == 'builtins':
        return klass.__qualname__
    return module + '.' + klass.__qualname__

def validate_prompt(prompt):
    outputs = set()
    for x in prompt:
        class_ = nodes.NODE_CLASS_MAPPINGS[prompt[x]['class_type']]
        if hasattr(class_, 'OUTPUT_NODE') and class_.OUTPUT_NODE == True:
            outputs.add(x)

    if len(outputs) == 0:
        error = {
            "type": "prompt_no_outputs",
            "message": "Prompt has no outputs",
            "details": "",
            "extra_info": {}
        }
        return (False, error, [], [])

    good_outputs = set()
    errors = []
    node_errors = {}
    validated = {}
    for o in outputs:
        valid = False
        reasons = []
        try:
            m = validate_inputs(prompt, o, validated)
            valid = m[0]
            reasons = m[1]
        except Exception as ex:
            typ, _, tb = sys.exc_info()
            valid = False
            exception_type = full_type_name(typ)
            reasons = [{
                "type": "exception_during_validation",
                "message": "Exception when validating node",
                "details": str(ex),
                "extra_info": {
                    "exception_type": exception_type,
                    "traceback": traceback.format_tb(tb)
                }
            }]
            validated[o] = (False, reasons, o)

        if valid is True:
            good_outputs.add(o)
        else:
            print(f"Failed to validate prompt for output {o}:")
            if len(reasons) > 0:
                print("* (prompt):")
                for reason in reasons:
                    print(f"  - {reason['message']}: {reason['details']}")
            errors += [(o, reasons)]
            for node_id, result in validated.items():
                valid = result[0]
                reasons = result[1]
                # If a node upstream has errors, the nodes downstream will also
                # be reported as invalid, but there will be no errors attached.
                # So don't return those nodes as having errors in the response.
                if valid is not True and len(reasons) > 0:
                    if node_id not in node_errors:
                        class_type = prompt[node_id]['class_type']
                        node_errors[node_id] = {
                            "errors": reasons,
                            "dependent_outputs": [],
                            "class_type": class_type
                        }
                        print(f"* {class_type} {node_id}:")
                        for reason in reasons:
                            print(f"  - {reason['message']}: {reason['details']}")
                    node_errors[node_id]["dependent_outputs"].append(o)
            print("Output will be ignored")

    if len(good_outputs) == 0:
        errors_list = []
        for o, errors in errors:
            for error in errors:
                errors_list.append(f"{error['message']}: {error['details']}")
        errors_list = "\n".join(errors_list)

        error = {
            "type": "prompt_outputs_failed_validation",
            "message": "Prompt outputs failed validation",
            "details": errors_list,
            "extra_info": {}
        }

        return (False, error, list(good_outputs), node_errors)

    return (True, None, list(good_outputs), node_errors)


class PromptQueue:
    def __init__(self, server):
        self.server = server
        self.mutex = threading.RLock()
        self.not_empty = threading.Condition(self.mutex)
        self.task_counter = 0
        self.queue = []
        self.currently_running = {}
        self.history = {}
        server.prompt_queue = self

    def put(self, item):
        with self.mutex:
            heapq.heappush(self.queue, item)
            self.server.queue_updated()
            self.not_empty.notify()

    def get(self):
        with self.not_empty:
            while len(self.queue) == 0:
                self.not_empty.wait()
            item = heapq.heappop(self.queue)
            i = self.task_counter
            self.currently_running[i] = copy.deepcopy(item)
            self.task_counter += 1
            self.server.queue_updated()
            return (item, i)

    def task_done(self, item_id, custom_outputs):
        with self.mutex:
            prompt = self.currently_running.pop(item_id)
            self.history[prompt[1]] = { "prompt": prompt }
            for key in extension_points.get_custom_outputs():
                custom_output = extension_points.get_custom_output(key)
                if custom_output.include_in_history:
                    self.history[prompt[1]][custom_output.override_history_key] = {}
                    for o in custom_outputs:
                        if key in custom_outputs[o]:
                            self.history[prompt[1]][custom_output.override_history_key][o] = custom_outputs[o][key]
            self.server.queue_updated()

    def get_current_queue(self):
        with self.mutex:
            out = []
            for x in self.currently_running.values():
                out += [x]
            return (out, copy.deepcopy(self.queue))

    def get_tasks_remaining(self):
        with self.mutex:
            return len(self.queue) + len(self.currently_running)

    def wipe_queue(self):
        with self.mutex:
            self.queue = []
            self.server.queue_updated()

    def delete_queue_item(self, function):
        with self.mutex:
            for x in range(len(self.queue)):
                if function(self.queue[x]):
                    if len(self.queue) == 1:
                        self.wipe_queue()
                    else:
                        self.queue.pop(x)
                        heapq.heapify(self.queue)
                    self.server.queue_updated()
                    return True
        return False

    def get_history(self, prompt_id=None):
        with self.mutex:
            if prompt_id is None:
                return copy.deepcopy(self.history)
            elif prompt_id in self.history:
                return {prompt_id: copy.deepcopy(self.history[prompt_id])}
            else:
                return {}

    def wipe_history(self):
        with self.mutex:
            self.history = {}

    def delete_history_item(self, id_to_delete):
        with self.mutex:
            self.history.pop(id_to_delete, None)
