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

class ExecutionResult(Enum):
    SUCCESS_EXECUTED = 0
    SUCCESS_RESOLVED_SUBGRAPH = 1
    SUCCESS_CACHED = 2
    FAILURE = 3
    SLEEPING_LAZY = 4
    SLEEPING_SUBGRAPH = 5

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

    def add_node(self, unique_id, include_lazy=False, subgraph_nodes=None):
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
                if subgraph_nodes is not None and from_node_id not in subgraph_nodes:
                    continue
                input_type, input_category, input_info = self.get_input_info(unique_id, input_name)
                is_lazy = "lazy" in input_info and input_info["lazy"]
                if include_lazy or not is_lazy:
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
    def __init__(self, dynprompt):
        super().__init__(dynprompt)
        self.executed = set()
        self.staged_node_id = None

    def add_strong_link(self, from_node_id, from_socket, to_node_id):
        if from_node_id in self.executed:
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
        self.executed.add(node_id)
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
        self.ephemeral_children = {}

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
        if parent_id not in self.ephemeral_children:
            self.ephemeral_children[parent_id] = []
        self.ephemeral_children[parent_id].append(node_id)

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

    def get_nodes_with_parent(self, parent_id):
        if parent_id is None:
            return [node_id for node_id in self.original_prompt]
        else:
            return self.ephemeral_children.get(parent_id, [])

    def copy_descendants(self, other, node_id):
        for child_id in other.get_nodes_with_parent(node_id):
            self.add_ephemeral_node(child_id, other.get_node(child_id), node_id, other.ephemeral_display.get(child_id, None))
            self.copy_descendants(other, child_id)

def get_input_data(inputs, class_def, unique_id, outputs={}, prompt={}, dynprompt=None, extra_data={}):
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
                input_data_all[x] = [prompt]
            if h[x] == "DYNPROMPT":
                input_data_all[x] = [dynprompt]
            if h[x] == "EXTRA_PNGINFO":
                if "extra_pnginfo" in extra_data:
                    input_data_all[x] = [extra_data['extra_pnginfo']]
            if h[x] == "UNIQUE_ID":
                input_data_all[x] = [unique_id]
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
    uis = []
    subgraph_results = []
    return_values = map_node_over_list(obj, input_data_all, obj.FUNCTION, allow_interrupt=True, execution_block_cb=execution_block_cb, pre_execute_cb=pre_execute_cb)
    has_subgraph = False
    for i in range(len(return_values)):
        r = return_values[i]
        if isinstance(r, dict):
            if 'ui' in r:
                uis.append(r['ui'])
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
    ui = dict()    
    if len(uis) > 0:
        ui = {k: [y for x in uis for y in x[k]] for k in uis[0].keys()}
    return output, ui, has_subgraph

def format_value(x):
    if x is None:
        return None
    elif isinstance(x, (int, float, bool, str)):
        return x
    else:
        return str(x)

def non_recursive_execute(server, dynprompt, outputs, current_item, extra_data, executed, prompt_id, outputs_ui, object_storage, execution_list, pending_subgraph_results, old_prompt):
    unique_id = current_item
    real_node_id = dynprompt.get_real_node_id(unique_id)
    display_node_id = dynprompt.get_display_node_id(unique_id)
    inputs = dynprompt.get_node(unique_id)['inputs']
    class_type = dynprompt.get_node(unique_id)['class_type']
    class_def = nodes.NODE_CLASS_MAPPINGS[class_type]
    if unique_id in outputs:
        print("--> Node {}: Using cached output".format(unique_id))
        return (ExecutionResult.SUCCESS_CACHED, None, None)

    input_data_all = None
    resolved_subgraph = False
    try:
        if unique_id in pending_subgraph_results:
            print("--> Node {}: Resolving subgraph".format(unique_id))
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
            output_ui = []
            has_subgraph = False
            resolved_subgraph = True
        else:
            print("--> Node {}: Executing".format(unique_id))
            input_data_all = get_input_data(inputs, class_def, unique_id, outputs, dynprompt.original_prompt, dynprompt, extra_data)
            if server.client_id is not None:
                server.last_node_id = display_node_id
                server.send_sync("executing", { "node": display_node_id, "prompt_id": prompt_id }, server.client_id)

            obj = object_storage.get(unique_id, None)
            if obj is None:
                obj = class_def()
                object_storage[unique_id] = obj

            if hasattr(obj, "check_lazy_status"):
                required_inputs = map_node_over_list(obj, input_data_all, "check_lazy_status", allow_interrupt=True)
                required_inputs = set(sum([r for r in required_inputs if isinstance(r,list)], []))
                required_inputs = [x for x in required_inputs if isinstance(x,str) and x not in input_data_all]
                if len(required_inputs) > 0:
                    for i in required_inputs:
                        execution_list.make_input_strong_link(unique_id, i)
                    return (ExecutionResult.SLEEPING_LAZY, None, None)

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
            output_data, output_ui, has_subgraph = get_output_data(obj, input_data_all, execution_block_cb=execution_block_cb, pre_execute_cb=pre_execute_cb)
        if len(output_ui) > 0:
            outputs_ui[unique_id] = output_ui
            if server.client_id is not None:
                server.send_sync("executed", { "node": display_node_id, "output": output_ui, "prompt_id": prompt_id }, server.client_id)
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
            return (ExecutionResult.SLEEPING_SUBGRAPH, None, None)
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

    result = ExecutionResult.SUCCESS_RESOLVED_SUBGRAPH if resolved_subgraph else ExecutionResult.SUCCESS_EXECUTED
    return (result, None, None)

def inputs_equal(a, b):
    if a.__class__ != b.__class__:
        return False
    if isinstance(a, list):
        if len(a) != len(b):
            return False
        for i in range(len(a)):
            if not inputs_equal(a[i], b[i]):
                return False
        return True
    elif isinstance(a, dict):
        if len(a) != len(b):
            return False
        for k in a.keys():
            if k not in b:
                return False
            if not inputs_equal(a[k], b[k]):
                return False
        return True
    elif isinstance(a, torch.Tensor):
        if a.shape != b.shape:
            return False
        return torch.all(a.eq(b)).item() == 1
    else:
        try:
            return a == b
        except:
            # We can't compare inputs -- assume they're different
            return False

def remove_stale_cache_entries(dynprompt, old_prompt, outputs, object_storage, dirty_set, parent_id=None):
    print("Removing stale cache entries with parent {}".format(parent_id))
    # Delete objects that are no longer here
    removed_set = set()
    to_delete = []
    old_subgraph_nodes = old_prompt.get_nodes_with_parent(parent_id)
    for node_id in old_subgraph_nodes:
        old_node = old_prompt.get_node(node_id)
        new_node = dynprompt.get_node(node_id)
        old_parent_id = old_prompt.get_parent_node_id(node_id)
        new_parent_id = dynprompt.get_parent_node_id(node_id)
        if old_node is not None and new_node is not None and old_parent_id != new_parent_id:
            print("Node {} has changed parents from {} to {}".format(node_id, old_parent_id, new_parent_id))
            removed_set.add(node_id)
        elif old_parent_id != parent_id:
            continue
        elif new_node is None:
            print("Node {} has been removed".format(node_id, old_parent_id, new_parent_id))
            removed_set.add(node_id)
        elif old_node["class_type"] != new_node["class_type"]:
            print("Node {} has changed types".format(node_id, old_parent_id, new_parent_id))
            removed_set.add(node_id)
    dirty_set.update(removed_set)
    for node_id in removed_set:
        d = outputs.pop(node_id, None)
        if d is not None:
            del d
        d = object_storage.pop(node_id, None)
        if d is not None:
            del d
        remove_stale_cache_entries(dynprompt, old_prompt, outputs, object_storage, dirty_set, parent_id=node_id)
        
    # Create a topological sort of new nodes
    subgraph_nodes = dynprompt.get_nodes_with_parent(parent_id)
    sorted = TopologicalSort(dynprompt)
    for node in subgraph_nodes:
        sorted.add_node(node, include_lazy=True, subgraph_nodes=subgraph_nodes)

    while not sorted.is_empty():
        ready_nodes = sorted.get_ready_nodes()
        print("Processing batch of {} ready nodes: {}".format(len(ready_nodes), ready_nodes))
        if len(ready_nodes) == 0:
            raise Exception("Circular dependency detected in subgraph")
        newly_dirty = set()
        for node_id in ready_nodes:
            print("Node {} checking dirty state".format(node_id))
            sorted.pop_node(node_id)
            node = dynprompt.get_node(node_id)
            old_node = old_prompt.get_node(node_id)
            inputs = node['inputs']
            class_type = node['class_type']
            class_def = nodes.NODE_CLASS_MAPPINGS[class_type]

            is_changed_old = ''
            is_changed = ''
            if hasattr(class_def, 'IS_CHANGED'):
                if old_node is not None and 'is_changed' in old_node:
                    is_changed_old = old_node['is_changed']
                if 'is_changed' not in node:
                    input_data_all = get_input_data(inputs, class_def, node_id, outputs)
                    if input_data_all is not None:
                        try:
                            #is_changed = class_def.IS_CHANGED(**input_data_all)
                            is_changed = map_node_over_list(class_def, input_data_all, "IS_CHANGED")
                            node['is_changed'] = [None if isinstance(x, ExecutionBlocker) else x for x in is_changed]
                        except:
                            newly_dirty.add(node_id)
                else:
                    is_changed = node['is_changed']

            if node_id not in outputs:
                print("Node {} is stale: not in outputs".format(node_id))
                newly_dirty.add(node_id)
            elif node_id not in newly_dirty:
                if not inputs_equal(is_changed, is_changed_old):
                    print("Node {} is stale: is_changed changed".format(node_id))
                    newly_dirty.add(node_id)
                elif not inputs_equal(inputs, old_node['inputs']):
                    print("Node {} is stale: inputs changed".format(node_id))
                    newly_dirty.add(node_id)
                else:
                    for x in inputs:
                        input_data = inputs[x]

                        if is_link(input_data):
                            input_unique_id = input_data[0]
                            if input_unique_id not in outputs:
                                print("Node {} is stale: input {} not in outputs".format(node_id, input_unique_id))
                                newly_dirty.add(node_id)
                                break
                            elif input_unique_id in dirty_set:
                                print("Node {} is stale: input {} is dirty".format(node_id, input_unique_id))
                                newly_dirty.add(node_id)
                                break
            if node_id in newly_dirty:
                d = outputs.pop(node_id, None)
                if d is not None:
                    del d
        dirty_set.update(newly_dirty)

class PromptExecutor:
    def __init__(self, server):
        self.outputs = {}
        self.object_storage = {}
        self.outputs_ui = {}
        self.old_prompt = DynamicPrompt({})
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

            dynamic_prompt = DynamicPrompt(prompt)
            dirty_set = set()
            remove_stale_cache_entries(dynamic_prompt, self.old_prompt, self.outputs, self.object_storage, dirty_set, parent_id=None)

            current_outputs = set(self.outputs.keys())
            for x in list(self.outputs_ui.keys()):
                if x not in current_outputs:
                    d = self.outputs_ui.pop(x)
                    del d

            comfy.model_management.cleanup_models()
            if self.server.client_id is not None:
                self.server.send_sync("execution_cached", { "nodes": list(current_outputs) , "prompt_id": prompt_id}, self.server.client_id)
            pending_subgraph_results = {}
            executed = set()
            execution_list = ExecutionList(dynamic_prompt)
            for node_id in list(execute_outputs):
                execution_list.add_node(node_id)

            while not execution_list.is_empty():
                node_id = execution_list.stage_node_execution()
                result, error, ex = non_recursive_execute(self.server, dynamic_prompt, self.outputs, node_id, extra_data, executed, prompt_id, self.outputs_ui, self.object_storage, execution_list, pending_subgraph_results, self.old_prompt)
                if result == ExecutionResult.SLEEPING_LAZY:
                    execution_list.unstage_node_execution()
                elif result == ExecutionResult.SUCCESS_RESOLVED_SUBGRAPH:
                    execution_list.complete_node_execution()
                elif result == ExecutionResult.SUCCESS_CACHED:
                    execution_list.complete_node_execution()
                    # Copy any children from the old dynamic prompt to the new one so we have access to the input for caching
                    dynamic_prompt.copy_descendants(self.old_prompt, node_id)
                else:
                    remove_stale_cache_entries(dynamic_prompt, self.old_prompt, self.outputs, self.object_storage, dirty_set, parent_id=node_id)
                    if result == ExecutionResult.FAILURE:
                        self.handle_execution_error(prompt_id, dynamic_prompt.original_prompt, current_outputs, executed, error, ex)
                        break
                    elif result == ExecutionResult.SLEEPING_SUBGRAPH:
                        execution_list.unstage_node_execution()
                    elif result == ExecutionResult.SUCCESS_EXECUTED:
                        execution_list.complete_node_execution()
                    else:
                        raise Exception("Unknown execution result")

            self.old_prompt = dynamic_prompt
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
                input_data_all = get_input_data(inputs, obj_class, unique_id)
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

    def task_done(self, item_id, outputs):
        with self.mutex:
            prompt = self.currently_running.pop(item_id)
            self.history[prompt[1]] = { "prompt": prompt, "outputs": {} }
            for o in outputs:
                self.history[prompt[1]]["outputs"][o] = outputs[o]
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
