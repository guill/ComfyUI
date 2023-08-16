from comfy.graph_utils import is_link
import copy

def register_node(node_class, class_name, display_name):
    from nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
    NODE_CLASS_MAPPINGS[class_name] = node_class
    NODE_DISPLAY_NAME_MAPPINGS[class_name] = display_name

# Merges two dictionaries or lists. Dictionaries are merged recursively, while lists are merged by concatenating them.
def deep_merge(merge_to, merge_from):
    if merge_from is None:
        return merge_to
    if merge_to is None:
        return copy.deepcopy(merge_from)
    for k in merge_from:
        if k in merge_to:
            if isinstance(merge_to[k], set) and isinstance(merge_from[k], set):
                merge_to[k] = merge_to[k].union(merge_from[k])
            elif isinstance(merge_to[k], list) and isinstance(merge_from[k], list):
                merge_to[k] = merge_to[k] + merge_from[k]
            elif isinstance(merge_to[k], dict) and isinstance(merge_from[k], dict):
                merge_to[k] = deep_merge(merge_to[k], merge_from[k])
            else:
                merge_to[k] = copy.deepcopy(merge_from[k])
        else:
            merge_to[k] = copy.deepcopy(merge_from[k])
    return merge_to

class CustomOutput:
    def __init__(self, key):
        self.key = key
        self.include_in_executed = False
        self.include_in_history = False
        self.override_client_key = key
        self.override_history_key = key

    def merge_list_results(self, results):
        # The default operation for merging multiple results (i.e. for lists) is merging all keys
        if len(results) > 0:
            return {k: [y for x in results for y in x[k]] for k in results[0].keys()}
        else:
            return dict()

    # This method is called when an ephemeral node expanded from this node has finished executing
    # and contains custom outputs of this type.
    def accumulate_results_as_parent(self, prev_results, new_results):
        # The default behavior is to ignore child results
        return prev_results

    # This method is called when this node is non-ephemeral and an ephemeral mode that is a descendant of this node
    # (at any level) has finished executing and contains custom outputs of this type.
    def accumulate_results_as_real(self, prev_results, new_results):
        # The default behavior is to ignore descendent results
        return prev_results

    # This method is called when an ephemeral node that has this node as its display node has finished executing
    # and contains custom outputs of this type.
    def accumulate_results_as_display(self, prev_results, new_results):
        # The default behavior is to ignore child results
        return prev_results

    def on_value_updated(self, server, prompt_id, node_id, new_value):
        pass

    def on_cached_value_used(self, server, prompt_id, node_id, new_value):
        pass

class CustomOutputSendToClient(CustomOutput):
    def __init__(self, key):
        super().__init__(key)

    def send_value(self, server, prompt_id, node_id, value):
        if server.client_id is None:
            return
        server.send_sync("custom_output", {
            "node": node_id,
            "prompt_id": prompt_id,
            "key": self.key,
            "value": value
        }, server.client_id)

    def on_value_updated(self, server, prompt_id, node_id, new_value):
        self.send_value(server, prompt_id, node_id, new_value)

    def on_cached_value_used(self, server, prompt_id, node_id, cached_value):
        self.send_value(server, prompt_id, node_id, cached_value)


CUSTOM_OUTPUTS = {}

def register_custom_output(custom_output):
    CUSTOM_OUTPUTS[custom_output.key] = custom_output

def get_custom_output(key):
    return CUSTOM_OUTPUTS[key]

def get_custom_outputs():
    return CUSTOM_OUTPUTS.keys()

# Access this object with the hidden input name "CONTEXT"
class InputContext():
    def __init__(self, unique_id, dynprompt=None, raw_prompt=None, extra_data={}, custom_outputs={}):
        self.unique_id = unique_id
        self.dynprompt = dynprompt
        self.extra_data = extra_data
        self.raw_prompt = raw_prompt
        self.custom_outputs = custom_outputs

    def get_current_node(self):
        return self.unique_id

    def get_raw_prompt(self):
        return self.dynprompt.original_prompt if self.dynprompt is not None else self.raw_prompt

    def get_dynamic_prompt(self):
        return self.dynprompt

    def get_png_info(self):
        return self.extra_data.get('extra_pnginfo', None)

    def get_custom_output(self, output_key, node_id):
        if node_id in self.custom_outputs:
            return self.custom_outputs[node_id].get(output_key, None)

    def get_ancestor_nodes(self, node_id=None, add_to_set=None):
        if self.dynprompt is None:
            return None
        if add_to_set is None:
            add_to_set = set()
        if node_id is None:
            node_id = self.unique_id

        inputs = self.dynprompt.get_node(node_id).get("inputs", {})
        for k, v in inputs.items():
            if is_link(v):
                if v[0] not in add_to_set:
                    add_to_set.add(v[0])
                    self.get_ancestor_nodes(v[0], add_to_set)

        # Parent node ancestors implicitly count as our ancestors, but the parent itself doesn't
        parent_id = self.dynprompt.get_parent_node_id(node_id)
        if parent_id is not None and parent_id not in add_to_set:
            self.get_ancestor_nodes(parent_id, add_to_set)

        return add_to_set
