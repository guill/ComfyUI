import itertools
from typing import Sequence, Mapping

import nodes

from comfy.graph_utils import is_link

class CacheKeySet:
    def __init__(self, dynprompt, node_ids, is_changed_cache):
        self.keys = {}
        self.subcache_keys = {}

    def all_node_ids(self):
        return set(self.keys.keys())

    def get_used_keys(self):
        return self.keys.values()

    def get_used_subcache_keys(self):
        return self.subcache_keys.values()

    def get_data_key(self, node_id):
        if node_id in self.keys:
            return self.keys[node_id]
        else:
            return None

    def get_subcache_key(self, node_id):
        if node_id in self.subcache_keys:
            return self.subcache_keys[node_id]
        else:
            return None

class Unhashable:
    def __init__(self):
        self.value = float("NaN")

def to_hashable(obj):
    # So that we don't infinitely recurse since frozenset and tuples
    # are Sequences.
    if isinstance(obj, (int, float, str, bool, type(None))):
        return obj
    elif isinstance(obj, Mapping):
        return frozenset([(to_hashable(k), to_hashable(v)) for k, v in sorted(obj.items())])
    elif isinstance(obj, Sequence):
        return frozenset(zip(itertools.count(), [to_hashable(i) for i in obj]))
    else:
        # TODO - Support other objects like tensors?
        return Unhashable()

class CacheKeySetID(CacheKeySet):
    def __init__(self, dynprompt, node_ids, is_changed_cache):
        super().__init__(dynprompt, node_ids, is_changed_cache)
        for node_id in node_ids:
            node = dynprompt.get_node(node_id)
            self.keys[node_id] = (node_id, node["class_type"])
            self.subcache_keys[node_id] = (node_id, node["class_type"])

class CacheKeySetInputSignature(CacheKeySet):
    def __init__(self, dynprompt, node_ids, is_changed_cache):
        super().__init__(dynprompt, node_ids, is_changed_cache)
        self.is_changed_cache = is_changed_cache
        for node_id in node_ids:
            node = dynprompt.get_node(node_id)
            self.keys[node_id] = self.get_node_signature(dynprompt, node_id)
            self.subcache_keys[node_id] = (node_id, node["class_type"])

    def get_node_signature(self, dynprompt, node_id):
        signature = []
        ancestors, order_mapping = self.get_ordered_ancestry(dynprompt, node_id)
        signature.append(self.get_immediate_node_signature(dynprompt, node_id, order_mapping))
        for ancestor_id in ancestors:
            signature.append(self.get_immediate_node_signature(dynprompt, ancestor_id, order_mapping))
        return to_hashable(signature)

    def get_immediate_node_signature(self, dynprompt, node_id, ancestor_order_mapping):
        node = dynprompt.get_node(node_id)
        class_type = node["class_type"]
        class_def = nodes.NODE_CLASS_MAPPINGS[class_type]
        signature = [class_type, self.is_changed_cache.get(node_id)]
        if hasattr(class_def, "NOT_IDEMPOTENT") and class_def.NOT_IDEMPOTENT:
            signature.append(node_id)
        inputs = node["inputs"]
        for key in sorted(inputs.keys()):
            if is_link(inputs[key]):
                (ancestor_id, ancestor_socket) = inputs[key]
                ancestor_index = ancestor_order_mapping[ancestor_id]
                signature.append((key,("ANCESTOR", ancestor_index, ancestor_socket)))
            else:
                signature.append((key, inputs[key]))
        return signature

    # This function returns a list of all ancestors of the given node. The order of the list is
    # deterministic based on which specific inputs the ancestor is connected by.
    def get_ordered_ancestry(self, dynprompt, node_id):
        ancestors = []
        order_mapping = {}
        self.get_ordered_ancestry_internal(dynprompt, node_id, ancestors, order_mapping)
        return ancestors, order_mapping

    def get_ordered_ancestry_internal(self, dynprompt, node_id, ancestors, order_mapping):
        inputs = dynprompt.get_node(node_id)["inputs"]
        input_keys = sorted(inputs.keys())
        for key in input_keys:
            if is_link(inputs[key]):
                ancestor_id = inputs[key][0]
                if ancestor_id not in order_mapping:
                    ancestors.append(ancestor_id)
                    order_mapping[ancestor_id] = len(ancestors) - 1
                    self.get_ordered_ancestry_internal(dynprompt, ancestor_id, ancestors, order_mapping)


class BasicCache:
    def __init__(self, key_class):
        self.key_class = key_class
        self.dynprompt = None
        self.cache_key_set = None
        self.cache = {}
        self.subcaches = {}

    def set_prompt(self, dynprompt, node_ids, is_changed_cache):
        self.dynprompt = dynprompt
        self.cache_key_set = self.key_class(dynprompt, node_ids, is_changed_cache)
        self.is_changed_cache = is_changed_cache

    def all_node_ids(self):
        assert self.cache_key_set is not None
        node_ids = self.cache_key_set.all_node_ids()
        for subcache in self.subcaches.values():
            node_ids = node_ids.union(subcache.all_node_ids())
        return node_ids

    def clean_unused(self):
        assert self.cache_key_set is not None
        preserve_keys = set(self.cache_key_set.get_used_keys())
        preserve_subcaches = set(self.cache_key_set.get_used_subcache_keys())
        to_remove = []
        for key in self.cache:
            if key not in preserve_keys:
                to_remove.append(key)
        for key in to_remove:
            del self.cache[key]

        to_remove = []
        for key in self.subcaches:
            if key not in preserve_subcaches:
                to_remove.append(key)
        for key in to_remove:
            del self.subcaches[key]

    def set_immediate(self, node_id, value):
        assert self.cache_key_set is not None
        cache_key = self.cache_key_set.get_data_key(node_id)
        self.cache[cache_key] = value

    def get_immediate(self, node_id):
        assert self.cache_key_set is not None
        cache_key = self.cache_key_set.get_data_key(node_id)
        if cache_key in self.cache:
            return self.cache[cache_key]
        else:
            return None

    def ensure_subcache(self, node_id, children_ids):
        assert self.cache_key_set is not None
        subcache_key = self.cache_key_set.get_subcache_key(node_id)
        subcache = self.subcaches.get(subcache_key, None)
        if subcache is None:
            subcache = BasicCache(self.key_class)
            self.subcaches[subcache_key] = subcache
        subcache.set_prompt(self.dynprompt, children_ids, self.is_changed_cache)
        return subcache

    def get_subcache(self, node_id):
        assert self.cache_key_set is not None
        subcache_key = self.cache_key_set.get_subcache_key(node_id)
        if subcache_key in self.subcaches:
            return self.subcaches[subcache_key]
        else:
            return None

class HierarchicalCache(BasicCache):
    def __init__(self, key_class):
        super().__init__(key_class)

    def get_cache_for(self, node_id):
        parent_id = self.dynprompt.get_parent_node_id(node_id)
        if parent_id is None:
            return self

        hierarchy = []
        while parent_id is not None:
            hierarchy.append(parent_id)
            parent_id = self.dynprompt.get_parent_node_id(parent_id)

        cache = self
        for parent_id in reversed(hierarchy):
            cache = cache.get_subcache(parent_id)
            if cache is None:
                return None
        return cache

    def get(self, node_id):
        cache = self.get_cache_for(node_id)
        if cache is None:
            return None
        return cache.get_immediate(node_id)

    def set(self, node_id, value):
        cache = self.get_cache_for(node_id)
        assert cache is not None
        cache.set_immediate(node_id, value)

    def ensure_subcache_for(self, node_id, children_ids):
        cache = self.get_cache_for(node_id)
        assert cache is not None
        return cache.ensure_subcache(node_id, children_ids)

