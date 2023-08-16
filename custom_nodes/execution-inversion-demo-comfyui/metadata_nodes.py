from nodes import CheckpointLoaderSimple, LoraLoader
import extension_points
import json

class SetEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, set):
            return list(obj)
        return json.JSONEncoder.default(self, obj)

class MetadataOutput(extension_points.CustomOutput):
    def __init__(self):
        super().__init__("metadata")

    def merge_list_results(self, results):
        if len(results) == 0:
            return dict()
        elif len(results) == 1:
            return results[0]
        else:
            final_result = results[0]
            for result in results[1:]:
                final_result = extension_points.deep_merge(final_result, result)
            return final_result

    def accumulate_results_as_parent(self, prev_results, new_results):
        return extension_points.deep_merge(prev_results, new_results)

class GetMetadataSummary:
    def __init__(self):
        pass


    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
            },
            "hidden": {
                "context": "CONTEXT",
            },
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "get_metadata"

    CATEGORY = "InversionDemo Nodes/Debug"

    def get_metadata(self, image, context=None):
        result = {}
        ancestors = context.get_ancestor_nodes()
        for ancestor in ancestors:
            metadata = context.get_custom_output("metadata", ancestor)
            if metadata is not None:
                result = extension_points.deep_merge(result, metadata)

        return (json.dumps(result, indent=4,cls=SetEncoder),)


# Example hooked nodes to attach extra metadata
class HookedCheckpointLoaderSimple(CheckpointLoaderSimple):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    FUNCTION="hooked_load"

    def hooked_load(self, *args, **kwargs):
        result = getattr(super(), super().FUNCTION)(*args, **kwargs)
        metadata = {
            "checkpoints": set([kwargs["ckpt_name"]]),
        }
        if isinstance(result, dict):
            result["metadata"] = metadata
            return result
        else:
            return {
                "result": result,
                "metadata": metadata,
            }

class HookedLoraLoader(LoraLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    FUNCTION="hooked_load"

    def hooked_load(self, *args, **kwargs):
        result = getattr(super(), super().FUNCTION)(*args, **kwargs)
        metadata = {
            "loras": [kwargs["lora_name"]],
        }
        if isinstance(result, dict):
            result["metadata"] = metadata
            return result
        else:
            return {
                "result": result,
                "metadata": metadata,
            }

extension_points.register_custom_output(MetadataOutput())
extension_points.register_node(HookedCheckpointLoaderSimple, "CheckpointLoaderSimple", "Load Checkpoint")
extension_points.register_node(HookedLoraLoader, "LoraLoader", "Load LoRA")
extension_points.register_node(GetMetadataSummary, "GetMetadataSummary", "Get Metadata Summary")
