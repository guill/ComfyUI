from typing import Type, List, NamedTuple, Optional
from comfy_api.internal.singleton import ProxiedSingleton
from packaging import version as packaging_version

class ComfyAPIBase(ProxiedSingleton):
    def __init__(self):
        pass

class ComfyAPIWithVersion(NamedTuple):
    version: str
    api_class: Type[ComfyAPIBase]

def parse_version(version_str: str) -> packaging_version.Version:
    """
    Parses a version string into a packaging_version.Version object.
    Raises ValueError if the version string is invalid.
    """
    if version_str == "latest":
        return packaging_version.parse("9999999.9999999.9999999")
    return packaging_version.parse(version_str)

registered_versions: List[ComfyAPIWithVersion] = []
def register_versions(versions: List[ComfyAPIWithVersion]):
    versions.sort(key=lambda x: parse_version(x.version))
    global registered_versions
    registered_versions = versions

def get_all_versions() -> List[ComfyAPIWithVersion]:
    """
    Returns a list of all registered ComfyAPI versions.
    """
    return registered_versions

def get_best_version(requested_version: str) -> Type[ComfyAPIBase]:
    """
    Returns the best matching version of ComfyAPI based on the requested version.
    If no exact match is found, it returns the latest version that is greater than the requested version.
    The reason for this is to allow custom node authors to specify the "next" version for bleeding edge features
    """
    requested = parse_version(requested_version)
    for version in registered_versions:
        found = parse_version(version.version)
        if found >= requested:
            return version.api_class
    return registered_versions[-1].api_class  # Return the latest version if no match found
