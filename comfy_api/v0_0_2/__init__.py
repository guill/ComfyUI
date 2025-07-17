from comfy_api.latest import ComfyAPI_latest
from typing import Type, TYPE_CHECKING
from comfy_api.internal.async_to_sync import create_sync_class

class ComfyAPIAdapter_v0_0_2(ComfyAPI_latest):
    VERSION = "0.0.2"
    STABLE = False

ComfyAPI = ComfyAPIAdapter_v0_0_2

# Create a synchronous version of the API
if TYPE_CHECKING:
    import comfy_api.ComfyAPISyncStub.ComfyAPISyncStub  # type: ignore

    ComfyAPISync: Type[comfy_api.v0_0_2.ComfyAPISyncStub.ComfyAPISyncStub]  # type: ignore
ComfyAPISync = create_sync_class(ComfyAPIAdapter_v0_0_2)
