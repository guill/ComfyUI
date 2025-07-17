from typing import Type, TYPE_CHECKING
from comfy_api.internal import ComfyAPIBase
from comfy_api.internal.singleton import ProxiedSingleton
from comfy_api.internal.async_to_sync import create_sync_class
from comfy_api.latest.input import ImageInput
from comfy_execution.progress import get_progress_state, PreviewImageTuple
from PIL import Image
from comfy.cli_args import args
import numpy as np


class ComfyAPI_latest(ComfyAPIBase):
    VERSION = "latest"
    STABLE = False

    class Execution(ProxiedSingleton):
        async def set_progress(
            self,
            node_id: str,
            value: float,
            max_value: float,
            preview_image: PreviewImageTuple | Image.Image | ImageInput | None = None,
        ) -> None:
            """
            Update the progress bar displayed in the ComfyUI interface.

            This function allows custom nodes and API calls to report their progress
            back to the user interface, providing visual feedback during long operations.

            Migration from previous API: comfy.utils.PROGRESS_BAR_HOOK
            """
            # Convert preview_image to PreviewImageTuple if needed
            if preview_image is not None:
                # First convert to PIL Image if needed
                if isinstance(preview_image, ImageInput):
                    # Convert ImageInput (torch.Tensor) to PIL Image
                    # Handle tensor shape [B, H, W, C] -> get first image if batch
                    tensor = preview_image
                    if len(tensor.shape) == 4:
                        tensor = tensor[0]

                    # Convert to numpy array and scale to 0-255
                    image_np = (tensor.cpu().numpy() * 255).astype(np.uint8)
                    preview_image = Image.fromarray(image_np)

                if isinstance(preview_image, Image.Image):
                    # Detect image format from PIL Image
                    image_format = preview_image.format if preview_image.format else "JPEG"
                    preview_image = (image_format, preview_image, args.preview_size)

            get_progress_state().update_progress(
                node_id=node_id,
                value=value,
                max_value=max_value,
                image=preview_image,
            )

    execution: Execution


ComfyAPI = ComfyAPI_latest

# Create a synchronous version of the API
if TYPE_CHECKING:
    import comfy_api.latest.ComfyAPISyncStub  # type: ignore

    ComfyAPISync: Type[comfy_api.latest.ComfyAPISyncStub.ComfyAPISyncStub]  # type: ignore
ComfyAPISync = create_sync_class(ComfyAPI_latest)

