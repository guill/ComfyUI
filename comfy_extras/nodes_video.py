from __future__ import annotations

import os
import av
import torch
import folder_paths
import json
from typing import Optional, Literal
from fractions import Fraction
from comfy.comfy_types import IO, FileLocator, ComfyNodeABC, VideoInput, AudioInput, ImageInput, VideoFromFile, VideoFromComponents
from comfy.cli_args import args

class SaveWEBM:
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.type = "output"
        self.prefix_append = ""

    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"images": ("IMAGE", ),
                     "filename_prefix": ("STRING", {"default": "ComfyUI"}),
                     "codec": (["vp9", "av1"],),
                     "fps": ("FLOAT", {"default": 24.0, "min": 0.01, "max": 1000.0, "step": 0.01}),
                     "crf": ("FLOAT", {"default": 32.0, "min": 0, "max": 63.0, "step": 1, "tooltip": "Higher crf means lower quality with a smaller file size, lower crf means higher quality higher filesize."}),
                     },
                "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
                }

    RETURN_TYPES = ()
    FUNCTION = "save_images"

    OUTPUT_NODE = True

    CATEGORY = "image/video"

    EXPERIMENTAL = True

    def save_images(self, images, codec, fps, filename_prefix, crf, prompt=None, extra_pnginfo=None):
        filename_prefix += self.prefix_append
        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(filename_prefix, self.output_dir, images[0].shape[1], images[0].shape[0])

        file = f"{filename}_{counter:05}_.webm"
        container = av.open(os.path.join(full_output_folder, file), mode="w")

        if prompt is not None:
            container.metadata["prompt"] = json.dumps(prompt)

        if extra_pnginfo is not None:
            for x in extra_pnginfo:
                container.metadata[x] = json.dumps(extra_pnginfo[x])

        codec_map = {"vp9": "libvpx-vp9", "av1": "libsvtav1"}
        stream = container.add_stream(codec_map[codec], rate=Fraction(round(fps * 1000), 1000))
        stream.width = images.shape[-2]
        stream.height = images.shape[-3]
        stream.pix_fmt = "yuv420p10le" if codec == "av1" else "yuv420p"
        stream.bit_rate = 0
        stream.options = {'crf': str(crf)}
        if codec == "av1":
            stream.options["preset"] = "6"

        for frame in images:
            frame = av.VideoFrame.from_ndarray(torch.clamp(frame[..., :3] * 255, min=0, max=255).to(device=torch.device("cpu"), dtype=torch.uint8).numpy(), format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        container.mux(stream.encode())
        container.close()

        results: list[FileLocator] = [{
            "filename": file,
            "subfolder": subfolder,
            "type": self.type
        }]

        return {"ui": {"images": results, "animated": (True,)}}  # TODO: frontend side

class LoadTestVideo(ComfyNodeABC):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_path": ("STRING", {"default": "", "tooltip": "The path to the video file."}),
            }
        }
    RETURN_TYPES = (IO.VIDEO,)
    FUNCTION = "load"

    CATEGORY = "_for_testing"
    DESCRIPTION = "Loads a video from disk."

    def load(self, video_path):
        return (VideoFromFile(video_path),)

class SaveVideo(ComfyNodeABC):
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.type: Literal["output"] = "output"
        self.prefix_append = ""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video": (IO.VIDEO, {"tooltip": "The video to save."}),
                "filename_prefix": ("STRING", {"default": "video/ComfyUI", "tooltip": "The prefix for the file to save. This may include formatting information such as %date:yyyy-MM-dd% or %Empty Latent Image.width% to include values from nodes."})
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO"
            },
        }

    RETURN_TYPES = ()
    FUNCTION = "save_video"

    OUTPUT_NODE = True

    CATEGORY = "image"
    DESCRIPTION = "Saves the input images to your ComfyUI output directory."

    def save_video(self, video: VideoInput, filename_prefix="video", prompt=None, extra_pnginfo=None):
        filename_prefix += self.prefix_append
        components = video.get_components()
        width = components.images.shape[2]
        height = components.images.shape[1]
        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(
            filename_prefix,
            self.output_dir,
            width,
            height
        )
        results: list[FileLocator] = list()
        saved_metadata = None
        if not args.disable_metadata:
            metadata = {}
            if extra_pnginfo is not None:
                metadata.update(extra_pnginfo)
            if prompt is not None:
                metadata["prompt"] = prompt
            if len(metadata) > 0:
                saved_metadata = metadata
        file = f"{filename}_{counter:05}_.mp4"
        video.save_to(os.path.join(full_output_folder, file), metadata=saved_metadata)

        results.append({
            "filename": file,
            "subfolder": subfolder,
            "type": self.type
        })
        counter += 1

        return { "ui": { "images": results, "animated": (True,) } }

class GetVideoFrames(ComfyNodeABC):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video": (IO.VIDEO, {"tooltip": "The video to get frames from."}),
            }
        }
    RETURN_TYPES = (IO.IMAGE,)
    FUNCTION = "get_frames"

    CATEGORY = "_for_testing"
    DESCRIPTION = "Get frames from a video."

    def get_frames(self, video: VideoInput):
        components = video.get_components()
        return (components.images,)

class GetVideoAudio(ComfyNodeABC):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video": (IO.VIDEO, {"tooltip": "The video to get frames from."}),
            }
        }
    RETURN_TYPES = (IO.AUDIO,)
    FUNCTION = "get_audio"

    CATEGORY = "_for_testing"
    DESCRIPTION = "Get audio from a video."

    def get_audio(self, video: VideoInput):
        components = video.get_components()
        return (components.audio,)

class GetVideoFramerate(ComfyNodeABC):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video": (IO.VIDEO, {"tooltip": "The video to get frames from."}),
            }
        }
    RETURN_TYPES = (IO.FLOAT,)
    FUNCTION = "get_framerate"

    CATEGORY = "_for_testing"
    DESCRIPTION = "Get the framerate of a video."

    def get_framerate(self, video: VideoInput):
        components = video.get_components()
        return (float(components.frame_rate),)

class CreateVideo(ComfyNodeABC):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": (IO.IMAGE, {"tooltip": "The images to create a video from."}),
                "fps": ("FLOAT", {"default": 30.0, "min": 1.0, "max": 120.0, "step": 1.0}),
            },
            "optional": {
                "audio": (IO.AUDIO, {"tooltip": "The audio to add to the video."}),
            }
        }

    RETURN_TYPES = (IO.VIDEO,)
    FUNCTION = "create_video"

    CATEGORY = "_for_testing"
    DESCRIPTION = "Create a video from images."

    def create_video(self, images: ImageInput, fps: float, audio: Optional[AudioInput] = None):
        return (VideoFromComponents(
            images=images,
            audio=audio,
            frame_rate=Fraction(fps),
        ),)

class GetVideoComponents(ComfyNodeABC):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video": (IO.VIDEO, {"tooltip": "The video to extract components from."}),
            }
        }
    RETURN_TYPES = (IO.IMAGE, IO.AUDIO, IO.FLOAT)
    RETURN_NAMES = ("images", "audio", "frame_rate")
    FUNCTION = "get_components"

    CATEGORY = "image/video"
    DESCRIPTION = "Extracts all components from a video: frames, audio, and framerate."

    def get_components(self, video: VideoInput):
        components = video.get_components()

        return (components.images, components.audio, float(components.frame_rate))
class GetVideoMetadata(ComfyNodeABC):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video": (IO.VIDEO, {"tooltip": "The video to get metadata from."}),
            }
        }
    RETURN_TYPES = (IO.STRING,)
    FUNCTION = "get_metadata"

    CATEGORY = "_for_testing"
    DESCRIPTION = "Get metadata from a video."

    def get_metadata(self, video: VideoInput):
        components = video.get_components()
        metadata = {}
        if components.metadata is not None:
            metadata.update(components.metadata)
        return (json.dumps(metadata),)

NODE_CLASS_MAPPINGS = {
    "SaveWEBM": SaveWEBM,
    "LoadTestVideo": LoadTestVideo,
    "SaveVideo": SaveVideo,
    "GetVideoFrames": GetVideoFrames,
    "GetVideoAudio": GetVideoAudio,
    "GetVideoFramerate": GetVideoFramerate,
    "CreateVideo": CreateVideo,
    "GetVideoComponents": GetVideoComponents,
    "GetVideoMetadata": GetVideoMetadata,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadTestVideo": "Load Test Video",
    "SaveVideo": "Save Video",
    "GetVideoFrames": "Get Video Frames",
    "GetVideoAudio": "Get Video Audio",
    "GetVideoFramerate": "Get Video Framerate",
    "CreateVideo": "Create Video",
    "GetVideoComponents": "Get Video Components",
    "GetVideoMetadata": "Get Video Metadata",
}
