from __future__ import annotations
from abc import ABC, abstractmethod
import av
import torch
from typing import Optional, TypedDict, NamedTuple
import numpy as np
import shutil
from fractions import Fraction
import json

ImageInput = torch.Tensor
"""
An image in format [B, H, W, C] where B is the batch size, C is the number of channels,
"""

class Dimensions(NamedTuple):
    width: int
    height: int

class AudioInput(TypedDict):
    """
    TypedDict representing audio input.
    """

    waveform: torch.Tensor
    """
    Tensor in the format [B, C, T] where B is the batch size, C is the number of channels,
    """

    sample_rate: int

class VideoInput(ABC):
    """
    Abstract base class for video input types.
    """

    @abstractmethod
    def get_images(self, begin_frame: int = 0, max_frames: Optional[int] = None) -> ImageInput:
        """
        Abstract method to get the image tensor.
        """
        pass

    @abstractmethod
    def get_audio(self) -> Optional[AudioInput]:
        """
        Abstract method to get the audio tensor.
        """
        pass

    @abstractmethod
    def save_to(self, path: str, metadata: Optional[dict] = None):
        """
        Abstract method to save the video input to a file.
        """
        pass

    @abstractmethod
    def get_frame_rate(self) -> Fraction:
        """
        Abstract method to get the frame rate of the video.
        """
        pass

    @abstractmethod
    def get_size(self) -> Dimensions:
        """
        Abstract method to get the size of the video.
        """
        pass

# TODO(Optimize) - Could maybe cache the container instead of reloading from disk each time?
class VideoFromFile(VideoInput):
    """
    Class representing video input from a file.
    """

    def __init__(self, path: str):
        self.path = path

    def get_images(self, begin_frame: int = 0, max_frames: Optional[int] = None) -> ImageInput:
        with av.open(self.path, mode='r') as container:
            container = av.open(self.path, mode='r')
            frames = []
            # TODO(Optimize) - Could optimize with `seek` probably.
            for i, frame in enumerate(container.decode(video=0)):
                if i < begin_frame:
                    continue
                if max_frames is not None and i >= max_frames:
                    break
                img = frame.to_ndarray(format='rgb24')  # shape: (H, W, 3)
                img = torch.from_numpy(img) / 255.0  # shape: (H, W, 3)
                frames.append(img)
            return torch.stack(frames) if len(frames) > 0 else torch.zeros(0, 3, 0, 0)
        return torch.zeros(0, 3, 0, 0)

    def get_audio(self) -> Optional[AudioInput]:
        with av.open(self.path, mode='r') as container:
            container = av.open(self.path, mode='r')
            audio_stream = next(s for s in container.streams if s.type == 'audio')
            if not isinstance(audio_stream, av.AudioStream):
                return None
            frames = []
            for frame in container.decode(audio=0):
                samples = frame.to_ndarray()  # shape: (channels, samples)
                frames.append(samples)
            if not frames:
                return None
            audio = np.concatenate(frames, axis=1)  # shape: (channels, total_samples)
            audio_tensor = torch.from_numpy(audio).unsqueeze(0)  # shape: (1, channels, total_samples)
            return {
                "waveform": audio_tensor,
                "sample_rate": int(audio_stream.sample_rate) if audio_stream.sample_rate else 1,
            }

    def save_to(self, path: str, metadata: Optional[dict] = None):
        # Just copy the file on disk
        shutil.copy(self.path, path)
        if metadata is not None:
            with av.open(path, mode='w') as container:
                    for key, value in metadata.items():
                        container.metadata[key] = json.dumps(value)

    def get_frame_rate(self) -> Fraction:
        container = av.open(self.path, mode='r')
        video_stream = next(s for s in container.streams if s.type == 'video')
        if not video_stream or not video_stream.average_rate:
            return Fraction(1) # Not sure what to do here
        return Fraction(video_stream.average_rate)

    def get_size(self) -> Dimensions:
        container = av.open(self.path, mode='r')
        video_stream = next(s for s in container.streams if s.type == 'video')
        if not isinstance(video_stream, av.VideoStream):
            raise ValueError("No video stream found in the file.")
        return Dimensions(width=video_stream.width, height=video_stream.height)

class VideoFromTensors(VideoInput):
    """
    Class representing video input from tensors.
    """

    def __init__(self, images: ImageInput, audio: Optional[AudioInput], frame_rate: Fraction):
        self.images = images
        self.audio = audio
        self.frame_rate = frame_rate

    def get_images(self, begin_frame: int = 0, max_frames: Optional[int] = None) -> ImageInput:
        if max_frames is not None:
            return self.images[begin_frame:begin_frame + max_frames]
        return self.images[begin_frame:]

    def get_audio(self) -> Optional[AudioInput]:
        return self.audio

    def save_to(self, path: str, metadata: Optional[dict] = None):
        if not path.endswith('.mp4'):
            raise ValueError("Output path must end with .mp4 (for now)")
        with av.open(path, mode='w') as output:
            # Create a video stream
            video_stream = output.add_stream('h264', rate=self.frame_rate)
            video_stream.width = self.images.shape[2]
            video_stream.height = self.images.shape[1]
            video_stream.pix_fmt = 'yuv420p'

            # Create an audio stream
            audio_sample_rate = 1
            audio_stream: Optional[av.AudioStream] = None
            if self.audio:
                # TODO(Tidy) - Do we have to create all streams first? If not, can be cleaned up
                audio_sample_rate = int(self.audio['sample_rate'])
                audio_stream = output.add_stream('aac', rate=audio_sample_rate)
                audio_stream.sample_rate = audio_sample_rate
                audio_stream.format = 'fltp'

            # Encode video
            for i, frame in enumerate(self.images):
                img = (frame * 255).clamp(0, 255).byte().cpu().numpy() # shape: (H, W, 3)
                frame = av.VideoFrame.from_ndarray(img, format='rgb24')
                frame = frame.reformat(format='yuv420p')  # Convert to YUV420P as required by h264
                packet = video_stream.encode(frame)
                output.mux(packet)

            # Flush video
            packet = video_stream.encode(None)
            output.mux(packet)

            if audio_stream and self.audio:
                # Encode audio
                samples_per_frame = int(audio_sample_rate / self.frame_rate)
                num_frames = self.audio['waveform'].shape[2] // samples_per_frame
                for i in range(num_frames):
                    start = i * samples_per_frame
                    end = start + samples_per_frame
                    # TODO(Feature) - Add support for stereo audio
                    chunk = self.audio['waveform'][0, 0, start:end].unsqueeze(0).numpy()
                    audio_frame = av.AudioFrame.from_ndarray(chunk, format='fltp', layout='mono')
                    audio_frame.sample_rate = audio_sample_rate
                    audio_frame.pts = i * samples_per_frame
                    for packet in audio_stream.encode(audio_frame):
                        output.mux(packet)

                # Flush audio
                for packet in audio_stream.encode(None):
                    output.mux(packet)

            # Add metadata
            if metadata is not None:
                for key, value in metadata.items():
                    output.metadata[key] = json.dumps(value)

    def get_frame_rate(self):
        return self.frame_rate

    def get_size(self) -> Dimensions:
        if self.images.shape[0] == 0:
            return Dimensions(0, 0)
        return Dimensions(self.images.shape[2], self.images.shape[1])
