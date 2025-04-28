from __future__ import annotations
from abc import ABC, abstractmethod
import av
import torch
from typing import Optional, TypedDict, NamedTuple
import numpy as np
import shutil
from fractions import Fraction
import json
import io
from dataclasses import dataclass
from av.subtitles.stream import SubtitleStream

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

@dataclass
class VideoComponents:
    """
    Dataclass representing the components of a video.
    """

    images: ImageInput
    frame_rate: Fraction
    audio: Optional[AudioInput] = None
    metadata: Optional[dict] = None

class VideoInput(ABC):
    """
    Abstract base class for video input types.
    """

    @abstractmethod
    def get_components(self) -> VideoComponents:
        """
        Abstract method to get the video components (images, audio, and frame rate).
        
        Returns:
            VideoComponents containing images, audio, and frame rate
        """
        pass
    
    @abstractmethod
    def save_to(self, path: str, metadata: Optional[dict] = None):
        """
        Abstract method to save the video input to a file.
        """
        pass

class VideoFromFile(VideoInput):
    """
    Class representing video input from a file.
    """

    def __init__(self, file: str | io.BytesIO):
        """
        Initialize the VideoFromFile object based off of either a path on disk or a BytesIO object
        containing the file contents.
        """
        self.file = file

    def get_components(self) -> VideoComponents:
        with av.open(self.file, mode='r') as container:
            # Get video frames
            frames = []
            for frame in container.decode(video=0):
                img = frame.to_ndarray(format='rgb24')  # shape: (H, W, 3)
                img = torch.from_numpy(img) / 255.0  # shape: (H, W, 3)
                frames.append(img)
            
            images = torch.stack(frames) if len(frames) > 0 else torch.zeros(0, 3, 0, 0)
            
            # Get frame rate
            video_stream = next(s for s in container.streams if s.type == 'video')
            frame_rate = Fraction(video_stream.average_rate) if video_stream and video_stream.average_rate else Fraction(1)
            
            # Get audio if available
            audio = None
            try:
                audio_stream = next(s for s in container.streams if s.type == 'audio')
                if isinstance(audio_stream, av.AudioStream):
                    audio_frames = []
                    for frame in container.decode(audio=0):
                        samples = frame.to_ndarray()  # shape: (channels, samples)
                        audio_frames.append(samples)
                    if audio_frames:
                        audio_data = np.concatenate(audio_frames, axis=1)  # shape: (channels, total_samples)
                        audio_tensor = torch.from_numpy(audio_data).unsqueeze(0)  # shape: (1, channels, total_samples)
                        audio = AudioInput({
                            "waveform": audio_tensor,
                            "sample_rate": int(audio_stream.sample_rate) if audio_stream.sample_rate else 1,
                        })
            except StopIteration:
                pass  # No audio stream
                
            metadata = container.metadata
            return VideoComponents(images=images, audio=audio, frame_rate=frame_rate, metadata=metadata)
        raise ValueError(f"No video stream found in file '{self.file}'")

    def save_to(self, path: str, metadata: Optional[dict] = None):
        with av.open(self.file, mode='r') as container:
            streams = container.streams
            print(f"Saving to {path}")
            with av.open(path, mode='w') as output_container:
                # Copy over the original metadata
                for key, value in container.metadata.items():
                    if metadata is None or key not in metadata:
                        output_container.metadata[key] = value

                # Add our new metadata
                if metadata is not None:
                    for key, value in metadata.items():
                        if isinstance(value, str):
                            output_container.metadata[key] = value
                        else:
                            output_container.metadata[key] = json.dumps(value)

                # Add streams to the new container
                for stream in streams:
                    assert isinstance(stream, (av.VideoStream, av.AudioStream, SubtitleStream))
                    out_stream = output_container.add_stream_from_template(template = stream, opaque = True)

                    # Write packets to the new container
                    for packet in container.demux(stream):
                        # Skip the "flush" packets that `demux` can produce
                        if packet.dts is None:
                            continue
                        packet.stream = out_stream
                        output_container.mux(packet)

        print(f"Saved to {path}")

class VideoFromComponents(VideoInput):
    """
    Class representing video input from tensors.
    """

    def __init__(self, images: ImageInput, audio: Optional[AudioInput], frame_rate: Fraction):
        self.images = images
        self.audio = audio
        self.frame_rate = frame_rate

    def get_components(self) -> VideoComponents:
        return VideoComponents(images=self.images, audio=self.audio, frame_rate=self.frame_rate)

    def save_to(self, path: str, metadata: Optional[dict] = None):
        if not path.endswith('.mp4'):
            raise ValueError("Output path must end with .mp4 (for now)")
        with av.open(path, mode='w', options={'movflags': 'use_metadata_tags'}) as output:
            # Add metadata before writing any streams
            if metadata is not None:
                for key, value in metadata.items():
                    output.metadata[key] = json.dumps(value)

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

