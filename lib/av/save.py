import av
import torch
import numpy as np
from fractions import Fraction

def save(images=None, masks=None, audio=None, fps=30, path="chunker_save"):
    if images is None and masks is None and audio is None:
        raise ValueError("At least one of images, masks, or audio must be provided.")

    path = f"{path}.mov"

    with av.open(path, mode='w') as container:
        # Video stream setup
        if images is not None or masks is not None:
            count = max(
                images.shape[0] if images is not None else 0,
                masks.shape[0] if masks is not None else 0,
            )
            W = images.shape[2] if images is not None else (masks.shape[2] if masks is not None else None)
            H = images.shape[1] if images is not None else (masks.shape[1] if masks is not None else None)

            video_stream = container.add_stream('prores_ks', rate=Fraction(fps))
            video_stream.pix_fmt = 'yuva444p10le'
            video_stream.width = W
            video_stream.height = H

        # Audio stream setup
        if audio is not None:
            waveform = audio["waveform"]
            sample_rate = audio["sample_rate"]
            audio_ndarray = waveform.squeeze(0).cpu().numpy().astype(np.float32)
            num_channels = audio_ndarray.shape[0]
            layout = 'mono' if num_channels == 1 else 'stereo'
            audio_stream = container.add_stream('alac', rate=int(sample_rate))

        # Combine images and masks into RGBA format and write to video stream
        if images is not None or masks is not None:
            for i in range(count):
                img = images[i] if images is not None and images[i] is not None else torch.full((H, W, 3), 0.5)
                mask = masks[i] if masks is not None and masks[i] is not None else torch.full((H, W, 1), 1)
                img = (img * 255).cpu().numpy().astype(np.uint8)
                mask = (mask * 255).cpu().numpy().astype(np.uint8)
                if mask.ndim == 2: mask = mask.reshape((H, W, 1))
                rgba = np.concatenate([img, mask], axis=2)
                frame = av.VideoFrame.from_ndarray(rgba, format='rgba')
                for packet in video_stream.encode(frame):
                    container.mux(packet)
            
            # Flush the video stream encoders
            for packet in video_stream.encode():
                container.mux(packet)

        # Write Audio to audio stream
        if audio is not None:
            frame = av.AudioFrame.from_ndarray(audio_ndarray, format='fltp', layout=layout)
            frame.sample_rate = sample_rate
            for packet in audio_stream.encode(frame):
                container.mux(packet)

            # Flush the audio stream encoders
            for packet in audio_stream.encode():
                container.mux(packet)

    return path
