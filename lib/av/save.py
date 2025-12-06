import av
import numpy as np
from fractions import Fraction

def save(images, masks, audio, fps, filename_prefix):
    path = f"{filename_prefix}.mov"
    B, H, W, C = images.shape

    with av.open(path, mode='w') as container:
        # Video Stream Setup
        video_stream = container.add_stream('prores_ks', rate=Fraction(fps))
        video_stream.pix_fmt = 'yuva444p10le'
        video_stream.width = W
        video_stream.height = H

        # Audio Stream Setup
        waveform = audio["waveform"] 
        sample_rate = audio["sample_rate"]
        audio_ndarray = waveform.squeeze(0).cpu().numpy().astype(np.float32)
        num_channels = audio_ndarray.shape[0]
        layout = 'mono' if num_channels == 1 else 'stereo'
        audio_stream = container.add_stream('alac', rate=int(sample_rate))

        # Combine images and masks into RGBA format and write to video stream
        for i in range(B):
            img = images[i]
            mask = masks[i]
            img = (img * 255).cpu().numpy().astype(np.uint8)
            mask = (mask * 255).cpu().numpy().astype(np.uint8)
            if mask.ndim == 2: 
                 mask = mask.reshape((H, W, 1))
            
            rgba = np.concatenate([img, mask], axis=2)
            frame = av.VideoFrame.from_ndarray(rgba, format='rgba')
            for packet in video_stream.encode(frame):
                container.mux(packet)
        
        # Flush the video stream encoders
        for packet in video_stream.encode():
            container.mux(packet)

        # Write Audio to audio stream        
        frame = av.AudioFrame.from_ndarray(audio_ndarray, format='fltp', layout=layout)
        frame.sample_rate = sample_rate
        for packet in audio_stream.encode(frame):
            container.mux(packet)

        # Flush the audio stream encoders
        for packet in audio_stream.encode():
            container.mux(packet)

    return path
