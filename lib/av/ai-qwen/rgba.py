import av
import numpy as np
from fractions import Fraction

def combine_images_masks(output_prefix, images, masks, audio, fps):
    B, H, W, C = images.shape

    # Create video container
    with av.open(f"{output_prefix}.mov", mode='w') as container:
        # add video stream
        video_stream = container.add_stream('prores_ks', rate=fps)
        video_stream.pix_fmt = 'yuva444p10le'
        video_stream.width = W
        video_stream.height = H

        # process images and masks
        for i in range(B):
            img = images[i]  # Shape: (H, W, C)
            mask = masks[i]  # Shape: (H, W)

            # Convert to 0-255 range and cast to uint8
            img = (img * 255).cpu().numpy().astype(np.uint8)
            mask = (mask * 255).cpu().numpy().astype(np.uint8)

            # Reshape mask to (H, W, 1)
            mask = mask.reshape((img.shape[0], img.shape[1], 1))

            # Concatenate image and mask to form RGBA
            rgba = np.concatenate([img, mask], axis=2)

            # Create video frame
            frame = av.VideoFrame.from_ndarray(rgba, format='rgba')
            for packet in video_stream.encode(frame):
                container.mux(packet)









        # Handle audio stream
        if audio is not None:
            waveform = audio['waveform']
            sample_rate = audio['sample_rate']

            # Convert waveform to numpy array if it's a tensor
            if not isinstance(waveform, np.ndarray):
                waveform = np.array(waveform)

            # Determine audio parameters
            # channels = waveform.shape[1] if len(waveform.shape) > 1 else 1
            sample_format = 's16'  # 16-bit signed integer

            # Create audio stream
            audio_stream = container.add_stream('flac', rate=sample_rate)
            # audio_stream.sample_rate = sample_rate
            # audio_stream.channels = channels
            # audio_stream.sample_format = sample_format

            # Write audio data
            audio_data = waveform.flatten().astype(np.int16)
            for packet in audio_stream.encode(audio_data):
                container.mux(packet)



















        # # Add audio stream with PCM encoding
        # audio_stream = container.add_stream('pcm_s24le', rate=audio['sample_rate'])
        # # audio_stream = container.add_stream('flac', rate=audio['sample_rate'])
        # audio_stream.sample_rate = audio['sample_rate']
        # audio_stream.layout = 'stereo'

        # Process audio
        # waveform = audio['waveform'].cpu().numpy().astype(np.float32)
        # # waveform = waveform.reshape(-1, 2)  # Assuming stereo
        # for samples in waveform:
        #     print(samples.shape)
        #     audio_frame = av.AudioFrame.from_ndarray(samples, format='fltp')
        #     audio_frame.sample_rate = audio["sample_rate"]
        #     for packet in audio_stream.encode(audio_frame):
        #         container.mux(packet)

        # # Process audio
        # print(audio['waveform'].shape)
        # waveform = audio['waveform'].cpu().numpy().astype(np.int32)
        # waveform = waveform.reshape(-1, 2)  # Assuming stereo
        # print(waveform.shape)
        # for samples in waveform:
        #     audio_frame = av.AudioFrame.from_ndarray(samples, format='s32')
        #     audio_frame.sample_rate = audio["sample_rate"]
        #     for packet in audio_stream.encode(audio_frame):
        #         container.mux(packet)

        # waveform = audio['waveform'].cpu().numpy().astype(np.float32)


        # # normalize shapes
        # if waveform.ndim == 3 and waveform.shape[0] == 1:
        #     waveform = waveform[0]
        # # now a shape should be (C,S)
        # if waveform.ndim == 1:
        #     waveform = np.expand_dims(waveform, 0)

        # # ensure float32
        # if waveform.dtype != np.float32:
        #     waveform = waveform.astype(np.float32)



        # sample_rate = audio["sample_rate"]
        # S = waveform.shape[1]
        # layout = 'stereo'
        # frame_size=2048
        # pos = 0
        # while pos < S:
        #     end = min(S, pos + frame_size)
        #     chunk = waveform[:, pos:end]
        #     # av expects ndarray shape (channels, samples) for planar formats
        #     frame = av.AudioFrame.from_ndarray(chunk, format='fltp', layout=layout)
        #     frame.sample_rate = sample_rate
        #     frame.pts = None
        #     # frame.time_base = Fraction(1, 10000)

        #     print(frame)
        #     for packet in audio_stream.encode(frame):
        #         # packet.dts = 1
        #         # packet.pts = 1
        #         print(packet)
        #         container.mux(packet)
        #     pos = end


        # flush audio encoder
        for packet in audio_stream.encode(None):
            container.mux(packet)