import av
import numpy as np

def combine_images_masks(images, masks, output_prefix):
    B, H, W, C = images.shape

    # Create video container
    with av.open(f"{output_prefix}.mov", mode='w') as container:
        video_stream = container.add_stream('prores_ks', rate=30)
        video_stream.pix_fmt = 'yuva444p10le'
        video_stream.width = W
        video_stream.height = H

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
