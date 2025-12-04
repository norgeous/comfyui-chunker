import os
import numpy as np
import av
from PIL import Image
import folder_paths
from ..utils import tensor2pil


def _get_next_save_path(filename_prefix, ext="mkv"):
    full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(
        filename_prefix,
        folder_paths.get_output_directory(),
    )
    file = f"{filename}_{counter:05}_.{ext}"
    full_path = os.path.join(full_output_folder, file)
    return full_path, {"filename": file, "subfolder": subfolder, "type": "output"}


def _mask_to_pil(mask_tensor):
    # mask_tensor expected shape: (1,H,W) or (H,W)
    m = mask_tensor
    try:
        import torch
        if hasattr(m, "cpu"):
            m = m.cpu().numpy()
    except Exception:
        pass
    # remove batch dim
    if m.ndim == 3 and m.shape[0] == 1:
        m = m[0]
    # assume float in 0..1
    m = np.clip(m * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(m, mode="L")


def _chunk_audio_and_encode(container, astream, audio_np, sample_rate, layout="stereo", frame_size=2048):
    # audio_np shape: (channels, samples)
    C, S = audio_np.shape
    # choose format float
    pos = 0
    while pos < S:
        end = min(S, pos + frame_size)
        chunk = audio_np[:, pos:end]
        # av expects ndarray shape (channels, samples)
        frame = av.AudioFrame.from_ndarray(chunk, format='flt', layout=layout)
        frame.sample_rate = sample_rate
        for packet in astream.encode(frame):
            container.mux(packet)
        pos = end
    # flush audio encoder
    for packet in astream.encode(None):
        container.mux(packet)


def save_prores_with_alpha(images=None, masks=None, audio=None, fps=30, filename_prefix="chunk", out_path=None):
    """Save frames (images) and masks (as alpha) plus optional audio to a ProRes+FLAC file.

    - images: torch tensor or numpy array of shape (N,H,W,3) with float in 0..1
    - masks: tensor/array of shape (N,H,W) or (N,1,H,W) with float in 0..1
    - audio: optional dict with keys `waveform` and `sample_rate`, where `waveform` is
        a tensor/ndarray shaped (1,C,S) or (C,S) and `sample_rate` is an int.
        For backward compatibility, passing a raw ndarray/tensor in `audio` is also supported,
        but then you must wrap it as `{"waveform": arr, "sample_rate": sr}`.
    - fps: frames per second for the video
    - filename_prefix: prefix used to construct output filename
    - out_path: optional explicit output path (overrides prefix)

    Returns: (full_path, frontend_meta)
    """
    if images is None and masks is None and audio is None:
        raise ValueError("At least one of images, masks or audio must be provided")

    # prepare output path
    if out_path is None:
        full_path, frontend = _get_next_save_path(filename_prefix, ext="mkv")
    else:
        full_path = out_path
        frontend = {"filename": os.path.basename(full_path), "subfolder": None, "type": "output"}

    os.makedirs(os.path.dirname(full_path), exist_ok=True)

    # determine frame count and whether we need a video stream
    N = 0
    if images is not None:
        try:
            N = images.shape[0]
        except Exception:
            N = len(images)
    elif masks is not None:
        try:
            N = masks.shape[0]
        except Exception:
            N = len(masks)

    # open container
    container = av.open(full_path, mode='w')

    vstream = None
    # add video stream only if we have frames to write
    if N > 0:
        vstream = container.add_stream('prores_ks', rate=fps)
        # guess width/height from first image or mask
        if images is not None:
            first_img_pil = tensor2pil(images[0:1])
        else:
            # create a dummy RGB from mask size
            mask_pil = _mask_to_pil(masks[0:1])
            first_img_pil = Image.new('RGB', mask_pil.size, (0, 0, 0))

        width, height = first_img_pil.size
        vstream.width = width
        vstream.height = height
        # use a pixel format that supports alpha; prores_ks typically supports yuva444p10le for 4444
        try:
            vstream.pix_fmt = 'yuva444p10le'
            vstream.options = {'profile': '4'}
        except Exception:
            # fallback to a generic
            vstream.pix_fmt = 'rgba'

    # add audio stream if provided
    astream = None
    audio_waveform = None
    audio_sample_rate = None
    if audio is not None:
        # accept dict {"waveform": tensor/ndarray, "sample_rate": int}
        if isinstance(audio, dict):
            audio_waveform = audio.get("waveform")
            audio_sample_rate = audio.get("sample_rate")
        else:
            # backward compatibility: treat audio as raw waveform but require sample_rate provided inside dict
            audio_waveform = audio
            audio_sample_rate = None

        if audio_waveform is None or audio_sample_rate is None:
            raise ValueError("audio must be a dict containing 'waveform' and 'sample_rate'")

        astream = container.add_stream('flac', rate=audio_sample_rate)

    # write frames (if any)
    if vstream is not None:
        for i in range(N):
            # build RGBA PIL image
            if images is not None:
                rgb_pil = tensor2pil(images[i:i+1])
            else:
                # create black RGB image matching mask size
                mask_pil_tmp = _mask_to_pil(masks[i:i+1])
                rgb_pil = Image.new('RGB', mask_pil_tmp.size, (0, 0, 0))

            # if mask exists, use it; otherwise write fully opaque
            if masks is not None:
                mask_pil = _mask_to_pil(masks[i:i+1])
                rgba = rgb_pil.convert('RGBA')
                rgba.putalpha(mask_pil)
            else:
                rgba = rgb_pil.convert('RGBA')

            vframe = av.VideoFrame.from_image(rgba)
            for packet in vstream.encode(vframe):
                container.mux(packet)

        # flush video encoder
        for packet in vstream.encode(None):
            container.mux(packet)

    # process audio (if present). Expect audio_waveform shape (1,C,S) or (C,S)
    if audio_waveform is not None and astream is not None:
        # convert to numpy
        try:
            import torch
            if hasattr(audio_waveform, 'cpu'):
                a = audio_waveform.cpu().numpy()
            else:
                a = np.array(audio_waveform)
        except Exception:
            a = np.array(audio_waveform)

        # normalize shapes
        if a.ndim == 3 and a.shape[0] == 1:
            a = a[0]
        # now a shape should be (C,S)
        if a.ndim == 1:
            a = np.expand_dims(a, 0)

        # ensure float32
        if a.dtype != np.float32:
            a = a.astype(np.float32)

        # choose layout string
        layout = 'stereo' if a.shape[0] == 2 else 'mono'
        _chunk_audio_and_encode(container, astream, a, audio_sample_rate, layout=layout)

    container.close()
    return full_path, frontend
