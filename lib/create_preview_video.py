from .utils_tensor import mask_to_image, simple_blend
from .image_text_overlay import batch_draw_text


def frameIndexInfo(i, previous_count, chunk_index, chunk_count, chunk_length, total, overlap):
    chunk = chunk_index + 1
    is_first_chunk = chunk_index == 0
    is_last_chunk = chunk_index == chunk_count - 1
    is_overlap = (not is_last_chunk and i >= chunk_length - overlap) or (not is_first_chunk and i < overlap)
    return (
        f"{str(previous_count + i + 1).zfill(len(str(total)))} / {total}", # frame_label
        f"{str(chunk).zfill(len(str(chunk_count)))} of {chunk_count}", # chunk_label
        is_overlap,
    )


def get_overlay_config(i, previous_count, chunk_index, chunk_count, total, w, h, chunk_length, overlap, fps, overlap_blend_mode, audio_layout):
    frame_label, chunk_label, is_overlap = frameIndexInfo(i, previous_count, chunk_index, chunk_count, chunk_length, total, overlap)
    configs = []
    em = h / 512
    configs.append(
        {
            "text": f"{frame_label}\n{chunk_label}",
            "font_size": int(em * 20),
            "vertical_alignment": "top",
            "horizontal_alignment": "right",
        },
    )
    configs.append(
        {
            "text": f"{w} x {h} @ {fps:.2f}FPS\n{audio_layout}\nchunk_length: {chunk_length}\nchunk_overlap: {overlap}\noverlap_blend_mode: {overlap_blend_mode}",
            "font_size": int(em * 16),
            "vertical_alignment": "bottom",
            "horizontal_alignment": "right",
        },
    )
    if is_overlap:
        configs.append(
            {
                "text": "OVERLAP",
                "font_size": int(em * 28),
                "fill_color_hex": "#FF0000",
                "stroke_color_hex": "#FFFFFF",
                "vertical_alignment": "top",
                "horizontal_alignment": "left",
            },
        )
    return configs


def overlay_debug_text(images, previous_count, chunk_index, chunk_count, chunk_length, chunk_overlap, total_length, fps, overlap_blend_mode, audio_layout):
    w = images.shape[2]
    h = images.shape[1]
    config = [get_overlay_config(i, previous_count, chunk_index, chunk_count, total_length, w, h, chunk_length, chunk_overlap, fps, overlap_blend_mode, audio_layout) for i in range(0, len(images))]
    images = batch_draw_text(images, config)
    return images


def combine_images_and_masks(images, masks):
    imasks = mask_to_image(masks) if masks is not None else None
    out = None
    if images is not None and imasks is None: out = images
    if images is None and imasks is not None: out = imasks
    if images is not None and imasks is not None: out = simple_blend(images, imasks)
    return out


def create_preview_video(images, masks, audio, d, c, overlap_blend_mode):
    previous_count = ((d["index"]) * (c["chunk_length"] - c["chunk_overlap"]))
    preview_video_chunk = combine_images_and_masks(images, masks)
    audio_channel_count = audio["waveform"].shape[1] if audio is not None else 0
    audio_layout = ["", "mono", "stereo"][audio_channel_count]
    if audio is not None: audio_layout = f"{audio['sample_rate']}Hz {audio_layout}"
    preview_video_chunk = overlay_debug_text(
        preview_video_chunk,
        previous_count,
        d["index"],
        c["chunk_count"],
        c["chunk_length"],
        c["chunk_overlap"],
        c["total_length"],
        d["fps"],
        overlap_blend_mode,
        audio_layout,
    )
    return preview_video_chunk

