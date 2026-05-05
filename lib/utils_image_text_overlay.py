import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from .utils_tensor import mask_to_image, simple_blend

# mostly from https://github.com/munkyfoot/ComfyUI-TextOverlay/blob/main/nodes.py

def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip("#")
    if len(hex_color) == 3:
        hex_color = hex_color * 2
    return tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4))

def draw_text(
    image,
    text,
    font_size=18,
    fill_color_hex="#FFFFFF",
    stroke_color_hex="#000000",
    stroke_thickness=0.2,
    padding=8,
    horizontal_alignment="center",
    vertical_alignment="bottom",
    x_shift=0,
    y_shift=0,
    line_spacing=0,
    use_cache=False,
):
    _loaded_font = ImageFont.load_default(font_size)
    _full_text = None
    _x = None
    _y = None

    # Prepare to draw on the image
    draw = ImageDraw.Draw(image)

    # Process text for multiline support and fit within image dimensions
    words = text.replace("\n", "\n ").split(" ")
    if _full_text is None or use_cache is False:
        text_lines, line = [], ""
        for word in words:
            extra_line = "\n" in word
            word = word.strip()
            if (
                draw.textlength(line + word, font=_loaded_font)
                < image.width - 2 * padding
            ):
                line += word + " "
            else:
                text_lines.append(line.strip())
                line = word + " "
            if extra_line:
                text_lines.append(line.strip())
                line = ""
        text_lines.append(line.strip())
        _full_text = "\n".join(text_lines)

    # Calculate text position based on alignment and position adjustments
    if _x is None or _y is None or use_cache is False:
        left, top, right, bottom = draw.multiline_textbbox(
            (0, 0),
            _full_text,
            font=_loaded_font,
            stroke_width=int(font_size * stroke_thickness * 0.5),
            align=horizontal_alignment,
            spacing=line_spacing,
        )
        if horizontal_alignment == "left":
            _x = padding
        elif horizontal_alignment == "center":
            _x = (image.width - (right - left)) / 2
        elif horizontal_alignment == "right":
            _x = image.width - (right - left) - padding
        _x += x_shift
        if vertical_alignment == "middle":
            _y = (image.height - (bottom - top)) / 2
        elif vertical_alignment == "top":
            _y = padding
        elif vertical_alignment == "bottom":
            _y = image.height - (bottom - top) - padding
        _y += y_shift

    # Draw the processed text onto the image
    draw.text(
        (_x, _y),
        _full_text,
        fill=hex_to_rgb(fill_color_hex),
        stroke_fill=hex_to_rgb(stroke_color_hex),
        stroke_width=int(font_size * stroke_thickness * 0.5),
        font=_loaded_font,
        align=horizontal_alignment,
        spacing=line_spacing,
    )
    return image

def batch_draw_text(
    image,
    configs,
):
    # Handles both single and batch image processing for text overlay
    if len(image.shape) == 3:  # Single image
        image_np = image.cpu().numpy()
        image = Image.fromarray((image_np.squeeze(0) * 255).astype(np.uint8))
        image = draw_text(
            image,
            **configs[0],
        )
        image_tensor_out = torch.tensor(np.array(image).astype(np.float32) / 255.0)
        image_tensor_out = torch.unsqueeze(image_tensor_out, 0)
        return image_tensor_out
    else:
        # Batch of images
        image_np = image.cpu().numpy()
        images = [Image.fromarray((img * 255).astype(np.uint8)) for img in image_np]
        images_out, use_cache = [], False

        # for each img in images
        for i, img in enumerate(images):

            # for each config in configs
            for config in configs[i]:
                img = draw_text(
                    img,
                    **config,
                    #use_cache,
                )
            images_out.append(np.array(img).astype(np.float32) / 255.0)
            # use_cache = True
        images_np = np.stack(images_out)
        images_tensor = torch.from_numpy(images_np)
        return images_tensor

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

# TODO: snake case
def getOverlayConfig(i, previous_count, chunk_index, chunk_count, total, w, h, chunk_length, overlap, fps, overlap_blend_mode):
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
            "text": f"{w} x {h} @ {fps:.2f}FPS\nchunk_length: {chunk_length}\nchunk_overlap: {overlap}\noverlap_blend_mode: {overlap_blend_mode}",
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

def overlay_debug_text(images, previous_count, chunk_index, chunk_count, chunk_length, chunk_overlap, total_length, fps, overlap_blend_mode):
    w = images.shape[2]
    h = images.shape[1]
    config = [getOverlayConfig(i, previous_count, chunk_index, chunk_count, total_length, w, h, chunk_length, chunk_overlap, fps, overlap_blend_mode) for i in range(0, len(images))]
    images = batch_draw_text(images, config)
    return images

def combine_images_and_masks(images, masks):
    imasks = mask_to_image(masks) if masks is not None else None
    out = None
    if images is not None and imasks is None: out = images
    if images is None and imasks is not None: out = imasks
    if images is not None and imasks is not None: out = simple_blend(images, imasks)
    return out

def create_preview_video(images, masks, d, c, overlap_blend_mode):
    previous_count = ((d["index"]) * (c["chunk_length"] - c["chunk_overlap"]))
    preview_video_chunk = combine_images_and_masks(images, masks)
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
    )
    return preview_video_chunk

