import os
import folder_paths
import ffmpeg
from comfy_extras.nodes_video import CreateVideo
from comfy_api.input_impl import VideoFromFile
from comfy_api.util import VideoContainer


def get_next_save_video_path(filename_prefix):
    format = "auto"
    full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(
        filename_prefix,
        folder_paths.get_output_directory(),
    )
    file = f"{filename}_{counter:05}_.{VideoContainer.get_extension(format)}"
    full_path = os.path.join(full_output_folder, file)
    return (
        full_path,
        {
            "filename": file,
            "subfolder": subfolder,
            "type": "output",
        },
    )


def save_video(images, fps, filename_prefix):
    format = "auto"
    codec = "auto"
    create_video_node = CreateVideo()
    video, = create_video_node.execute(images, fps)
    full_path, frontend_data = get_next_save_video_path(filename_prefix)
    video.save_to(full_path, format=format, codec=codec, metadata=None)
    return (
        full_path,
        frontend_data,
    )


def load_video_images_exclude_overlap(full_path, overlap):
    if full_path is None: return None
    images = VideoFromFile(full_path).get_components().images
    if overlap == 0: return images
    return images[:-overlap]

def ffmpeg_info(path):
    info = ffmpeg.probe(path)
    video = list(filter(lambda stream: stream['codec_type'] == "video", info["streams"]))[0]
    fps_parts = video["avg_frame_rate"].split("/")
    return {
        "fps": int(fps_parts[0]) / int(fps_parts[1]),
        "frame_count": int(video["nb_frames"]),
    }

def ffmpeg_first_frame(path):
    out_path = os.path.join(folder_paths.get_output_directory(), "first-frames", os.path.basename(path), ".png")
    # TODO: check if png already created
    # if not use ffmpeg to extract first frame
    input = ffmpeg.input(path).filter("select", f"eq(n,0)")
    try:
        input.output(
            out_path,
        ).run(capture_stdout=True, capture_stderr=True)
    except ffmpeg.Error as e:
        print('stdout:', e.stdout.decode('utf8'))
        print('stderr:', e.stderr.decode('utf8'))
        raise e
    return out_path

def ffmpeg_load_chunk(path, start, end, filename_prefix, crf=0):
    full_path, frontend_data = get_next_save_video_path(filename_prefix)
    input = ffmpeg.input(path).filter("select", f"between(n,{start},{end})")
    try:
        input.output(
            full_path,
            crf=crf, # between 0 (max quality) and 51 (lowest quality for h264)
        ).run(capture_stdout=True, capture_stderr=True)
    except ffmpeg.Error as e:
        print('stdout:', e.stdout.decode('utf8'))
        print('stderr:', e.stderr.decode('utf8'))
        raise e
    images = load_video_images_exclude_overlap(full_path, 0)
    # perhaps delete the temporary file
    return images

def ffmpeg_hstack(paths, filename_prefix, crf=0):
    full_path, frontend_data = get_next_save_video_path(filename_prefix)
    inputs = [ffmpeg.input(path) for path in paths]
    stacked = ffmpeg.filter(inputs, "hstack")
    try:
        stacked.output(
            full_path,
            crf=crf, # between 0 (max quality) and 51 (low quality)
        ).run(capture_stdout=True, capture_stderr=True)
    except ffmpeg.Error as e:
        print('stdout:', e.stdout.decode('utf8'))
        print('stderr:', e.stderr.decode('utf8'))
        raise e
    return (
        full_path,
        frontend_data,
    )

def ffmpeg_cat(paths, length, overlap, filename_prefix, crf=0, select_overlaps_from="this_chunk"):
    full_path, frontend_data = get_next_save_video_path(filename_prefix)

    if select_overlaps_from == "this_chunk":
        inputs = [ffmpeg.input(path).filter("select", f"between(n,0,{length-1 - overlap})") if i != len(paths) - 1 else ffmpeg.input(path) for i, path in enumerate(paths)]

    if select_overlaps_from == "previous_chunk":
        info = ffmpeg_info(paths[0])
        fps = info["fps"]
        inputs = [ffmpeg.input(path) if i == 0 else ffmpeg.input(path, ss=overlap/fps, t=length/fps) for i, path in enumerate(paths)]

    joined = ffmpeg.concat(*inputs, v=1, a=0)
    try:
        joined.output(
            full_path,
            crf=crf, # between 0 (max quality) and 51 (low quality)
        ).run(capture_stdout=True, capture_stderr=True)
    except ffmpeg.Error as e:
        print('stdout:', e.stdout.decode('utf8'))
        print('stderr:', e.stderr.decode('utf8'))
        raise e

    return (
        full_path,
        frontend_data,
    )
