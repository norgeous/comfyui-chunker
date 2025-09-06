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


def ffmpeg_cat(paths, end, length, overlap, filename_prefix):
    full_path, frontend_data = get_next_save_video_path(filename_prefix)
    inputs = [ffmpeg.input(p) for p in paths]
    inputs = [
        ffmpeg.input(paths[0]).filter("select", f"between(n,0,{end})"),
        ffmpeg.input(paths[1]),
    ]
    joined = ffmpeg.concat(inputs[0], inputs[1], v=1, a=0)
    try:
        joined.output(full_path).run(capture_stdout=True, capture_stderr=True)
    except ffmpeg.Error as e:
        print('stdout:', e.stdout.decode('utf8'))
        print('stderr:', e.stderr.decode('utf8'))
        raise e
    return (
        full_path,
        frontend_data,
    )
