import os
import folder_paths
import ffmpeg
from comfy_extras.nodes_video import CreateVideo
# from comfy_api.input_impl import VideoFromFile
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

def ffmpeg_first_frame(path):
    first_frames_dir = os.path.join(folder_paths.get_input_directory(), "first-frames") 
    out_file = f"{os.path.basename(path)}.png"
    out_path = os.path.join(first_frames_dir, out_file)
    frontend_data = {
        "type": "input",
        "filename": out_file,
        "subfolder": "first-frames",
    }
    # check if png already created
    if os.path.isfile(out_path):
        return frontend_data
    if not os.path.isdir(first_frames_dir):
        os.mkdir(first_frames_dir)
    # if not use ffmpeg to extract first frame
    input = ffmpeg.input(path).filter("select", f"eq(n,0)")
    try:
        input.output(
            out_path,
            vframes=1
        ).run(capture_stdout=True, capture_stderr=True)
    except ffmpeg.Error as e:
        print('stdout:', e.stdout.decode('utf8'))
        print('stderr:', e.stderr.decode('utf8'))
        raise e
    return frontend_data
