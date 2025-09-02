import os
import folder_paths
from comfy_extras.nodes_video import CreateVideo
from comfy_api.util import VideoContainer

def save_video(images, fps, filename_prefix):

    create_video_node = CreateVideo()
    video, = create_video_node.execute(images, fps)

    width, height = video.get_dimensions()
    full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(
        filename_prefix,
        folder_paths.get_output_directory(),
        width,
        height
    )

    file = f"{filename}_{counter:05}_.{VideoContainer.get_extension(format)}"
    video.save_to( os.path.join(full_output_folder, file), format="auto", codec="auto", metadata={})

    return {
        "filename": file,
        "subfolder": subfolder,
        "type": "output",
    }
