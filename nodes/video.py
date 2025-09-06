import os
import folder_paths
from comfy_extras.nodes_video import CreateVideo
from comfy_api.input_impl import VideoFromFile
from comfy_api.util import VideoContainer

def save_video(images, fps, filename_prefix):
    format = "auto"
    codec = "auto"
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
    full_path = os.path.join(full_output_folder, file)
    video.save_to(full_path, format=format, codec=codec, metadata=None)

    return (
        full_path,
        {
            "filename": file,
            "subfolder": subfolder,
            "type": "output",
        },
    )

def load_video_images_exclude_overlap(full_path, overlap):
    if full_path is None: return None
    images = VideoFromFile(full_path).get_components().images
    if overlap == 0: return images 
    return images[:-overlap]
