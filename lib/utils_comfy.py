import os
import folder_paths

def get_next_save_path(filename_prefix, extension):
    full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(
        filename_prefix,
        folder_paths.get_output_directory(),
    )
    file = f"{filename}_{counter:05}_.{extension}"
    full_path = os.path.join(full_output_folder, file)
    return (
        full_path,
        {
            "filename": file,
            "subfolder": subfolder,
            "type": "output",
        },
    )
