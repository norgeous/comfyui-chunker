# Video Module Specification

## Environment
- Use `conda activate comfyui` to get PyAV
- use PyAV 17.0.0+
- read the docs at https://pyav.basswood-io.com/docs/stable/
- do not use ffmpeg subprocess calls

## mux Function
- code inside `./mux2.py` only, dont edit other files
- Concatinate sequential mp4 videos into one mp4, with optional overlap.
- inputs:
    - **videos**: List of 1 or more similar h264 aac mp4 file paths (required)
    - **overlap_count**: count of frames to overlap between consecutive videos (0 or more) (required)
    - **overlap_blend_mode**: use the enum from utils_blend_mode.py
    - **output_path**: Path to save the output mp4 file (required)
- process
    - processing video packets from all source videos
        - you must use pyav remuxing when possible, dont decode all packets
        - itterate through each video stream and collect a list of list of packets
        - use utils_blend_videos.py to combine packets into a flat list
    - processing audio packets from all sources
        - decode all audio packets from all sources first into samples, collect a list of list of samples
        - using samples per frame logic, convert overlap to audio overlap
        - itterate through each audio packet and decode into a new list of samples
    - it's important to setup the both empty audio and video streams inside the output container before starting to add packets to either stream.
    - add packets into both streams

