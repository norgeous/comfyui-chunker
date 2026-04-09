# Video Module Specification

## Environment
- Use `conda activate comfyui` to get PyAV
- use PyAV 17.0.0+
- read the docs at https://pyav.basswood-io.com/docs/stable/
- do not use ffmpeg subprocess calls
- use the `with av.open` syntax

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
        - itterate through each video stream and collect a list of list of packets
        - use utils_blend_packets.py to combine packets into a flat list
        - use pyav remuxing to push flat list of packets into the output container, you might need to fix the pts and dts
    - processing audio packets from all sources
        - itterate through each audio stream and collect a list of list of packets
        - use utils_blend_packets.py to combine packets into a flat list
        - use pyav remuxing to push flat list of packets into the output container, you might need to fix the pts and dts
    - it's important to setup the both empty audio and video streams inside the output container before starting to add packets to either stream.
    - add processed packets into both streams

## tests
- copy tests from test_mux.py exactly, do not edit the tests
