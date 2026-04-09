# Video Module Specification

## Environment
- Use `conda activate comfyui` to get PyAV
- use PyAV 17.0.0+
- read the docs at https://pyav.basswood-io.com/docs/stable/
- do not use ffmpeg subprocess calls

## mux Function
- code inside `./mux.py`
- Concatinate sequential mp4 videos into one mp4, with optional overlap.
- inputs:
    - **videos**: List of 1 or more similar h264 aac mp4 file paths (required)
    - **overlap_count**: count of frames to overlap between consecutive videos (0 or more) (required)
    - **overlap_blend_mode**: (required)
        - older_only
        - linear_blend
        - ease_in_out
        - newer_only
    - **output_path**: Path to save the output mp4 file (required)
- process
    - it's important to setup the both empty audio and video streams inside the output container before starting to add packets to either stream.
    - processing video packets from all source videos
        - you must use pyav remuxing when possible, dont decode all packets
        - itterate through each video packet
            - calculate blend_factor using utils_blend_mode.py 
                - if blend_factor is 1 remux the packet
                - if 0 < blend_factor < 1
                    - 
            - packets outside the overlap region use remux to cleanly add the packet into the output
            - if packets are inside the overlap region
                - these packets are decoded
                - processed (blended according to overlap_blend_mode), then encoded back into packets and added to the output stream
    - processing audio packets from all sources
        - decode all audio packets from all sources first into samples
        - itterate through each audio packet and decode into a new list of samples
            - samples outside the overlap region use remux to cleanly add the frame into the output
            - samples inside the 2 adjacent overlap region are blended according to overlap_blend_mode
            - then encoded back into packets and added to the audio output stream

