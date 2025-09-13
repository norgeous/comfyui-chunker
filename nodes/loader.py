# https://stackoverflow.com/a/77782755

import torch
import numpy as np
import av

def load_video_chunk(
        video_path="/home/user/ComfyUI/input/#cosplay.mp4",
        start_n=0,
        end_n=None,
    ):
    container = av.open(video_path)
    stream = container.streams.video[0]
    duration = container.duration / av.time_base
    fps = stream.average_rate
    total_length = stream.frames

    #print(f"total_length: {total_length} frames")
    #print(f"fps: {fps}")

    if end_n is None: end_n = total_length # missing end fix
    if end_n < 0: end_n = total_length + end_n # negative end fix

    # sanity checks
    assert start_n < end_n, "Start beyond end"
    assert end_n <= total_length, f"End {end_n} beyond total_length {total_length}"

    start_time = start_n / fps
    end_time = end_n / fps

    # seek
    print(f"Seeking to {start_time:.2f} seconds or earlier")
    container.seek(int(start_time / stream.time_base), stream=stream)

    # decode between frames
    all_frames_n = []
    frames = []
    for frame in container.decode(stream):
        assert frame.time == float(frame.pts * stream.time_base)

        frame_n = round(frame.pts * stream.time_base * stream.average_rate)
        all_frames_n.append(frame_n)

        if frame_n >= end_n:
            print("end", frame_n, " > ", end_n)
            break
        elif frame_n < start_n:
            continue
        else:
            img = frame.to_ndarray(format='rgb24')  # shape: (H, W, 3)
            img = torch.from_numpy(img) / 255.0  # shape: (H, W, 3)
            frames.append(img)

    print(f"itterated {len(all_frames_n)} frames and collected {len(frames)} frames")
    return (frames, fps, total_length)


def awesome_loader(path, start, end):
    img_ext = ["jpeg", "jpg", "png"]
    vid_ext = ["mp4"]
    file_ext = path.replace(" [input]", "").split(".")[-1]
    if file_ext in img_ext:
        image = None
        fps = 0
        total_length = 1
        return (frames, fps, total_length)
    if file_ext in vid_ext:
        frames, fps, total_length = load_video_chunk(path, start_n=start, end_n=end)
        return (frames, fps, total_length)
