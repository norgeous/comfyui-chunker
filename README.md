# comfyui-chunker

Create longer Wan videos by automatically taking the last few frames of the last generated video and feeding it into the first few frames of the next generation. 

- RAM: Chunker saves previous generation in an mp4 file, this means less ram is used
- 

![image](./workflows/chunker-wan21vace-t2v.png)
[workflow](./workflows/chunker-wan21vace-t2v.json)

## Samples

https://github.com/user-attachments/assets/587530f3-1752-46dd-9156-9343fa2ab16d

## Nodes

### 🍫 Divide

- Chop large av tensor (images, masks or audio) into smaller chunks and process the chunks sequentially.
- Optionally use the end of last chunk as start of this chunk (with `overlap`).

### 🍫 Combine

- Combine sequential chunks back into single av tensors
- Each chunk is saved to a mp4 file and recombined excluding the overlap frames

### 🍫 VACE To First Last

- Convert an i2v VACE control_video into First Last frames
- A fully grey first or last image in the input control_video will result in `None` output for the corresponding `clip_vision_` and `_image`

## [Workflows](workflows/)

### [chunker-sam3](workflows/chunker-sam3.json)

- Create masks for long video (by prompt) in batches of 100 images, up to any length
- Requires: [ComfyUI-SAM3](https://github.com/PozzettiAndrea/ComfyUI-SAM3)
- Example input video: [example.mp4](https://www.pexels.com/video/hip-hop-dancer-2795752/) 1080x1920, 425 frames @ 25FPS
- Output: ![video](.github/assets/chunker-sam3-output.mp4)

### [chunker-wan21-vace](workflows/chunker-wan21-vace.json)

- Output 1: ![video](.github/assets/chunker-wan21-vace-output1.mp4)

### [chunker-wan22-flf2v](workflows/chunker-wan22-flf2v.json)

- todo

### [chunker-mmaudio](workflows/chunker-mmaudio.json)

- Create audio for long video in chunks of 8 seconds (200 frames)

### [chunker-wan22-svi-pro](workflows/chunker-wan22-svi-pro.json)

- todo

## prep for release

- Divide
  - load_frames gets 4 when asking for 5
  - cached Divide, causes wrong time estimates in Combine

- Combine
  - bug: glitched video if "previous_chunk"? muxing issue?
  - buggy warning emoji in timestamps
  - Combine setting: "overlap_blend": older_only, linear_blend, ease_in_out, newer_only
  - set crf on intermediate mp4
  - combined audio sounds glitchy? also not working? muxing issue where 1024 samples in each packet?
  - tqdm + comfy progressbar for video ops
  - think about any way to show underlap when "previous_chunk"
  - seed
    - option to enable seed incrementing (might want either)
    - KSampler with manual seed connected to values causes Combine to error when setting the value
    - increment "RandomNoise" seed for workflows with "SamplerCustomAdvanced"
  - keep progress bar shown after final execution

- remove extra nodes
- Tidy unused code
- revise readme and samples
- ensure builtin docs are working
- finalise a few workflows with previews
- swapping internal comfyui workflow tab erases progress bar
- test everything
- fix preview webm alpha
- animated css bg behind video (for alpha)
- comb for todos and address all?
- comb for ' and swap to "
- delete tmp files after execution
- trash this repo make a new one with one commit
- publish to comfyui-manager via PR

## Known limitations and issues

- VACE sometimes rejects the size (esp. some smaller sizes)
- Can't have 2x sequencial Divide + Combine in same workflow because random ComfyUI execution order means an "out of vram" error is likely
- VAE Decode (tiled) can cause an error if the current chunk length is small, which may happen in the last chunk of a long video

## Future / Ideas

- Chunker Composer idea
  - for i2v
  - takes a set of images and masks and generates a control_video and masks for Chunker to consume
  - allows the user to compose control_video and masks with text string
  - example sequence: `1,fill,2`
    - put 1st image as first image
    - fill middle of chunk with blank panels 
    - put 2nd image as last image
  - it needs to discover or be told the chunk length and overlap settings and use them in the calculations


## Donations

- ??