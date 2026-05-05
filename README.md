# comfyui-chunker

Create longer videos by automatically taking the last few frames of the last generated video and feeding it into the first few frames of the next generation. 

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

## [Workflows](workflows/)

### [chunker-sam3](workflows/chunker-sam3.json)

- Create masks for long video (by prompt) in batches of 100 images, up to any length
- Example input video: [example.mp4](https://www.pexels.com/video/hip-hop-dancer-2795752/) 1080x1920, 425 frames @ 25FPS
- Output: ![video](.github/assets/chunker-sam3-output.mp4)

### [chunker-wan21-vace](workflows/chunker-wan21-vace.json)

- Output 1: ![video](.github/assets/chunker-wan21-vace-output1.mp4)

### [chunker-wan22-svi-pro](workflows/chunker-wan22-svi-pro.json)

- todo

## prep for release

- Combine
  - tqdm + comfy progressbar for video ops
  - write tmp files to tmp dir and delete tmp files after needed
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