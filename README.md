# comfyui-chunker

Create longer videos by automatically taking the last few frames of the last generated video and feeding it into the first few frames of the next generation. 

- RAM: Chunker saves previous generation in an mp4 file, this means less ram is used
- 

![image](./workflows/chunker-wan21vace-t2v.png)
[workflow](./workflows/chunker-wan21vace-t2v.json)

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

## prep for release

- Combine
  - fix preview webm alpha -> show the real masking as alpha, but keep the overlayed text
  - show chunk progresion in pbar as we finishing node execution?
- Divide
  - its not currently loading the masks from overlap
  - show chunk_lengths in ui?
- Combine frontend
  - swapping internal comfyui workflow tab erases progress bar and preview video
  - "~0" on cached, should be "unknown"
  - "~overdue" should be "overdue"
  - video width fix not work in nodes 2.0
  - execute a fully cached chunker gives wrong ui
- Tidy unused code
- revise readme and samples
- ensure builtin docs are working
- finalise a few workflows with previews - just sam3 for now?
- comb for todos and address all?
- comb for ' and swap to "
- test everything
- trash this repo make a new one with one commit
- publish to comfyui-manager via PR

## Donations

- ??
