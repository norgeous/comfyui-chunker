# comfyui-chunker

Create longer videos by automatically taking the last few frames of the last generated video and feeding it into the first few frames of the next generation. 

- RAM: Chunker saves previous generation in an mp4 file, this means less ram is used
- blends overlaps

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

## Donations

- ??
