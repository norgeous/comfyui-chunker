# comfyui-chunker

Create longer videos by automatically taking the last few frames of the last generated video and feeding it into the first few frames of the next generation.

- Easy to understand nodes that help reduce complexity and increase readability of workflows.
- Chunker reduces RAM usage by saving the generation into a high quality mp4 file on disk.
- Easily blend overlaps of images, masks and audio, with a variety of blend methods.
- Preview completed chunks as they are generated, track overall progress, view an estimate of time remaining.
- Uses PyAV for video processing, nothing extra to install as PyAV is included in the default ComfyUI install.

## Nodes

### 🍫 Repeat

Repeat nodes between this node and `🍫 Combine`. Optionally divide long batches images, masks and / or audio into smaller chunks and process the chunks sequentially. Optionally use the end of last chunk as start of this chunk (with `chunk_overlap`).

### 🍫 Data

Extract values from `chunker_data` as individual outputs, for use elsewhere in your workflow.

### 🍫 Combine

Combine sequential chunks back into single av tensors. Each chunk is saved to a mp4 file and recombined excluding the overlap frames.

Note: You may need a (very) large pagefile or swap configured.

## [Workflows](workflows/)

### [chunker-sam3](workflows/chunker-sam3.json)

- Create masks for long video (by prompt) in batches of 100 images, up to any length.
- Input: [example.mp4](https://www.pexels.com/video/hip-hop-dancer-2795752/) (1080x1920, 425 frames @ 25FPS)

https://github.com/user-attachments/assets/9b8462b1-08d2-4827-b42b-65734739de1c


