# comfyui-chunker

Create longer videos by automatically taking the last few frames of the last generated video and feeding it into the first few frames of the next generation.

- Easy to understand nodes that help reduce complexity and increase readability of workflows.
- Chunker reduces RAM usage by saving the generation into a high quality mp4 file on disk.
- Easily blend overlaps of images, masks and audio, with a variety of blend methods.
- Preview completed chunks as they are generated, track overall progress, view an estimate of time remaining.
- Uses PyAV for video processing, nothing extra to install as PyAV is included in the default ComfyUI install.

## Nodes

(image of nodes?)

### 🍫 Divide

Chop large av tensor (images, masks and / or audio) into smaller chunks and process the chunks sequentially. Optionally use the end of last chunk as start of this chunk (with `overlap`).

#### Inputs

| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| images | IMAGE | No | - | Input images to chunk |
| masks | MASK | No | - | Input masks to chunk |
| audio | AUDIO | No | - | Input audio to chunk |
| fps | FLOAT | No | - | Frame rate (forced input) |
| mode | COMBO | No | "default" | Chunk format: "default" (2n), "wan" (4n+1), "ltx2" (8n+1) |
| chunk_length | INT | Yes | 81 | Count of images in each chunk (1-4096) |
| chunk_overlap | INT | Yes | 4 | Count of images to overlap between chunks (0-4096) |
| total_length | INT | Yes | 0 | Minimum count of images in final output (0 = use input length, max 10000) |

#### Outputs

| Name | Type | Description |
|------|------|-------------|
| chunker_data | CHUNKER_DATA | Connect to ChunkerCombine node |
| images | IMAGE | Current chunk of images |
| masks | MASK | Current chunk of masks |
| audio | AUDIO | Current chunk of audio |
| fps | FLOAT | Frame rate |
| chunk_length | INT | Count of images in this chunk |
| chunk_overlap | INT | Count of images to overlap between each chunk |
| total_length | INT | Total length of output images |
| chunk_count | INT | Total count of chunks |
| index | INT | Current iteration index (0, 1, 2, ...) |

### 🍫 Combine

Combine sequential chunks back into single av tensors. Each chunk is saved to a mp4 file and recombined excluding the overlap frames.

Note: You may need a (very) large pagefile or swap configured.

#### Inputs

| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| chunker_data | CHUNKER_DATA | Yes | - | Connect from ChunkerDivide node |
| images | IMAGE | No | - | Processed chunk of images |
| masks | MASK | No | - | Processed chunk of masks |
| audio | AUDIO | No | - | Processed chunk of audio |
| overlap_blend_mode | COMBO | Yes | "newer_only" | Blend mode for overlaps: "newer_only", "older_only", "equal_power", "linear", "ease_in_out" |
| increment_seeds | BOOLEAN | Yes | True | Increment seed/noise_seed in Sampler/Noise nodes between chunks |

#### Outputs

| Name | Type | Description |
|------|------|-------------|
| images | IMAGE | Combined images from all chunks |
| masks | MASK | Combined masks from all chunks |
| audio | AUDIO | Combined audio from all chunks |
| fps | FLOAT | Frame rate |

## [Workflows](workflows/)

### [chunker-sam3](workflows/chunker-sam3.json)

- Create masks for long video (by prompt) in batches of 100 images, up to any length.
- Input: [example.mp4](https://www.pexels.com/video/hip-hop-dancer-2795752/) (1080x1920, 425 frames @ 25FPS)

https://github.com/user-attachments/assets/9b8462b1-08d2-4827-b42b-65734739de1c


