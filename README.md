# comfyui-chunker

Create longer Wan VACE videos by automatically taking the last few frames of the last generated video and feeding it into the first few frames of the next generation.

## Nodes

### `🍫 Chunker`

Creates chunks of `chunk_length` from a resized version of `control_video` and `control_masks` if provided. If not provided blank ones are prepared.

### `🍫 Combine`

Combines modified chunks into one single output.
