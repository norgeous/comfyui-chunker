# comfyui-chunker

Create longer Wan VACE videos by automatically taking the last few frames of the last generated video and feeding it into the first few frames of the next generation.

## Nodes

### 🍫 Chunker

Creates chunks of `chunk_length` from a resized version of `control_video` and `control_masks` if provided. If `control_video` and `control_masks` are not provided then blank ones are created. 

### 🍫 Combine

Combines all modified chunks into one single sequence.
