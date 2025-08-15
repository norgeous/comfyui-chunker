# comfyui-chunker

Create longer Wan VACE videos by automatically taking the last few frames of the last generated video and feeding it into the first few frames of the next generation.

## Nodes

### 🍫 Chunker

Creates chunks of `chunk_length` from a resized version of `control_video` and `control_masks` if provided. If `control_video` and `control_masks` are not provided then blank ones are created. 

| Input          | Required | Default    | Note                                     |
|----------------|----------|------------|------------------------------------------|
| control_video  | NO       | None       | Connect an Image or Images to be chunked |
| control_masks  | NO       | None       |                                          |
| width          | YES      | 480        |                                          |
| height         | YES      | 832        |                                          |
| aspect_ratio   | YES      | keep_input |                                          |
| chunk_length   | YES      | 81         |                                          |
| chunk_overlap  | YES      | 4          |                                          |
| total_length   | YES      | 158        | The minimum length of images output from `🍫 Combine` node |


### 🍫 Combine

Combines all modified chunks into one single sequence.

| Input         | Required | Default | Note                                     |
|---------------|----------|---------|------------------------------------------|
| chunk_info    | YES      | None    |                                          |
| images        | YES      | None    |                                          |
