# comfyui-chunker

https://github.com/user-attachments/assets/587530f3-1752-46dd-9156-9343fa2ab16d

Create longer Wan VACE videos by automatically taking the last few frames of the last generated video and feeding it into the first few frames of the next generation.

![image](./workflows/chunker-wan21vace-t2v.png)

## Nodes

### 🍫 Chunker

Creates chunks of `chunk_length` from a resized version of `control_video` and `control_masks` if provided. If `control_video` and `control_masks` are not provided then blank ones are created. 

| Input          | Required | Default      | Note                                                       |
|----------------|----------|--------------|------------------------------------------------------------|
| control_video  | NO       | None         | None, Single Image or Images to be chunked                 |
| control_masks  | NO       | None         | None, Single Mask or Masks to be chunked                   |
| width          | YES      | 480          | Width of the output control_video and control_masks        |
| height         | YES      | 832          | Height of the output control_video and control_masks       |
| aspect_ratio   | YES      | `keep_input` | `keep_input` = use width and height as megapixel density and retain original aspect ratio<br>`stretch_to_new` = stretch to exact size specified<br>`crop_to_new` = scale and crop to exact specified size |
| chunk_length   | YES      | 81           | Count of images in each chunk                              |
| chunk_overlap  | YES      | 4            | Count of images to overlap between chunks                  |
| total_length   | YES      | 158          | The minimum length of images output from `🍫 Combine` node |


### 🍫 Combine

Combines all modified chunks into one sequence.

| Input         | Required | Default | Note                                         |
|---------------|----------|---------|----------------------------------------------|
| chunk_info    | YES      | None    | Connect chunk_info from Chunker node to here |
| images        | YES      | None    | Processed chunk of images                    |
