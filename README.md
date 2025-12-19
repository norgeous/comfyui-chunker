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
- Optionally use the end of last chunk as start of this chunk (overlap).

### 🍫 Combine

- Combine sequential chunks back into single tensors

### 🍫 VACE To First Last

- Convert an i2v VACE control_video into First Last frames
- A fully grey first or last image in the input control_video will result in `None` output for the corresponding `clip_vision_` and `_image`

## prep for release
- combined audio sounds glitchy
- KSampler with seed connected to values causes Combine to error
- remove extra nodes
- Tidy unused code
- revise readme and samples
- finalise a few workflows with animated preview
  - chunker-mmaudio
  - chunker-sam3
  - chunker-wan21-vace
  - chunker-wan22-flf2v
- test everything
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

## Known bugs

- VACE sometimes rejects the size (esp. some smaller sizes)
- When using Load Image with mask connected, but no mask drawn, then Chunker throws error
- Can't have 2x sequencial divide+combine in same workflow because random ComfyUI execution order means out of vram error is likely