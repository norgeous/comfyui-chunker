# comfyui-chunker

## thoughts on behaviour

| input images | input control_video | input control_masks | output control_video 0 | output control_masks 0         | What you get? why would you need this?                     |
|--------------|---------------------|---------------------|------------------------|--------------------------------|------------------------------------------------------------|
| 0            | 0                   | 0                   | blank                  | all white                      | chunked t2v                                                |
| 0            | 0                   | 1                   | blank                  | copies of mask1                | chunked t2v of non-rectangular shape                       |
| 0            | 1                   | 0                   | cv1 then blank         | all white                      | chunked t2v with starting pose                             |
| 0            | 1                   | 1                   | cv1 then blank         | copies of mask1                | chunked t2v of non-rectangular shape with starting pose    |
| 1            | 0                   | 0                   | im1 then blank         | one black then all white       | chunked i2v                                                |
| 1            | 0                   | 1                   | im1 then blank         | one black then copies of mask1 | chunked i2v of non-rectangular shape (vincents bottle?)    |
| 1            | 1                   | 0                   | cv1 then blank         | all white                      | chunked i2v with starting pose                             |
| 1            | 1                   | 1                   | cv1 then blank         | one white then copies of mask1 | chunked i2v of non-rectangular shape with starting pose    | 
| 0            | 0                   | 2                   | blank                  | chunk of masks                 | chunked t2v of animated non-rectangular shape              |
| 0            | 2                   | 0                   | chunk of cv            | all white                      | chunked v2v with poses                                     |
| 0            | 2                   | 2                   | chunk of cv            | chunk of masks                 | chunked v2v with poses and masks                           |
| 2            | 0                   | 0                   | chunk of im            | all white                      | chunked v2v (denoise video?)                               |
| 2            | 0                   | 2                   | chunk of im            | chunk of masks                 | chunked v2v with mask (denoise specific area? inpainting?) |
| 2            | 2                   | 0                   | chunk of cv            | all white                      | chunked v2v with poses                                     | 
| 2            | 2                   | 2                   | chunk of cv            | chunk of masks                 | chunked v2v with poses and masks                           |
