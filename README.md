# comfyui-chunker

## thoughts on behaviour

| input images | input control_video | input control_masks | output control_video   | output control_masks           | What you get?                                              |  why would you need this / potential use case                 |
|--------------|---------------------|---------------------|------------------------|--------------------------------|------------------------------------------------------------|---------------------------------------------------------------|
| 0            | 0                   | 0                   | blank                  | all white                      | chunked t2v                                                | t2v                                                           |
| 0            | 0                   | 1                   | blank                  | copies of mask1                | chunked t2v of non-rectangular shape                       | t2v circle shaped video? to be tested...                      |
| 0            | 1                   | 0                   | cv1 then blank         | all white                      | chunked t2v with starting pose                             | t2v with starting pose                                        |
| 0            | 1                   | 1                   | cv1 then blank         | copies of mask1                | chunked t2v of non-rectangular shape with starting pose    | t2v circle shaped video w/ starting pose? to be tested...     |

| 1            | 0                   | 0                   | im1 then blank         | one black then all white       | chunked i2v                                                | i2v                                                           |
| 1            | 0                   | 1                   | im1 then blank         | one black then copies of mask1 | chunked i2v of non-rectangular shape                       | i2v animated component in static image                        |
| 1            | 1                   | 0                   | cv1 then blank         | all white                      | chunked i2v with starting pose                             | i2v with starting pose                                        |
| 1            | 1                   | 1                   | cv1 then blank         | one white then copies of mask1 | chunked i2v of non-rectangular shape with starting pose    | i2v animated component in static image w/ starting pose       |

| 0            | 0                   | 2                   | blank                  | chunk of masks                 | chunked t2v of animated non-rectangular shape              | t2v                                                           |
| 0            | 2                   | 0                   | chunk of cv            | all white                      | chunked v2v with poses                                     | v2v                                                           |
| 0            | 2                   | 2                   | chunk of cv            | chunk of masks                 | chunked v2v with poses and masks                           | v2v                                                           |
| 2            | 0                   | 0                   | chunk of im            | all white                      | chunked v2v                                                | v2v denoise                                                   |
| 2            | 0                   | 2                   | chunk of im            | chunk of masks                 | chunked v2v with mask                                      | v2v inpainting, outpainting or denoise specific area in video |
| 2            | 2                   | 0                   | chunk of cv            | all white                      | chunked v2v with poses                                     |  |
| 2            | 2                   | 2                   | chunk of cv            | chunk of masks                 | chunked v2v with poses and masks                           |  |
