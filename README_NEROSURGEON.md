## Profile Yolov4

1. To run the profiling, download the pretrained yolox_x.pth weight file (300+ MB) from the original Yolox repo ([link](https://github.com/bubbliiiing/yolox-pytorch/releases/download/v1.0/yolox_x.pth)) and store the file to [model_data](model_data) directory. Please refer to the original README.md for more info about the weights. 
2. In [nets/](nets), run

```shell
python3 test.py
```

See [README_PROF_WRAPPER.md](README_PROF_WRAPPER.md) for more info about the profiling tool. 