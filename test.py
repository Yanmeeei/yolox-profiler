import time

import torch
from yolo import decode_outputs
from nets.yolo import YoloBody
from nets.profilerwrapper import ProfilerWrapper

prof_wrapper = ProfilerWrapper()

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)
if device != 'cpu':
    usingcuda = True
else:
    usingcuda = False


model = YoloBody(num_classes=20, phi='s')
model.to(device)
print("YOLOX is Ready")

n = 1
img = torch.randn(n, 3, 640, 640).to(device)

preds = model.forward(img)

t0 = time.time()
outputs = decode_outputs(preds, img.shape)
t1 = time.time()

print(f"decode: {t1-t0}ms")

prof_wrapper.scale.report()
prof_wrapper.tt.report()
prof_wrapper.mr.report()