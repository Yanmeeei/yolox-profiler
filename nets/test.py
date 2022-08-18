import time

import torch
from yolo_general import decode_outputs
from torch.profiler import profile, record_function, ProfilerActivity

from yolo import YoloBody
from profilerwrapper import ProfilerWrapper


prof_wrapper = ProfilerWrapper()

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)
if device != 'cpu':
    usingcuda = True
else:
    usingcuda = False


model = YoloBody(num_classes=20, phi='s').to(device)
print("model to cuda")
print("YOLOX is Ready")

n = 1
img = torch.randn(n, 3, 640, 640).to(device)

preds = model.forward(img, prof_wrapper)
#
# with profile(
#         activities=
#         [
#             ProfilerActivity.CPU
#         ] if not torch.cuda.is_available() else
#         [
#             ProfilerActivity.CPU,
#             ProfilerActivity.CUDA
#         ],
#         profile_memory=True, record_shapes=True, with_flops=True
# ) as prof:
#     with record_function("model_inference"):
#         outputs = decode_outputs(preds, img.shape, prof_wrapper)
# prof_report = str(prof.key_averages().table()).split("\n")
# prof_wrapper.mr.get_mem("output_decode", prof_report, torch.cuda.is_available())
#
outputs = decode_outputs(preds, img.shape, prof_wrapper)
t0 = time.time()
outputs = decode_outputs(preds, img.shape, prof_wrapper)
t1 = time.time()


print(f"decode warmed: {t1-t0}ms")
prof_wrapper.tt.get_time("output_decode", t1-t0)
prof_wrapper.scale.dependency_check("output_decode", "output_decode", "output")

prof_wrapper.scale.report()
prof_wrapper.tt.report()
prof_wrapper.mr.report()