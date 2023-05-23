## profilerWrapper
A wrapper that wraps all profile tools including timer, memory recorder, data size recorder and dependency recorder.

The output of the wrapper can be used in our NS optimization and visualization repo. 
## How to use
For each layer, wrap it in the following code block:
```python
# Dependency log
prof_wrapper.scale.dependency_check(tensor_name="x2", src="d1_conv2", dest="d1_conv4")
# Memory usage
tmp_input = torch.clone(x2)
with profile(
        activities=
        [
            ProfilerActivity.CPU
        ] if not usingcuda else
        [
            ProfilerActivity.CPU,
            ProfilerActivity.CUDA
        ],
        profile_memory=True, record_shapes=True
) as prof:
    with record_function("model_inference"):
        self.conv4(tmp_input)
prof_report = str(prof.key_averages().table()).split("\n")
prof_wrapper.mr.get_mem("d1_conv4", prof_report, usingcuda)
# Layer time
torch.cuda.synchronize()
prof_wrapper.tt.tic("d1_conv4")
x4 = self.conv4(x2)
prof_wrapper.tt.toc("d1_conv4")
# Data size
prof_wrapper.scale.weight(tensor_src="d1_conv4", data=x4)
```

Note: all testing results move to ./results directory. 

