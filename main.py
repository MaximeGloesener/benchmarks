# idea = iterate over all pytorch models and benchmark them
# then create a streamlit app to display the results
# models will be benchmarked on different devices -> in streamlite, it should be possible to choose device and see benchmark 

import torch
from torchvision import models
import numpy as np
import torchvision.models as models
from pytorch_bench import benchmark

Byte = 8
KiB = 1024 * Byte
MiB = 1024 * KiB
GiB = 1024 * MiB

# Iterate over all models 
models_list = []
for model_name in dir(models):
    if model_name.islower() and model_name[0] != '_':
        models_list.append(model_name)

for model_name in models_list:
    model_class = getattr(models, model_name)
    if callable(model_class):
        try:
            torch.cuda.empty_cache() # empty cache to be sure 
            model = model_class()
            print(f"Successfully instantiated {model_name}")
            results = benchmark(model, torch.randn(1, 3, 224, 224), n_warmup=50, n_test=200)
            # write into file to display in streamlit
            with open('GTX3060.txt', 'a') as f:
                f.write(f'{model_name.capitalize()},{float(results["fps_cpu"]):.2f},{float(results["fps_gpu"]):.2f},{results["num_parameters"]/1e6:.2f},{int(results["model_size"]/MiB)},{int(results["num_macs"]/1e6)},{float(results["max_memory_used"])}\n')
        except Exception as e:
            print(f"Failed to instantiate {model_name}: {e}")

