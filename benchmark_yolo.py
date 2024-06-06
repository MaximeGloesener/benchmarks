import torch 
from ultralytics import YOLO
import numpy as np 
import time 
from tqdm import tqdm 
from benchmark import get_num_parameters, get_model_size, get_model_macs


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

Byte = 8
KiB = 1024 * Byte
MiB = 1024 * KiB
GiB = 1024 * MiB


# benchmark all yolo models 
models = ['n','s', 'm', 'l', 'x']

for model_name in models:
    model = YOLO(f'yolov8{model_name}.pt')

    # input data 
    input_data = np.random.rand(224, 224, 3).astype(np.float32) 

    # Warmup runs for gpu
    elapsed = 0.0
    for _ in range(5):
        start_time = time.time()
        for _ in range(20):
            model(input_data, imgsz=224, verbose=False)
        elapsed = time.time() - start_time

    eps = 1e-3
    # Compute number of runs as higher of min_time or num_timed_runs
    num_runs = max(round(60 / (elapsed + eps) * 20), 100 * 50)

    # Timed runs
    run_times = []
    for _ in tqdm(range(num_runs)):
        results = model(input_data, imgsz=224, verbose=False)
        run_times.append(results[0].speed["inference"])  # Convert to milliseconds

    # Compute statistics
    run_times = np.array(run_times)
    mean_time = np.mean(run_times)
    std_time = np.std(run_times)
    print(f"Mean inference time: {mean_time:.2f} ms")
    print(f'FPS GPU: {1000/mean_time:.2f}')

    num_params = get_num_parameters(model)
    print(f'Num param = {num_params/1e6:.2f} M')
    model_size = get_model_size(model)
    print(f'Model size = {model_size/MiB} MiB')
    with open('benchmark_yolo_GTX3060.txt', 'a') as f:
        f.write(f'YOLOv8{model_name}, {mean_time:.2f}, {1000/mean_time:.2f}, {num_params/1e6:.2f}, {model_size/MiB:.2f}\n')