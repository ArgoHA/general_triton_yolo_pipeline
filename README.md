This is a repo with a Triton Server deployment template. As a model example [YOLOv5](https://github.com/ultralytics/yolov5) was chosen, converted to [TensorRT](https://developer.nvidia.com/tensorrt). As a hardware [Nvidia Jetson Nano](https://developer.nvidia.com/embedded/jetson-nano-developer-kit) was used. Pipeline runs on a test video.
This example should easily transfer to other hardware and models. Note, for the model I recommend using [YOLOv8](https://github.com/ultralytics/ultralytics), as it's a newer and better verison of YOLOv5.

### Preperations
- Install nvidia libs for deep learning (nvidia-drivers, cuda toolkit, cudnn)
- Install PyTorch
- Install Triton server
- Install YOLOv5

### Training and export
- Trained Yolov5s with custom dataset and save .pt weights
Example:
```
python train.py --data dataset/dataset.yaml --weights yolov5m.pt --img 640 --batch 40 --epochs 80
```

- Use export.py on your deployment hardware (Jetson nano in this case) to get model.engine (which is the same as model.plan)
Example:
```
python3 export.py --weights yolov5s.pt --include engine --imgsz 640 640 --device 0 # --half
```

### Deployment
- Put model.plan in model_repository/yolov5/1/
- Correct config.pbtxt if needed (if ypu have another input/output for your model). Number of classes should be changed in output dims (number of classes + 5)

### Start triton
I use systemctl to make a service from triton backend, so it is always alive when machine is powered. Use command like this to start triton server:

```
/home/argo/installation_triton/bin/tritonserver --model-repository=/home/argo/general_triton_yolo_pipeline/model_repository/ --backend-directory=/home/argo/installation_triton/backends
```

### Run pipeline with test video
```
python3 main.py
```
