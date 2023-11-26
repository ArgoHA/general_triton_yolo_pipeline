This is a repo with a Triton Server deployment template. As a model example [YOLOv8](https://github.com/ultralytics/ultralytics) was chosen, converted to [TensorRT](https://developer.nvidia.com/tensorrt). Tested on Nvidia 3060. Pipeline runs on a test video.

### Preperations
- Install nvidia libs for deep learning (nvidia-drivers, cuda toolkit, cudnn, tensorrt)
- Install PyTorch
- Install [Triton inference server](https://docs.nvidia.com/deeplearning/triton-inference-server/archives/triton_inference_server_220/user-guide/docs/install.html) (docker recommended)
- Install Ultralytics

Get your yolov8 detection model weigts exported to engine with [ultralytics docs](https://docs.ultralytics.com/modes/export/)

### Deployment
- Put model.plan in model_repository/yolov8/1/
- Correct config.pbtxt if needed (if ypu have another input/output for your model). Number of classes should be changed in output dims (number of classes + 4)

### Start triton

```
docker run --gpus=all --rm -d -p8000:8000 -p8001:8001 -p8002:8002 -v/model_repo_path:/models nvcr.io/nvidia/tritonserver:23.07-py3 tritonserver --model-repository=/models
```
model_repo_path - full path to your model_repository
23.07-py3 - version of your triton inference server

### Configs

- change name_to_label_mapping to fit your labels
- change video src link (supports webcams, rtsp and just videos)

### Run pipeline with test video
```
python -m src.main
```
