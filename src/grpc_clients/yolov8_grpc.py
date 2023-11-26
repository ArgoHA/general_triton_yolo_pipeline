import sys
from typing import Dict, List, Tuple

import numpy as np
import torch
import tritonclient.grpc as grpcclient

from src.helpers.ptypes import YOLOScores
from src.helpers.utils import letterbox, non_max_suppression, scale_boxes


class Yolov8_grpc:
    def __init__(
        self,
        url="localhost:8001",
        model_name="yolov8",
        input_width=640,
        input_height=640,
        conf_thresh=0.5,
    ) -> None:
        super(Yolov8_grpc).__init__()
        self.model_name = model_name

        self.input_width = input_width
        self.input_height = input_height
        self.batch_size = 1
        self.conf_thresh = conf_thresh
        self.input_shape = [self.batch_size, 3, self.input_height, self.input_width]
        self.input_name = "images"
        self.output_name = "output0"
        self.output_size = 8400
        self.triton_client = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.iou_thresh = 0.45

        self.fp = "FP32"

        if "16" in self.fp:
            self.np_dtype = np.float16
        else:
            self.np_dtype = np.float32

        self.init_triton_client(url)
        self.test_predict()

    def init_triton_client(self, url: str) -> None:
        try:
            triton_client = grpcclient.InferenceServerClient(
                url=url,
                verbose=False,
                ssl=False,
            )
        except Exception as e:
            print("channel creation failed: " + str(e))
            sys.exit()
        self.triton_client = triton_client

    def test_predict(self) -> None:
        input_images = np.zeros((*self.input_shape,), dtype=self.np_dtype)
        _ = self._predict(input_images)

    def _predict(self, input_images: np.ndarray) -> torch.tensor:
        inputs = [grpcclient.InferInput(self.input_name, [*input_images.shape], self.fp)]
        inputs[0].set_data_from_numpy(input_images)
        outputs = [grpcclient.InferRequestedOutput(self.output_name)]

        results = self.triton_client.infer(
            model_name=self.model_name, inputs=inputs, outputs=outputs
        )
        return torch.from_numpy(np.copy(results.as_numpy(self.output_name))).to(self.device)

    def postprocess(self, preds: torch.tensor, origin_h: int, origin_w: int) -> List[YOLOScores]:
        preds = non_max_suppression(preds, self.conf_thresh, self.iou_thresh)

        results = []
        for pred in preds:
            result_dict = {}
            pred[:, :4] = scale_boxes(
                (self.input_height, self.input_width), pred[:, :4], (origin_h, origin_w)
            )

            result_dict["boxes"] = pred[:, :4].cpu().numpy() if len(pred) else np.array([])
            result_dict["scores"] = pred[:, 4].cpu().numpy() if len(pred) else np.array([])
            result_dict["class_ids"] = pred[:, 5].cpu().numpy() if len(pred) else np.array([])
            results.append(result_dict)
        return results

    def preprocess(self, img: np.ndarray, stride: int = 32) -> np.ndarray:
        img = letterbox(img, max(self.input_width, self.input_height), stride=stride, auto=False)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, then HWC to CHW
        img = np.ascontiguousarray(img, dtype=self.np_dtype)
        img /= 255.0  # Normalize to 0.0 - 1.0
        img = img.reshape([1, *img.shape])  # Add batch dimension
        return img

    def predict(self, image: np.ndarray) -> Tuple[List[YOLOScores], np.ndarray]:
        processed_image = self.preprocess(image)
        pred = self._predict(processed_image)
        return self.postprocess(pred, image.shape[0], image.shape[1])
