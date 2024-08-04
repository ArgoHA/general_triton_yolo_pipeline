import sys
from typing import List, Tuple

import numpy as np
import torch
import tritonclient.grpc as grpcclient
from loguru import logger
from numpy.typing import NDArray
from src.helpers.ptypes import YOLOScores, main_model_cfg
from src.helpers.utils import letterbox, non_max_suppression, scale_boxes, scale_coords


class Yolov8_grpc:
    def __init__(
        self,
        model_name,
        conf_thresh,
        half: bool = False,
        input_width=640,
        input_height=640,
        url="localhost:8001",
    ) -> None:
        self.model_name = model_name

        self.input_width = input_width
        self.input_height = input_height
        self.batch_size = 1
        self.conf_thresh = conf_thresh
        self.input_shape = [self.batch_size, 3, self.input_height, self.input_width]
        self.input_name = "images"
        self.output_name = "output0"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.iou_thresh = 0.5

        if half:
            self.fp = "FP16"
            self.np_dtype = np.float16
        else:
            self.fp = "FP32"
            self.np_dtype = np.float32

        if "kpt" in self.model_name:
            from src.helpers.ptypes import kpt_shape

            self.task = "kpt"
            self.kpt_shape = kpt_shape
            self.nc = 1
        else:
            self.task = "detect"
            self.nc = main_model_cfg["num_classes"]

        self._init_triton_client(url)
        self._test_predict()

    def _init_triton_client(self, url: str) -> None:
        try:
            triton_client = grpcclient.InferenceServerClient(
                url=url,
                verbose=False,
                ssl=False,
            )
        except Exception as e:
            logger.error(f"Triton channel creation failed: {str(e)}", extra={"error_type": "yolov8_grpc_error"})
            sys.exit()
        self.triton_client = triton_client

    def _test_predict(self) -> None:
        input_images = np.zeros((*self.input_shape,), dtype=self.np_dtype)
        _ = self._predict(input_images)

    def _predict(self, input_images: NDArray) -> torch.tensor:
        inputs = [grpcclient.InferInput(self.input_name, [*input_images.shape], self.fp)]
        inputs[0].set_data_from_numpy(input_images)
        outputs = [grpcclient.InferRequestedOutput(self.output_name)]
        results = self.triton_client.infer(
            model_name=self.model_name, inputs=inputs, outputs=outputs
        )
        return torch.from_numpy(np.copy(results.as_numpy(self.output_name))).to(self.device)

    def _postprocess(self, preds: torch.tensor, origin_h: int, origin_w: int) -> List[YOLOScores]:
        preds_list = non_max_suppression(preds, self.conf_thresh, self.iou_thresh, nc=self.nc)
        results = []
        for pred in preds_list:
            result_dict = {}

            pred[:, :4] = scale_boxes(
                (self.input_height, self.input_width), pred[:, :4], (origin_h, origin_w)
            )

            if self.task == "kpt":
                pred_kpts = (
                    pred[:, 6:].view(len(pred), *self.kpt_shape) if len(pred) else pred[:, 6:]
                )
                result_dict["kpts"] = scale_coords(
                    (self.input_height, self.input_width), pred_kpts, (origin_h, origin_w)
                )

            result_dict["boxes"] = pred[:, :4]
            result_dict["scores"] = pred[:, 4]
            result_dict["class_ids"] = pred[:, 5]

            results.append(YOLOScores(result_dict))
        return results

    def _preprocess(self, img: NDArray, stride: int = 32) -> NDArray:
        img = letterbox(img, max(self.input_width, self.input_height), stride=stride, auto=False)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, then HWC to CHW
        img = np.ascontiguousarray(img, dtype=self.np_dtype)
        img /= 255.0
        img = img.reshape([1, *img.shape])  # Add batch dimension
        return img

    # predict
    def __call__(self, image: NDArray) -> List[YOLOScores]:
        processed_image = self._preprocess(image)
        pred = self._predict(processed_image)
        return self._postprocess(pred, image.shape[0], image.shape[1])
