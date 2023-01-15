# This script is based on different grpc examples for triton server
import tritonclient.grpc as grpcclient
from typing import List
import numpy as np
import sys

from src.utils import (
    draw_boxes,
    letterbox,
    xywh2xyxy,
    bbox_iou,
    scale_boxes,
)

IOU_THRESHOLD = 0.45


class Yolov5_grpc:
    def __init__(
        self,
        url="localhost:8001",
        model_name="yolov5",
        input_width=640,
        input_height=640,
        conf_thresh=0.5,
    ) -> None:
        super(Yolov5_grpc).__init__()
        self.model_name = model_name

        self.input_width = input_width
        self.input_height = input_height
        self.batch_size = 1
        self.conf_thresh = conf_thresh
        self.input_shape = [self.batch_size, 3, self.input_height, self.input_width]
        self.input_name = "images"
        self.output_name = "output0"
        self.output_size = 25200
        self.triton_client = None

        self.fp = "FP32"

        if "16" in self.fp:
            self.np_dtype = np.float16
        else:
            self.np_dtype = np.float32

        self.init_triton_client(url)
        self.test_predict()

    def init_triton_client(self, url):
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

    def test_predict(self):
        input_images = np.zeros((*self.input_shape,), dtype=self.np_dtype)
        _ = self.predict(input_images)

    def predict(self, input_images):
        inputs = []
        outputs = []

        inputs.append(
            grpcclient.InferInput(self.input_name, [*input_images.shape], self.fp)
        )
        # Initialize the data
        inputs[-1].set_data_from_numpy(input_images)
        outputs.append(grpcclient.InferRequestedOutput(self.output_name))

        # Test with outputs
        results = self.triton_client.infer(
            model_name=self.model_name, inputs=inputs, outputs=outputs
        )

        # Get the output arrays from the results
        return results.as_numpy(self.output_name)

    def non_max_suppression(
        self, prediction, origin_h, origin_w, conf_thres=0.5, nms_thres=0.4
    ):
        """
        description: Removes detections with lower object confidence score than 'conf_thres' and performs
        Non-Maximum Suppression to further filter detections.
        param:
            prediction: detections - (center_x, center_y, w, h, obj_conf, classes)
            origin_h: original image height
            origin_w: original image width
            conf_thres: a confidence threshold to filter detections
            nms_thres: a iou threshold to filter detections
        return:
            boxes: output after nms (x1, y1, x2, y2, conf, cls_id)
        """
        # Get the boxes that score > conf_thresh
        boxes = prediction[prediction[:, 4] >= conf_thres]

        # Trandform bbox from [center_x, center_y, w, h] to [x1, y1, x2, y2]
        boxes[:, :4] = xywh2xyxy(boxes[:, :4])
        # clip the coordinates
        boxes[:, 0] = np.clip(boxes[:, 0], 0, origin_w - 1)
        boxes[:, 2] = np.clip(boxes[:, 2], 0, origin_w - 1)
        boxes[:, 1] = np.clip(boxes[:, 1], 0, origin_h - 1)
        boxes[:, 3] = np.clip(boxes[:, 3], 0, origin_h - 1)
        # Object confidence
        confs = boxes[:, 4]

        # Sort by the confs
        boxes = boxes[np.argsort(-confs)]
        # Perform non-maximum suppression
        keep_boxes = []
        while boxes.shape[0]:
            large_overlap = (
                bbox_iou(np.expand_dims(boxes[0, :4], 0), boxes[:, :4]) > nms_thres
            )
            label_match = np.round(boxes[0, -1]) == np.round(boxes[:, -1])
            # Indices of boxes with lower confidence scores, large IOUs and matching labels
            invalid = large_overlap & label_match
            keep_boxes += [boxes[0]]
            boxes = boxes[~invalid]
        boxes = np.stack(keep_boxes, 0) if len(keep_boxes) else np.array([])
        return boxes

    def post_process(self, output, origin_h, origin_w):
        """
        description: postprocess the prediction
        param:
            output:     (n_anchors, n_classes + 5), example: (25200, 85)
            origin_h:   height of original image
            origin_w:   width of original image
        return:
            result_boxes: finally boxes, a boxes numpy, each row is a box [x1, y1, x2, y2]
            result_scores: finally scores, a numpy, each element is the score correspoing to box
            result_classid: finally classid, a numpy, each element is the classid correspoing to box
        """
        # Do nms
        boxes = self.non_max_suppression(
            output,
            origin_h,
            origin_w,
            conf_thres=self.conf_thresh,
            nms_thres=IOU_THRESHOLD,
        )
        result_boxes = boxes[:, :4] if len(boxes) else np.array([])
        result_scores = boxes[:, 4] if len(boxes) else np.array([])
        result_classid = boxes[:, 5] if len(boxes) else np.array([])

        # rescale boxes to original image size from processing size (640x640 -> 1920x1080)
        result_boxes = scale_boxes(
            (self.input_height, self.input_width), result_boxes, (origin_h, origin_w)
        )
        return result_boxes, result_scores, result_classid

    def preprocess(self, img, stride):
        img = letterbox(
            img, max(self.input_width, self.input_height), stride=stride, auto=False
        )[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB
        img = np.ascontiguousarray(img)
        img = img.astype(self.np_dtype)
        img = img / 255.0  # 0 - 255 to 0.0 - 1.0
        img = img.reshape([1, *img.shape])
        return img

    def postprocess(
        self, host_outputs, batch_origin_h, batch_origin_w, min_accuracy=0.5
    ):
        output = host_outputs[0]
        # Do postprocess
        answer = []
        valid_scores = []
        for i in range(self.batch_size):
            result_boxes, result_scores, result_classid = self.post_process(
                output[i * self.output_size : (i + 1) * self.output_size],
                batch_origin_h,
                batch_origin_w,
            )
            for box, score in zip(result_boxes, result_scores):
                if score > min_accuracy:
                    answer.append(box)
                    valid_scores.append(score)
        return answer, valid_scores

    def grpc_detect(
        self, image: np.ndarray, stride: int = 32, min_accuracy: float = 0.5
    ) -> List:
        processed_image = self.preprocess(image, stride)
        pred = self.predict(processed_image)
        boxes, scores = self.postprocess(pred, image.shape[0], image.shape[1])
        return boxes, scores

    def get_boxes_debug(self, image):
        boxes, scores = self.grpc_detect(image)
        debug_image = draw_boxes(image, boxes, scores)
        return boxes, debug_image, scores
