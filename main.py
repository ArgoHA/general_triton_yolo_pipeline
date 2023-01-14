import cv2
from pathlib import Path

from src.yolov5_grpc import Yolov5_grpc
from src.utils import fps_counter


class Video_stream:
    def __init__(self, src):
        self.cap = cv2.VideoCapture(src)

    def read(self):
        ret, frame = self.cap.read()
        if ret:
            return frame


class Pipeline:
    def __init__(
        self, src: str, detector_thres: float = 0.5, save_images: bool = False
    ):
        self.detector_thres = detector_thres
        self.save_images = save_images
        self.root_path = Path(__file__).parent.absolute()
        self.images_path_save = self.root_path / "images"

        self.camera = Video_stream(src)
        self.detector = Yolov5_grpc(conf_thresh=detector_thres)
        self.create_images_folder()
        self.idx = 0
        self.running = True

    def create_images_folder(self):
        Path(self.images_path_save).mkdir(parents=True, exist_ok=True)

    def save_output(self, pred_frame):
        output_path = (self.images_path_save / f"image_{self.idx}").with_suffix(".jpeg")
        cv2.imwrite(str(output_path), pred_frame)

    @fps_counter
    def _runner(self):
        frame = self.camera.read()
        if frame is None:
            self.running = False
            return

        boxes, pred_frame, _ = self.detector.get_boxes_debug(frame)
        if boxes and self.save_images:
            self.save_output(pred_frame)

        self.idx += 1

    def run(self):
        while self.running:
            self._runner()


def main():
    src = "test_vid.mp4"
    detector_thres = 0.5
    save_images = True

    Pipeline(src, detector_thres, save_images).run()


if __name__ == "__main__":
    main()
