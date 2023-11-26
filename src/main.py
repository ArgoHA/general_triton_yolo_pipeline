from pathlib import Path
from threading import Thread
import time
import cv2
import hydra
from omegaconf import DictConfig

from src.helpers.camera import connect_source
from src.helpers.utils import draw_results
from src.grpc_clients.yolov8_grpc import Yolov8_grpc


class Pipeline:
    def __init__(
        self,
        src: str,
        output_path: str,
        idx: int,
        src_mode: str,
        device: str,
        cam_fps: int,
        detector_conf: float,
    ):
        self.detector = Yolov8_grpc(conf_thresh=detector_conf)
        self.src = src
        self.idx = idx
        self.device = device
        self.save_images = True

        self.src_mode = src_mode
        self.camera = connect_source(src_mode, src, cam_fps)
        self.output_path = Path(output_path)
        self.output_path.mkdir(exist_ok=True, parents=True)

        self.counter = 0

        print("Starting camera:", idx)

    def save_output(self, frame):
        cv2.imwrite(str(self.output_path / f"frame{self.idx}_{self.counter}.jpg"), frame)

    def run(self):
        while True:
            t0 = time.perf_counter()
            frame = self.camera.read()
            if frame is None:
                break

            results = self.detector.predict(frame)
            if results[0]["boxes"].any() and self.save_images:
                pred_frame = draw_results(results, frame)
                self.save_output(pred_frame)

            self.counter += 1
            time_delta = time.perf_counter() - t0

            if self.src_mode == "rtsp":
                time.sleep(max(0, self.camera.delay - time_delta))


@hydra.main(version_base=None, config_path="../", config_name="config")
def main(cfg: DictConfig) -> None:
    cam_id = 0
    for src_mode in cfg["src"]:
        fps = cfg["src"][src_mode]["fps"]
        for src in cfg["src"][src_mode]["links"]:
            cam_id += 1

            pipeline = Pipeline(
                src=src,
                output_path=cfg["output_path"],
                idx=cam_id,
                src_mode=src_mode,
                device=cfg["device"],
                cam_fps=fps,
                detector_conf=cfg["detector_conf"],
            )
            Thread(target=pipeline.run).start()


if __name__ == "__main__":
    main()
