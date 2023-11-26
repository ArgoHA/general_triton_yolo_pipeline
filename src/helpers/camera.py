from threading import Thread
import cv2
import time
import numpy as np
import subprocess
import platform


# Capture rtsp stream
class Camera:
    def __init__(self, src, fps):
        self.cap = cv2.VideoCapture(src)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
        self.delay = 1 / fps

        self.thread = Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()

        self.status = False
        self.frame = None

    def update(self):
        while True:
            if self.cap.isOpened():
                self.status, self.frame = self.cap.read()

    def read(self):
        if self.status:
            return self.frame
        return None

    def wait_for_cam(self):
        for _ in range(5):
            if self.status:
                return True
            else:
                time.sleep(1)
        return False

    def release(self):
        self.cap.release()


# Capture webcam
class VideoStream:
    def __init__(self, src, fps, resolution):
        self.src = src
        self.fps = fps
        self.resolution = resolution
        self.pipe = self._open_ffmpeg()
        self.frame_shape = (self.resolution[1], self.resolution[0], 3)
        self.frame_size = np.prod(self.frame_shape)  # total bytes of a frame

    def _open_ffmpeg(self):
        os_name = platform.system()
        if os_name == "Darwin":  # macOS
            input_format = "avfoundation"
            video_device = f"{self.src}:none"
        elif os_name == "Linux":
            input_format = "v4l2"
            video_device = f"{self.src}"
        elif os_name == "Windows":
            input_format = "dshow"
            video_device = f"video={self.src}"
        else:
            raise ValueError("Unsupported OS")

        command = [
            "ffmpeg",
            "-f",
            input_format,
            "-r",
            str(self.fps),
            "-video_size",
            f"{self.resolution[0]}x{self.resolution[1]}",
            "-i",
            video_device,
            "-vcodec",
            "mjpeg",  # Input codec set to mjpeg
            "-an",
            "-vcodec",
            "rawvideo",  # Decode the MJPEG stream to raw video
            "-pix_fmt",
            "bgr24",
            "-vsync",
            "2",
            "-f",
            "image2pipe",
            "-",
        ]

        if os_name == "Linux":
            command.insert(2, "-input_format")
            command.insert(3, "mjpeg")

        return subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=10**8
        )

    def read(self):
        raw_image = self.pipe.stdout.read(self.frame_size)
        if len(raw_image) != self.frame_size:
            return None
        image = np.frombuffer(raw_image, dtype=np.uint8).reshape(self.frame_shape)
        return image

    def release(self):
        self.pipe.terminate()

    def wait_for_cam(self):
        for _ in range(30):
            frame = self.read()
        if frame is not None:
            return True
        return False

    @property
    def get_resolution(self):
        return self.resolution

    @property
    def get_fps(self):
        return self.fps


class VideoReader:
    def __init__(self, filepath):
        self.cap = cv2.VideoCapture(filepath)

    def read(self):
        status, frame = self.cap.read()
        if status:
            return frame
        return None

    def wait_for_cam(self):
        return True

    def release(self):
        self.cap.release()


def connect_source(src_mode, src_path, fps=30, resolution=(1920, 1080)):
    if src_mode == "webcam":
        cap = VideoStream(src_path, fps, resolution)
    elif src_mode == "rtsp":
        cap = Camera(src_path, fps)
    elif src_mode == "video":
        cap = VideoReader(src_path)

    if not cap.wait_for_cam():
        raise Exception("Can not read the frame")
    return cap
