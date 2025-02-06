import cv2
import numpy as np


class VideoFrameIterator:
    def __init__(self, video_file: str, stride: int = 1) -> None:
        self.cap: cv2.VideoCapture | None = None
        self.stride = stride
        self.frame_height: int
        self.frame_width: int
        self.expected_frame_count: int
        self.fps: float

        self.cap = cv2.VideoCapture(video_file)
        assert self.cap.isOpened()
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.expected_frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)

    def reset(self, stride: int = 1) -> "VideoFrameIterator":
        if self.cap is not None:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self.stride = stride
        return self

    def __len__(self):
        return self.expected_frame_count // self.stride

    def __iter__(self):
        return self

    def __next__(self) -> np.ndarray:
        if self.cap is None:
            raise StopIteration

        ret, frame = self.cap.read()
        if not ret:
            raise StopIteration
        if self.stride > 1:
            current_frame_idx = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame_idx + self.stride - 1)

        return frame.copy()

    def __getitem__(self, idx: int) -> np.ndarray:
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        assert int(self.cap.get(cv2.CAP_PROP_POS_FRAMES)) == idx
        ret, frame = self.cap.read()
        assert ret
        return frame.copy()

    def __del__(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None
