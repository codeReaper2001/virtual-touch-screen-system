from typing import Callable
import PyQt5.QtCore as core
import cv2

##################
CAPTURE_NO = 1
cap_width = 640
cap_height = 480
##################

class Camera():
    cap = cv2.VideoCapture(CAPTURE_NO)
    is_open = False

    def __init__(self, func: Callable[[cv2.Mat], None], width=cap_width, height=cap_height) -> None:
        self.width = width
        self.height = height
        self.timer = core.QTimer()
        self.timer.timeout.connect(self.timeout_func)
        self.func = func

    def timeout_func(self) -> None:
        success, img = self.cap.read()
        if not success:
            print("error")
            return
        img = cv2.flip(img, 1)
        self.func(img)

    def open(self) -> bool:
        if Camera.is_open:
            return False
        Camera.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        Camera.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        Camera.is_open = True
        self.timer.start(10)
        return True

    def close(self) -> None:
        Camera.is_open = False
        self.timer.stop()