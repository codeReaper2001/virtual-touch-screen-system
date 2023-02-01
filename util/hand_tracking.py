import cv2
import time
import math
import numpy as np
from typing import List
from mediapipe.python.solutions import hands
from mediapipe.python.solutions import drawing_utils


class AttrDisplay:
    def gatherAttrs(self):
        return ",".join("{}={}"
                        .format(k, getattr(self, k))
                        for k in self.__dict__.keys())

    def __str__(self):
        return "<{}:{}>".format(self.__class__.__name__, self.gatherAttrs())


class HandDetector():
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5) -> None:
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mp_hands = hands
        self.hands = hands.Hands(
            self.mode, self.maxHands, 1, self.detectionCon, self.trackCon)
        self.tip_ids = [4, 8, 12, 16, 20]

    def __find_hands(self, img: cv2.Mat, draw=True):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.hands.process(img_rgb)

        if not results.multi_hand_landmarks:  # type: ignore
            return (results, False)

        if not draw:
            return (results, True)

        for hand_lms in results.multi_hand_landmarks:  # type: ignore
            drawing_utils.draw_landmarks(
                img, hand_lms, hands.HAND_CONNECTIONS)  # type: ignore
        return (results, True)

    class LmData(AttrDisplay):
        def __init__(self, id: int, x: int, y: int) -> None:
            self.id = id
            self.x = x
            self.y = y

        def get_data(self):
            return (self.x, self.y)

    def find_hands(self, img: cv2.Mat, hand_NO=0, draw=True):
        x_list = []
        y_list = []
        lm_list: List[HandDetector.LmData] = []
        result, find = self.__find_hands(img, draw)
        if not find:
            return
        my_hand = result.multi_hand_landmarks[hand_NO]  # type: ignore
        for id, lm in enumerate(my_hand.landmark):
            h, w, c = img.shape
            cx, cy = int(lm.x * w), int(lm.y * h)
            x_list.append(cx)
            y_list.append(cy)
            lm_list.append(self.LmData(id, cx, cy))

        xmin, xmax = min(x_list), max(x_list)
        ymin, ymax = min(y_list), max(y_list)

        if draw:
            cv2.rectangle(img, (xmin - 20, ymin - 20),
                          (xmax + 20, ymax + 20), 2)

        return lm_list

    def fingers_up(self, lm_list: List[LmData]):
        fingers: List[bool] = []

        # if lm_list[self.tip_ids[0]].x < lm_list[self.tip_ids[0] - 1].x:
        #     fingers.append(True)
        # else:
        #     fingers.append(False)
        fingers.append(False)
        for id in range(1, 5):
            if lm_list[self.tip_ids[id]].y < lm_list[self.tip_ids[id] - 2].y:
                fingers.append(True)
            else:
                fingers.append(False)
        return fingers


def do_something(img: cv2.Mat, detector: HandDetector):
    img = cv2.flip(img, 1)
    lm_list = detector.find_hands(img)
    cv2.imshow("img", img)
    if lm_list == None:
        return
    fingers = detector.fingers_up(lm_list)
    print(fingers)


def main():
    cap = cv2.VideoCapture(1)
    detector = HandDetector()
    while True:
        success, img = cap.read()
        if not success:
            break
        do_something(img, detector)
        key = cv2.waitKey(5)
        if key == 27:
            break


if __name__ == "__main__":
    main()
