import cv2
from typing import List, Tuple
from mediapipe.python.solutions import hands
from mediapipe.python.solutions import drawing_utils
import numpy as np



class AttrDisplay:
    def gatherAttrs(self):
        return ",".join("{}={}"
                        .format(k, getattr(self, k))
                        for k in self.__dict__.keys())

    def __str__(self):
        return "<{}:{}>".format(self.__class__.__name__, self.gatherAttrs())


class LmData(AttrDisplay):
    def __init__(self, id: int, x: float, y: float, z: float = 0) -> None:
        self.id = id
        self.x = x
        self.y = y
        self.z = z

    def get_data(self) -> Tuple[float, float, float]:
        return (self.x, self.y, self.z)


def fingers_up(lm_list: List[LmData]) -> List[bool]:
    tip_ids = [4, 8, 12, 16, 20]
    fingers: List[bool] = []

    # if lm_list[self.tip_ids[0]].x < lm_list[self.tip_ids[0] - 1].x:
    #     fingers.append(True)
    # else:
    #     fingers.append(False)
    fingers.append(False)
    for id in range(1, 5):
        if lm_list[tip_ids[id]].y < lm_list[tip_ids[id] - 2].y:
            fingers.append(True)
        else:
            fingers.append(False)
    return fingers

def get_degree(v1: np.ndarray, v2: np.ndarray):
    cosangle = v1.dot(v2)/(np.linalg.norm(v1) * np.linalg.norm(v2))
    angle = np.arccos(cosangle)
    degree = np.degrees(angle)
    return degree

def fingers_up_new(world_lm_list: List[LmData]) -> List[bool]:
    fingers: List[bool] = []

    fingers.append(False)
    id1 = [8, 12, 16, 20]
    id2 = [6, 10, 14, 18]
    id3 = [5, 9, 13, 17]
    for i in range(len(id1)):
        p1 = np.array(world_lm_list[id1[i]].get_data())
        p2 = np.array(world_lm_list[id2[i]].get_data())
        p3 = np.array(world_lm_list[id3[i]].get_data())

        v1 = p1 - p2
        v2 = p3 - p2
        degree = get_degree(v1, v2)
        if degree > 120:
            fingers.append(True)
        else:
            fingers.append(False)

    return fingers


class DetectResult():
    def __init__(self, results, img_w, img_h) -> None:
        self.results = results
        self.img_w = img_w
        self.img_h = img_h

    def get_hand_lm_list(self, hand_NO=0) -> Tuple[List[LmData], Tuple[int, int, int, int]]:
        x_list: List[int] = []
        y_list: List[int] = []
        lm_list: List[LmData] = []
        my_hand = self.results.multi_hand_landmarks[hand_NO]
        for id, lm in enumerate(my_hand.landmark):
            cx, cy = int(lm.x * self.img_w), int(lm.y * self.img_h)
            x_list.append(cx)
            y_list.append(cy)
            lm_list.append(LmData(id, cx, cy))

        xmin, xmax = min(x_list), max(x_list)
        ymin, ymax = min(y_list), max(y_list)

        return lm_list, (xmin, ymin, xmax, ymax)

    def get_hand_world_lm_list(self, hand_NO=0) -> List[LmData]:
        my_hand = self.results.multi_hand_world_landmarks[hand_NO]
        lm_list: List[LmData] = []
        for id, lm in enumerate(my_hand.landmark):
            lm_list.append(LmData(id, lm.x, lm.y, lm.z))
        return lm_list


class HandDetector():
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5) -> None:
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mp_hands = hands
        self.hands = hands.Hands(
            self.mode, self.maxHands, 1, self.detectionCon, self.trackCon)

    def find_hands(self, img: cv2.Mat, draw=True):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.hands.process(img_rgb)

        if not results.multi_hand_landmarks:  # type: ignore
            return None
        h, w, c = img.shape

        detect_result = DetectResult(results, w, h)
        if not draw:
            return detect_result

        for hand_lms in results.multi_hand_landmarks:  # type: ignore
            drawing_utils.draw_landmarks(
                img, hand_lms, hands.HAND_CONNECTIONS)  # type: ignore
        return detect_result


def do_something(img: cv2.Mat, detector: HandDetector):
    img = cv2.flip(img, 1)
    detect_result = detector.find_hands(img)
    cv2.imshow("img", img)
    if detect_result == None:
        return
    lm_list, _ = detect_result.get_hand_lm_list()
    fingers = fingers_up(lm_list)
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
