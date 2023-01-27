import cv2
import numpy as np
from mediapipe.python.solutions import hands
import math
import autopy
import pyautogui
from typing import List

import util

################
cap_width = 640
cap_height = 480
frame_r = 100
smoothening = 6
################

scr_width, scr_height = autopy.screen.size()

smoothen_util = util.SmoothenUtil(smoothening)

is_toggle = False


def to_finger_bitmap(fingers: List[bool]):
    res = 0
    for id, finger in enumerate(fingers):
        if finger:
            res += 1 << (4 - id)
    return res


def do_something(img: cv2.Mat, detector: util.HandDetector):
    # global is_toggle
    index_finger_idx = hands.HandLandmark.INDEX_FINGER_TIP
    mid_finger_idx = hands.HandLandmark.MIDDLE_FINGER_TIP
    thumb_finger_idx = hands.HandLandmark.THUMB_TIP
    lm_list = detector.find_hands(img)
    if not lm_list:
        return
    fingers = detector.fingers_up(lm_list)

    cv2.rectangle(img, (frame_r, frame_r), (cap_width - frame_r, cap_height - frame_r),
                  (255, 0, 255), 2)

    def right_click():
        autopy.mouse.click(autopy.mouse.Button.RIGHT)

    def move():
        fx, fy = lm_list[index_finger_idx].get_data()
        mx = np.interp(fx, (frame_r, cap_width - frame_r), (0, scr_width))
        my = np.interp(fy, (frame_r, cap_height - frame_r), (0, scr_height))
        sx, sy = smoothen_util.get_smooth_val(mx.item(), my.item())
        autopy.mouse.move(sx, sy)

    def left_click():
        global is_toggle
        ifx, ify = lm_list[index_finger_idx].get_data()
        tfx, tfy = lm_list[thumb_finger_idx].get_data()
        length = math.hypot(tfx - ifx, tfy - ify)
        cv2.line(img, (ifx, ify), (tfx, tfy), (255, 0, 0), 2)
        # print(length)
        if length < 20:
            # print("click")
            if is_toggle == False:
                autopy.mouse.toggle(autopy.mouse.Button.LEFT, True)
                is_toggle = True
        else:
            if is_toggle == True:
                autopy.mouse.toggle(autopy.mouse.Button.LEFT, False)
                is_toggle = False

    def scrolling():
        fx, fy = lm_list[index_finger_idx].get_data()
        # print(fy, (cap_height - 2 * frame_r) / 2)
        cap_mid_y = frame_r + (cap_height - 2 * frame_r) / 2
        distance = cap_mid_y - fy
        speed = distance
        speed = 100 if speed > 100 else speed
        speed = -100 if speed < -100 else speed
        speed = int(speed)
        pyautogui.scroll(speed)

    fingerbitmap_operation = {
        0b1000: move,
        0b1100: left_click,
        0b1110: right_click,
        0b1111: scrolling,
    }
    finger_bitmap = to_finger_bitmap(fingers)

    if finger_bitmap in fingerbitmap_operation:
        func = fingerbitmap_operation[finger_bitmap]
        func()


def main():
    global pTime
    cap = cv2.VideoCapture(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cap_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cap_height)
    detector = util.HandDetector(maxHands=1)
    fps_cal = util.FPSCalculator()

    while True:
        success, img = cap.read()
        if not success:
            break

        img = cv2.flip(img, 1)

        do_something(img, detector)

        # fps
        fps = fps_cal.get_fps()
        if fps:
            cv2.putText(img, str(int(fps)), (20, 50),
                        cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

        cv2.imshow("img", img)
        key = cv2.waitKey(5)
        if key == 27:
            break


if __name__ == "__main__":
    main()
