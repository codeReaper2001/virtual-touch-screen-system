import tensorflow as tf
import numpy as np
import cv2
from mediapipe.python.solutions import hands
import math
import os
import autopy
import pyautogui
from typing import List
from enum import Enum

import util

class State(Enum):
    Common = 1
    Draw = 2
    Gesture = 3


pyautogui.PAUSE = 0.005
pyautogui.FAILSAFE = False

index_finger_idx = hands.HandLandmark.INDEX_FINGER_TIP
mid_finger_idx = hands.HandLandmark.MIDDLE_FINGER_TIP
thumb_finger_idx = hands.HandLandmark.THUMB_TIP

model = tf.keras.models.load_model("./model/5_handwrite_shape_plus.h5")
################
cap_width = 640
cap_height = 480
frame_r = 100
smoothening = 6
state = State.Common  # 初始为Common模式
################

scr_width, scr_height = autopy.screen.size()

smoothen_move = util.SmoothenUtil(smoothening)
smoothen_draw = util.SmoothenUtil(3)

is_toggle = False
is_right_click = False

img_canvas = np.zeros((cap_height, cap_width, 3), np.uint8)
xp, yp = 0, 0
draw_color = (255, 0, 255)
brush_thickness = 5


def to_finger_bitmap(fingers: List[bool]):
    res = 0
    for id, finger in enumerate(fingers):
        if finger:
            res += 1 << (4 - id)
    return res


def common_state(img: cv2.Mat, lm_list: List[util.HandDetector.LmData], finger_bitmap: int):
    global is_right_click
    cv2.rectangle(img, (frame_r, frame_r), (cap_width - frame_r, cap_height - frame_r),
                  (255, 0, 255), 2)

    def right_click():
        global is_right_click
        if is_right_click:
            return
        autopy.mouse.click(autopy.mouse.Button.RIGHT)
        is_right_click = True

    def move():
        fx, fy = lm_list[index_finger_idx].get_data()
        mx = np.interp(fx, (frame_r, cap_width - frame_r), (0, scr_width))
        my = np.interp(fy, (frame_r, cap_height - frame_r), (0, scr_height))
        sx, sy = smoothen_move.get_smooth_val(mx.item(), my.item())
        autopy.mouse.move(sx, sy)
        # pyautogui.moveTo(sx, sy)

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
                # pyautogui.mouseDown(button='left')
                is_toggle = True
        else:
            if is_toggle == True:
                autopy.mouse.toggle(autopy.mouse.Button.LEFT, False)
                # pyautogui.mouseUp(button='left')
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

    def change2draw():
        global state
        state = State.Draw

    fingerbitmap_operation = {
        0b1000: move,
        0b1100: left_click,
        0b1001: right_click,
        0b1111: scrolling,
        0b0001: change2draw,
    }

    if finger_bitmap in fingerbitmap_operation:
        func = fingerbitmap_operation[finger_bitmap]
        func()
    
    if finger_bitmap != 0b1001:
        is_right_click = False


has_predict = False

def draw_state(img: cv2.Mat, lm_list: List[util.HandDetector.LmData], finger_bitmap: int) -> cv2.Mat:
    cv2.putText(img, "draw mode", (100, 30),
                cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1)
    start_x, start_y = 100, 100
    end_x, end_y = 300, 300
    cv2.rectangle(img, (start_y, start_x), (end_y, end_x),
                  (255, 0, 255), 2)

    def drawing():
        fx, fy = lm_list[index_finger_idx].get_data()
        px, py = smoothen_draw.get_px_py()
        px, py = int(px), int(py)
        sx, sy = smoothen_draw.get_smooth_val(fx, fy)
        sx, sy = int(sx), int(sy)
        if px == 0 and py == 0:
            px, py = sx, sy
        cv2.line(img_canvas, (px, py), (sx, sy), draw_color, brush_thickness)

    def reset():
        smoothen_draw.reset()

    def draw2option():
        global model, has_predict
        if has_predict:
            return
        shapes = ["circles", "squares", "triangles"]
        draw_part = img_canvas[start_y + 5:end_y - 5, start_x + 5:end_x - 5]
        draw_part = cv2.resize(draw_part, (28, 28))
        draw_part = cv2.cvtColor(draw_part, cv2.COLOR_BGR2GRAY)
        _, draw_part = cv2.threshold(draw_part, 50, 255, cv2.THRESH_BINARY_INV)
        # cv2.imshow("draw_part", draw_part)
        draw_part = draw_part[None]
        preds = model.predict(draw_part) # type: ignore
        lb_idx = np.argmax(preds)
        label = shapes[lb_idx]
        res = "{}: {:.2f}%".format(lb_idx, preds[0][lb_idx] * 100)
        print(res)
        has_predict = True
        if label == "triangles":
            os.system("start D:/WeChat/WeChat.exe")
        elif label == "squares":
            os.system('start ""  "D:/学习/翻墙/clash/Clash for Windows.exe"')
        else:
            os.system('start RunDll32.exe user32.dll,LockWorkStation')
        clear_img_canvas()
        change2common()

    def clear_img_canvas():
        global img_canvas, has_predict
        img_canvas = np.zeros((cap_height, cap_width, 3), np.uint8)
        has_predict = False

    def change2common():
        global state
        state = State.Common

    fingerbitmap_operation = {
        0b1000: reset,
        0b1100: drawing,
        0b1111: draw2option,
        0b0000: clear_img_canvas,
        0b0011: change2common,
    }
    if finger_bitmap in fingerbitmap_operation:
        func = fingerbitmap_operation[finger_bitmap]
        func()

    img_gray = cv2.cvtColor(img_canvas, cv2.COLOR_BGR2GRAY)
    _, img_inv = cv2.threshold(img_gray, 50, 255, cv2.THRESH_BINARY_INV)
    img_inv = cv2.cvtColor(img_inv, cv2.COLOR_GRAY2BGR)
    img = cv2.bitwise_and(img, img_inv)
    img = cv2.bitwise_or(img, img_canvas)
    return img


def do_something(img: cv2.Mat, detector: util.HandDetector) -> cv2.Mat:
    # global is_toggle
    lm_list = detector.find_hands(img)
    if not lm_list:
        return img
    fingers = detector.fingers_up(lm_list)
    finger_bitmap = to_finger_bitmap(fingers)
    if state is State.Common:
        common_state(img, lm_list, finger_bitmap)
    elif state is State.Draw:
        img = draw_state(img, lm_list, finger_bitmap)
    return img


def main():
    global pTime, img_canvas
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

        img = do_something(img, detector)

        # fps
        fps = fps_cal.get_fps()
        if fps:
            cv2.putText(img, str(int(fps)), (20, 50),
                        cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

        cv2.imshow("img", img)
        # cv2.imshow("img_canvas", img_canvas)
        key = cv2.waitKey(5)
        if key == 27:
            break


if __name__ == "__main__":
    main()
