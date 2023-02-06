from typing import Tuple, List
from enum import Enum
import os
import math
import numpy as np
import cv2
import autopy
import pyautogui

import util
from database import ops
from mediapipe.python.solutions import hands

################
pyautogui.PAUSE = 0.005
pyautogui.FAILSAFE = False
frame_r = 100
smoothening = 6
scr_width, scr_height = autopy.screen.size()
################
index_finger_idx = hands.HandLandmark.INDEX_FINGER_TIP
mid_finger_idx = hands.HandLandmark.MIDDLE_FINGER_TIP
thumb_finger_idx = hands.HandLandmark.THUMB_TIP
################

class State(Enum):
    Common = 1
    Draw = 2
    Gesture = 3

class AppStateMachine():
    def __init__(self, detector:util.HandDetector, img_shape:Tuple[int, int], model_shape) -> None:
        # self.db_client = db_client
        self.detector = detector
        self.cap_width, self.cap_height = img_shape
        self.img_canvas = np.zeros((self.cap_height, self.cap_width, 3), np.uint8)
        
        self.draw_color = (255, 0, 255)
        self.brush_thickness = 5

        self.state = State.Common # 初始为Common模式
        self.context_common = AppStateMachine.CommonStateContext()
        self.context_draw = AppStateMachine.DrawStateContext()
        self.model_shape = model_shape

    def img_to_operation(self, img:cv2.Mat) -> cv2.Mat:
        detect_result = self.detector.find_hands(img)
        if not detect_result:
            return img
        lm_list, _ =  detect_result.get_hand_lm_list()
        fingers = util.fingers_up(lm_list)

        finger_bitmap = util.fingerlist_to_finger_bitmap(fingers)
        if self.state is State.Common:
            self.common_state(img, lm_list, finger_bitmap)
        elif self.state is State.Draw:
            img = self.draw_state(img, lm_list, finger_bitmap)
        return img
        
    class CommonStateContext():
        def __init__(self) -> None:
            self.is_right_click = False
            self.smoothen_move = util.SmoothenUtil(smoothening)
            self.is_toggle = False
    
    def common_state(self, img: cv2.Mat, lm_list: List[util.LmData], finger_bitmap: int):
        cv2.rectangle(img, 
            (frame_r, frame_r), 
            (self.cap_width - frame_r, self.cap_height - frame_r),
            (255, 0, 255), 2)

        def right_click():
            if self.context_common.is_right_click:
                return
            autopy.mouse.click(autopy.mouse.Button.RIGHT)
            self.context_common.is_right_click = True

        def move():
            fx, fy, _ = lm_list[index_finger_idx].get_data()
            mx = np.interp(fx, (frame_r, self.cap_width - frame_r), (0, scr_width))
            my = np.interp(fy, (frame_r, self.cap_height - frame_r), (0, scr_height))
            sx, sy = self.context_common.smoothen_move.get_smooth_val(mx.item(), my.item())
            autopy.mouse.move(sx, sy)

        def left_click():
            ifx, ify, _ = lm_list[index_finger_idx].get_data()
            tfx, tfy, _ = lm_list[thumb_finger_idx].get_data()
            length = math.hypot(tfx - ifx, tfy - ify)
            cv2.line(img, (ifx, ify), (tfx, tfy), (255, 0, 0), 2)
            # print(length)
            if length < 20:
                # print("click")
                if self.context_common.is_toggle == False:
                    autopy.mouse.toggle(autopy.mouse.Button.LEFT, True)
                    self.context_common.is_toggle = True
            else:
                if self.context_common.is_toggle == True:
                    autopy.mouse.toggle(autopy.mouse.Button.LEFT, False)
                    self.context_common.is_toggle = False

        def scrolling():
            fx, fy, _ = lm_list[index_finger_idx].get_data()
            cap_mid_y = frame_r + (self.cap_height - 2 * frame_r) / 2
            distance = cap_mid_y - fy
            speed = distance
            speed = 100 if speed > 100 else speed
            speed = -100 if speed < -100 else speed
            speed = int(speed)
            pyautogui.scroll(speed)

        def change2draw():
            self.state = State.Draw

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
            self.context_common.is_right_click = False
        
    class DrawStateContext():
        def __init__(self) -> None:
            self.smoothen_draw = util.SmoothenUtil(3)
            self.has_predict = False
            self.start_x = 100
            self.end_x = 300
            self.start_y = 100
            self.end_y = 300
        
        def reset(self):
            self.start_x = 1000
            self.end_x = 0
            self.start_y = 1000
            self.end_y = 0

    def draw_state(self, img: cv2.Mat, lm_list: List[util.LmData], finger_bitmap: int) -> cv2.Mat:
        cv2.putText(img, "draw mode", (100, 30),
                    cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1)

        def drawing():
            fx, fy, _ = lm_list[index_finger_idx].get_data()
            px, py = self.context_draw.smoothen_draw.get_px_py()
            px, py = int(px), int(py)
            sx, sy = self.context_draw.smoothen_draw.get_smooth_val(fx, fy)
            sx, sy = int(sx), int(sy)
            if sx < self.context_draw.start_x: self.context_draw.start_x = sx
            if sx > self.context_draw.end_x: self.context_draw.end_x = sx
            if sy < self.context_draw.start_y: self.context_draw.start_y = sy
            if sy > self.context_draw.end_y: self.context_draw.end_y = sy
            if px == 0 and py == 0:
                px, py = sx, sy
            cv2.line(self.img_canvas, (px, py), (sx, sy), self.draw_color, self.brush_thickness)

        def reset():
            self.context_draw.smoothen_draw.reset()

        def draw2option():
            if self.context_draw.has_predict:
                return
            start_x, start_y = self.context_draw.start_x, self.context_draw.start_y
            end_x, end_y = self.context_draw.end_x, self.context_draw.end_y
            if start_x > end_x or start_y > end_y:
                return
            shapes = ["circles", "squares", "triangles"]
            draw_part = self.img_canvas[start_y + 5:end_y - 5, start_x + 5:end_x - 5]
            draw_part = cv2.resize(draw_part, (28, 28))
            draw_part = cv2.cvtColor(draw_part, cv2.COLOR_BGR2GRAY)
            _, draw_part = cv2.threshold(draw_part, 50, 255, cv2.THRESH_BINARY_INV)
            # cv2.imshow("draw_part", draw_part)
            draw_part = draw_part[None]
            preds = self.model_shape.predict(draw_part)
            lb_idx = np.argmax(preds)
            label = shapes[lb_idx]
            res = "{}: {:.2f}%".format(lb_idx, preds[0][lb_idx] * 100)
            print(res)
            self.context_draw.has_predict = True
            if label == "triangles":
                os.system("start D:/WeChat/WeChat.exe")
            elif label == "squares":
                os.system('start ""  "D:/学习/翻墙/clash/Clash for Windows.exe"')
            else:
                os.system('start RunDll32.exe user32.dll,LockWorkStation')
            clear_img_canvas()
            self.context_draw.reset()
            change2common()

        def clear_img_canvas():
            self.img_canvas = np.zeros((self.cap_height, self.cap_width, 3), np.uint8)
            self.context_draw.has_predict = False

        def change2common():
            self.state = State.Common

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

        img_gray = cv2.cvtColor(self.img_canvas, cv2.COLOR_BGR2GRAY)
        _, img_inv = cv2.threshold(img_gray, 50, 255, cv2.THRESH_BINARY_INV)
        img_inv = cv2.cvtColor(img_inv, cv2.COLOR_GRAY2BGR)
        img = cv2.bitwise_and(img, img_inv)
        img = cv2.bitwise_or(img, self.img_canvas)
        return img