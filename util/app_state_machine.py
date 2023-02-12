from typing import Tuple, List
from enum import Enum
import time
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
frame_r = 120
smoothening = 6
scr_width, scr_height = autopy.screen.size()
################
index_finger_idx = hands.HandLandmark.INDEX_FINGER_TIP
mid_finger_idx = hands.HandLandmark.MIDDLE_FINGER_TIP
thumb_finger_idx = hands.HandLandmark.THUMB_TIP
m_idx = hands.HandLandmark.INDEX_FINGER_MCP
################

class State(Enum):
    Common = 1
    Draw = 2
    Gesture = 3

class CommonStateHandler():
    def __init__(self, m: "AppStateMachine") -> None:
        self.m = m
        self.is_right_click = False
        self.smoothen_move = util.SmoothenUtil(smoothening)
        self.is_toggle = False

    def handle(self, img: cv2.Mat, lm_list: List[util.LmData], finger_bitmap: int):
        cv2.rectangle(img, 
            (frame_r, frame_r), 
            (self.m.cap_width - frame_r, self.m.cap_height - frame_r),
            (255, 0, 255), 2)

        def right_click():
            if self.is_right_click:
                return
            autopy.mouse.click(autopy.mouse.Button.RIGHT)
            self.is_right_click = True

        def move():
            ifx, ify, _ = lm_list[index_finger_idx].get_data()
            tfx, tfy, _ = lm_list[thumb_finger_idx].get_data()
            fx, fy, _ = lm_list[m_idx].get_data()
            # fx, fy = (ifx + tfx) / 2, (ify + tfy) / 2
            mx = np.interp(fx, (frame_r, self.m.cap_width - frame_r), (0, scr_width))
            my = np.interp(fy, (frame_r, self.m.cap_height - frame_r), (0, scr_height))
            sx, sy = self.smoothen_move.get_smooth_val(mx.item(), my.item())
            autopy.mouse.move(sx, sy)

            length = math.hypot(tfx - ifx, tfy - ify)
            cv2.line(img, (ifx, ify), (tfx, tfy), (255, 0, 0), 2)
            if length < 20:
                # print("click")
                if self.is_toggle == False:
                    autopy.mouse.toggle(autopy.mouse.Button.LEFT, True)
                    self.is_toggle = True
            else:
                if self.is_toggle == True:
                    autopy.mouse.toggle(autopy.mouse.Button.LEFT, False)
                    self.is_toggle = False

        def left_click():
            ifx, ify, _ = lm_list[index_finger_idx].get_data()
            tfx, tfy, _ = lm_list[thumb_finger_idx].get_data()
            length = math.hypot(tfx - ifx, tfy - ify)
            cv2.line(img, (ifx, ify), (tfx, tfy), (255, 0, 0), 2)
            # print(length)
            if length < 20:
                # print("click")
                if self.is_toggle == False:
                    autopy.mouse.toggle(autopy.mouse.Button.LEFT, True)
                    self.is_toggle = True
            else:
                if self.is_toggle == True:
                    autopy.mouse.toggle(autopy.mouse.Button.LEFT, False)
                    self.is_toggle = False

        def scrolling():
            fx, fy, _ = lm_list[m_idx].get_data()
            cap_mid_y = frame_r + (self.m.cap_height - 2 * frame_r) / 2
            distance = cap_mid_y - fy
            speed = distance
            speed = 100 if speed > 100 else speed
            speed = -100 if speed < -100 else speed
            speed = int(speed)
            pyautogui.scroll(speed)

        def change2draw():
            self.m.state = State.Draw

        def change2gesture():
            self.m.state = State.Gesture

        fingerbitmap_operation = {
            0b1000: move,
            0b1100: left_click,
            0b1110: left_click,
            0b1001: right_click,
            0b1111: scrolling,
            0b0001: change2draw,
            0b0111: change2gesture,
        }

        if finger_bitmap in fingerbitmap_operation:
            func = fingerbitmap_operation[finger_bitmap]
            func()
        
        if finger_bitmap != 0b1001:
            self.is_right_click = False

class DrawStateHandler():
    def __init__(self, m: "AppStateMachine") -> None:
        self.smoothen_draw = util.SmoothenUtil(3)
        self.has_predict = False
        self.start_x = 100
        self.end_x = 300
        self.start_y = 100
        self.end_y = 300
        self.m = m
        self.img_canvas = np.zeros((self.m.cap_height, self.m.cap_width, 3), np.uint8)
        self.draw_color = (255, 0, 255)
        self.brush_thickness = 5

    def reset(self):
        self.start_x = 1000
        self.end_x = 0
        self.start_y = 1000
        self.end_y = 0

    def handle(self, img: cv2.Mat, lm_list: List[util.LmData], finger_bitmap: int) -> cv2.Mat:
        cv2.putText(img, "draw mode", (100, 30),
                    cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1)

        def drawing():
            fx, fy, _ = lm_list[index_finger_idx].get_data()
            px, py = self.smoothen_draw.get_px_py()
            px, py = int(px), int(py)
            sx, sy = self.smoothen_draw.get_smooth_val(fx, fy)
            sx, sy = int(sx), int(sy)
            if sx < self.start_x: self.start_x = sx
            if sx > self.end_x: self.end_x = sx
            if sy < self.start_y: self.start_y = sy
            if sy > self.end_y: self.end_y = sy
            if px == 0 and py == 0:
                px, py = sx, sy
            cv2.line(self.img_canvas, (px, py), (sx, sy), self.draw_color, self.brush_thickness)

        def reset():
            self.smoothen_draw.reset()

        def draw2option():
            if self.has_predict:
                return
            start_x, start_y = self.start_x, self.start_y
            end_x, end_y = self.end_x, self.end_y
            if start_x > end_x or start_y > end_y:
                return
            shapes = ["圆形", "正方形", "三角形"]
            draw_part = self.img_canvas[start_y + 5:end_y - 5, start_x + 5:end_x - 5]
            draw_part = cv2.resize(draw_part, (28, 28))
            draw_part = cv2.cvtColor(draw_part, cv2.COLOR_BGR2GRAY)
            _, draw_part = cv2.threshold(draw_part, 50, 255, cv2.THRESH_BINARY_INV)
            # cv2.imshow("draw_part", draw_part)
            draw_part = draw_part[None]
            preds = self.m.model_shape.predict(draw_part)
            lb_idx = np.argmax(preds)
            shape_name = shapes[lb_idx]
            res = "{}: {:.2f}%".format(lb_idx, preds[0][lb_idx] * 100)
            print(res)
            self.has_predict = True

            # 执行操作
            operation = self.m.db_client.get_shape_operation(shape_name)
            util.excute_operation(operation)
            clear_img_canvas()
            self.reset()
            change2common()

        def clear_img_canvas():
            self.img_canvas = np.zeros((self.m.cap_height, self.m.cap_width, 3), np.uint8)
            self.has_predict = False

        def change2common():
            self.m.state = State.Common

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

class GestureStateHandler():
    def __init__(self, m: "AppStateMachine") -> None:
        self.m = m
        self.start = False
        self.pre_gesture: str = ""
        self.gesture_list: List[str] = []
        self.gestures_operation_mapping = self.m.db_client.get_gestures_operation_mapping()

    def update_mapping(self):
        self.gestures_operation_mapping = self.m.db_client.get_gestures_operation_mapping()

    def change2common(self):
        self.m.state = State.Common

    def handle(self, img: cv2.Mat, world_lm_list:List[util.LmData]):
        data = util.flatten_data(world_lm_list)
        data = np.array(data)
        data = data[None]
        preds = self.m.gesture_model.predict(data)  # type: ignore
        idx = np.argmax(np.squeeze(preds))
        gesture_name = self.m.classes[idx]
        # 展示手势名
        cv2.putText(img, gesture_name, (50, 100),
                    cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 2)
        time.sleep(0.2)
        if gesture_name == "OK":
            self.start = True
            self.pre_gesture = gesture_name
            self.gesture_list.clear()
            return
        self.inner_handle(gesture_name)

    def inner_handle(self, cur_gesture: str):
        if not self.start:
            return
        if cur_gesture == self.pre_gesture:
            return
        # 新手势
        self.gesture_list.append(cur_gesture)
        print(self.gesture_list)
        self.pre_gesture = cur_gesture

        gesture_name_list_str = "+".join(self.gesture_list)
        if gesture_name_list_str in self.gestures_operation_mapping:
            operation = self.gestures_operation_mapping[gesture_name_list_str]
            util.excute_operation(operation)
            self.start = False
            self.change2common()
        

class AppStateMachine():
    def __init__(self, 
        db_client: ops.DBClient, 
        detector:util.HandDetector, 
        img_shape:Tuple[int, int], 
        model_shape
    ) -> None:
        self.db_client = db_client
        self.detector = detector
        self.cap_width, self.cap_height = img_shape

        self.state = State.Common # 初始为Common模式
        self.common_state_handler = CommonStateHandler(self)
        self.draw_state_handler = DrawStateHandler(self)
        self.gesture_state_handler = GestureStateHandler(self)
        self.model_shape = model_shape

        self.classes:List[str] = []
        self.gesture_model = None

    def img_to_operation(self, img:cv2.Mat) -> cv2.Mat:
        detect_result = self.detector.find_hands(img)
        if not detect_result:
            return img
        lm_list, _ =  detect_result.get_hand_lm_list()
        world_lm_list = detect_result.get_hand_world_lm_list()
        fingers = util.fingers_up_new(world_lm_list)

        finger_bitmap = util.fingerlist_to_finger_bitmap(fingers)
        if self.state is State.Common:
            self.common_state_handler.handle(img, lm_list, finger_bitmap)
        elif self.state is State.Draw:
            img = self.draw_state_handler.handle(img, lm_list, finger_bitmap)
        elif self.state is State.Gesture:
            self.gesture_state_handler.handle(img, detect_result.get_hand_world_lm_list())
        return img

    def set_gesture_model(self, gesture_model):
        self.gesture_model = gesture_model
        self.classes = self.db_client.get_gesture_name_list()

    