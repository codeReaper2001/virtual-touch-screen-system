from PyQt5.QtWidgets import QWidget
import PyQt5.QtWidgets as qt
import PyQt5.QtCore as core
import PyQt5.QtGui as qtgui
from PyQt5 import uic
from typing import List
import cv2

import util
from database import ops
import gui
from gui.interface import TabActivationListener

class Counter():
    def __init__(self, target: int) -> None:
        self.cnt = 0
        self.target = target
    def increase(self) -> bool:
        self.cnt += 1
        if self.cnt == self.target:
            return True
        return False
    def reset(self):
        self.cnt = 0

class TabGenDataset(QWidget, TabActivationListener):
    def __init__(self, db_client: ops.DBClient, detector: util.HandDetector, fps_calc: util.FPSCalculator) -> None:
        super().__init__()
        self.setFocusPolicy(core.Qt.FocusPolicy.StrongFocus)

        ui = uic.loadUi("./ui/tab_gen_dataset.ui", self)
        self.init_ui_elem(ui)
        self.camera = gui.Camera(self.camera_callback)
        self.detector = detector
        self.fps_calc = fps_calc
        self.db_client = db_client
        self.counter = Counter(25)
        self.lmdata_generator = util.LmDataGenerator(30)
        # 是否支持双手
        self.add_flip: bool = False

        self.pre_process_data: List[float] = []
        # 当前手势
        self.cur_gesture_name: str = ""

    def init_ui_elem(self, ui) -> None:
        self.btn_start_cap: qt.QPushButton = ui.btn_start_cap
        self.cbox_flip: qt.QCheckBox = ui.cbox_flip
        self.input_new_gesture: qt.QLineEdit = ui.input_new_gesture
        self.label_capture: qt.QLabel = ui.label_capture
        self.bind_slot()

    def bind_slot(self) -> None:
        self.btn_start_cap.clicked.connect(self.btn_start_cap_click)

    # 点击开始捕获
    def btn_start_cap_click(self) -> None:
        self.cur_gesture_name = self.input_new_gesture.text()
        if self.cur_gesture_name == "":
            qt.QMessageBox.information(self, "info", "手势名称不能为空")
            return
        gesture = self.db_client.get_gesture_by_name(self.cur_gesture_name)
        if gesture != None:
            qt.QMessageBox.information(self, "info", "该手势名称已存在，请换个手势名称")
            return
        self.add_flip = self.cbox_flip.isChecked()
        self.camera.open()
        self.input_new_gesture.setEnabled(False)
        self.btn_start_cap.setEnabled(False)
        self.db_client.add_gesture(self.cur_gesture_name)
        self.counter.reset()

    def camera_callback(self, img: cv2.Mat) -> None:
        gui.show_fps(self.fps_calc, img)
        gui.show_count(self.counter, img)
        detect_result = self.detector.find_hands(img)
        if detect_result:
            lm_list = detect_result.get_hand_world_lm_list()
            self.pre_process_data = util.flatten_data(lm_list)
        gui.show_img(self.label_capture, img)

    def keyPressEvent(self, event: qtgui.QKeyEvent):
        if event.key() == core.Qt.Key.Key_A:
            if self.cur_gesture_name == "":
                return
            enhanced_data = self.lmdata_generator.get_enhanced_data([self.pre_process_data], self.add_flip)
            self.db_client.add_gesture_data(
                self.cur_gesture_name, enhanced_data)
            print("添加数据成功")
            is_over = self.counter.increase()
            if not is_over:
                return
            qt.QMessageBox.information(self, "info", "数据集采集成功")
            self.camera.close()
            self.label_capture.setText("摄像头")
            self.input_new_gesture.setEnabled(True)
            self.input_new_gesture.setText("")
            self.btn_start_cap.setEnabled(True)
