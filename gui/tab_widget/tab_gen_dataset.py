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
        self.lmdata_generator = util.LmDataGenerator(30)

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

    def btn_start_cap_click(self) -> None:
        startTxt = "开始捕获"
        endTxt = "结束捕获"
        if self.btn_start_cap.text() == startTxt:
            self.camera.open()
            self.btn_start_cap.setText(endTxt)
            self.input_new_gesture.setEnabled(False)
            self.cur_gesture_name = self.input_new_gesture.text()
            self.db_client.add_gesture(self.cur_gesture_name)
        else:
            self.camera.close()
            self.label_capture.setText("摄像头")
            self.btn_start_cap.setText(startTxt)
            self.input_new_gesture.setEnabled(True)

    def camera_callback(self, img: cv2.Mat) -> None:
        gui.show_fps(self.fps_calc, img)
        detect_result = self.detector.find_hands(img)
        if detect_result:
            lm_list = detect_result.get_hand_world_lm_list()
            self.pre_process_data = util.flatten_data(lm_list)
        gui.show_img(self.label_capture, img)

    def keyPressEvent(self, event: qtgui.QKeyEvent):
        if event.key() == core.Qt.Key.Key_A:
            if self.cur_gesture_name == "":
                return
            enhanced_data = self.lmdata_generator.get_enhanced_data([self.pre_process_data], True)
            self.db_client.add_gesture_data(
                self.cur_gesture_name, enhanced_data)
            print("添加数据成功")
