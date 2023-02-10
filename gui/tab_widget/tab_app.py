import tensorflow as tf
from PyQt5.QtWidgets import QWidget
import PyQt5.QtWidgets as qt
import PyQt5.QtGui as qtgui
from PyQt5 import uic
import cv2

import gui
import util
import gui.camera as camera
from gui.interface import TabActivationListener
from database import ops

class TabApp(QWidget, TabActivationListener):
    def __init__(self, db_client: ops.DBClient, detector: util.HandDetector, model_save_path:str) -> None:
        super().__init__()
        ui = uic.loadUi("./ui/tab_app.ui", self)
        self.init_ui_elem(ui)
        
        self.camera = gui.Camera(self.camera_callback)
        self.fps_calc = util.FPSCalculator()
        self.detector = detector
        self.db_client = db_client
        self.model_save_path = model_save_path
        model_shape = tf.keras.models.load_model("./model/handwrite_shape_classifier/handwrite_shape_classifier.h5")
        self.app_state_machine = util.AppStateMachine(db_client, detector, (camera.cap_width, camera.cap_height), model_shape)
        # 提前触发
        self.on_tab_activated()
    
    def init_ui_elem(self, ui):
        self.btn_start:qt.QPushButton = ui.btn_start
        self.label_capture:qt.QLabel = ui.label_capture
        self.bind_slot()
        
    def bind_slot(self):
        self.btn_start.clicked.connect(self.btn_start_click)
    
    def btn_start_click(self):
        if self.btn_start.text() == "开启":
            self.camera.open()
            self.btn_start.setText("关闭")
            self.app_state_machine.gesture_state_handler.update_mapping()
        else:
            self.camera.close()
            self.label_capture.setText("摄像头")
            self.btn_start.setText("开启")

    def camera_callback(self, img:cv2.Mat) -> None:
        gui.show_fps(self.fps_calc, img)
        img = self.app_state_machine.img_to_operation(img)
        gui.show_img(self.label_capture, img)

    def on_tab_activated(self) -> None:
        gesture_model = tf.keras.models.load_model(self.model_save_path)
        self.app_state_machine.set_gesture_model(gesture_model)