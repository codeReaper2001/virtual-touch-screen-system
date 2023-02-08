from typing import Tuple, Callable
import sys
from PyQt5.QtWidgets import QWidget
import PyQt5.QtWidgets as qt
import PyQt5.QtCore as core
import PyQt5.QtGui as qtgui
from PyQt5 import uic
from sqlalchemy import Select
import tensorflow as tf
import numpy as np
import cv2

import util
import gui
import database.ops as ops
import database.schema as schema
from gui.camera import Camera
from gui.interface import TabActivationListener


class TrainThread(core.QThread):
    def __init__(self, db_client: ops.DBClient, model_save_path: str, complete_callback: Callable[[], None]) -> None:
        super().__init__()

        dataset = db_client.get_dataset()
        self.trainer = util.ModelTrainer(
            dataset.data, dataset.labels, dataset.classes_num, model_save_path)
        self.complete_callback = complete_callback

    def run(self) -> None:
        self.trainer.train()
        self.complete_callback()


class TabTrainModel(QWidget, TabActivationListener):
    def __init__(self,
                 db_client: ops.DBClient,
                 detector: util.HandDetector,
                 fps_calc: util.FPSCalculator,
                 model_save_path:str
                 ) -> None:
        super().__init__()
        ui = uic.loadUi("./ui/tab_train_model.ui", self)
        self.camera = Camera(self.camera_callback, 578, 316)
        self.db_client = db_client
        self.detector = detector
        self.fps_calc = fps_calc
        self.model_save_path = model_save_path
        self.classes = []
        self.need_load_model = True

        self.init_ui_elem(ui)

    def load_model_if_needed(self) -> None:
        if self.need_load_model:
            self.model = tf.keras.models.load_model(self.model_save_path)
            self.need_load_model = False
            self.classes = self.db_client.get_gesture_name_list()

    def init_ui_elem(self, ui) -> None:
        self.btn_train_model: qt.QPushButton = ui.btn_train_model
        self.btn_get_train_data: qt.QPushButton = ui.btn_get_train_data
        self.text_debug: qt.QTextBrowser = ui.text_debug

        self.btn_start_test_cap: qt.QPushButton = ui.btn_start_test_cap
        self.label_test_capture: qt.QLabel = ui.label_test_capture

        self.list_trained_gesture: qt.QListView = ui.list_trained_gesture
        self.list_new_gesture: qt.QListView = ui.list_new_gesture

        self.trained_gestures_qtmodel = core.QStringListModel([])
        self.list_trained_gesture.setModel(self.trained_gestures_qtmodel)

        self.new_gestures_qtmodel = core.QStringListModel([])
        self.list_new_gesture.setModel(self.new_gestures_qtmodel)
        self.update_list_show()

        self.bind_slot()

    def bind_slot(self) -> None:
        self.btn_train_model.clicked.connect(self.btn_train_model_click)
        self.btn_start_test_cap.clicked.connect(self.btn_start_test_cap_click)

    def btn_train_model_click(self) -> None:
        if self.new_gestures_qtmodel.rowCount() == 0:
            result = qt.QMessageBox.question(self, "提示", "没有新的手势数据，是否重新训练模型？")
            if result != qt.QMessageBox.StandardButton.Yes:
                return
        self.btn_train_model.setEnabled(False)
        def complete_callback():
            self.text_debug.append("模型训练完成！")
            self.db_client.update_trained_gestures()
            self.update_list_show()
            self.btn_train_model.setEnabled(True)
        self.text_debug.append("模型训练中...")
        self.train_thread = TrainThread(self.db_client, self.model_save_path, complete_callback)
        self.train_thread.start()
        self.need_load_model = True

    def btn_start_test_cap_click(self) -> None:
        if self.btn_start_test_cap.text() == "开始测试":
            self.load_model_if_needed()
            self.camera.open()
            self.btn_start_test_cap.setText("停止测试")
        else:
            self.camera.close()
            self.btn_start_test_cap.setText("开始测试")
            self.label_test_capture.setText("摄像头")

    def update_list_show(self) -> None:
        def trained_condition(g:Select[Tuple[schema.Gesture]]) -> Select[Tuple[schema.Gesture]]:
            return g.where(schema.Gesture.trained == True)
        def new_condition(g:Select[Tuple[schema.Gesture]]) -> Select[Tuple[schema.Gesture]]:
            return g.where(schema.Gesture.trained == False)
        trained_gestures = self.db_client.get_gesture_name_list(trained_condition)
        self.trained_gestures_qtmodel.setStringList(trained_gestures)
        new_gestures = self.db_client.get_gesture_name_list(new_condition)
        self.new_gestures_qtmodel.setStringList(new_gestures)

    def camera_callback(self, img: cv2.Mat) -> None:
        gui.show_fps(self.fps_calc, img)
        self.detect_and_predict(img)
        gui.show_img(self.label_test_capture, img)

    def detect_and_predict(self, img: cv2.Mat) -> None:
        detect_result = self.detector.find_hands(img)
        if not detect_result:
            return
        lm_list = detect_result.get_hand_world_lm_list()
        data = util.flatten_data(lm_list)
        data = np.array(data)
        data = data[None]
        preds = self.model.predict(data)  # type: ignore
        idx = np.argmax(np.squeeze(preds))
        cv2.putText(img, self.classes[idx], (100, 100),
                    cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 2)

    def on_tab_activated(self):
        self.update_list_show()