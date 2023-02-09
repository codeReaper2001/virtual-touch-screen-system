from typing import List, Callable
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

def gesture_table2model(table: ops.GestureTable) -> qtgui.QStandardItemModel:
    model = qtgui.QStandardItemModel()
    model.setHorizontalHeaderLabels(table.header)
    table_array = table.get_body_array()
    for row in table_array:
        qt_row: List[qtgui.QStandardItem] = []
        for col in row:
            qt_row.append(qtgui.QStandardItem(col))
        model.appendRow(qt_row)
    return model

class TrainThread(core.QThread):
    def __init__(self, db_client: ops.DBClient, model_save_path: str, complete_callback) -> None:
        super().__init__()

        dataset = db_client.get_dataset()
        self.trainer = util.ModelTrainer(
            dataset.data, dataset.labels, dataset.classes_num, model_save_path)
        if complete_callback:
            self.finished.connect(complete_callback)

    def run(self) -> None:
        self.trainer.train()


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

        self.table_gesture: qt.QTableView = ui.table_gesture
        self.update_table_gesture()

        self.bind_slot()

    def bind_slot(self) -> None:
        self.btn_train_model.clicked.connect(self.btn_train_model_click)
        self.btn_start_test_cap.clicked.connect(self.btn_start_test_cap_click)

    def btn_train_model_click(self) -> None:
        result = qt.QMessageBox.question(self, "提示", "确认要训练模型吗？")
        if result != qt.QMessageBox.StandardButton.Yes:
            return
        self.btn_train_model.setEnabled(False)
        self.text_debug.append("模型训练中...")
        self.train_thread = TrainThread(self.db_client, self.model_save_path, self.complete_callback)
        self.train_thread.start()
        self.need_load_model = True

    def complete_callback(self):
        self.text_debug.append("模型训练完成！")
        self.db_client.update_trained_gestures()
        self.update_table_gesture()
        self.btn_train_model.setEnabled(True)

    def btn_start_test_cap_click(self) -> None:
        if self.btn_start_test_cap.text() == "开始测试":
            self.load_model_if_needed()
            self.camera.open()
            self.btn_start_test_cap.setText("停止测试")
        else:
            self.camera.close()
            self.btn_start_test_cap.setText("开始测试")
            self.label_test_capture.setText("摄像头")

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

    def update_table_gesture(self):
        gesture_table = self.db_client.get_gesture_table()
        self.table_gesture_qtmodel = gesture_table2model(gesture_table)
        self.table_gesture.setModel(self.table_gesture_qtmodel)

        vertical_header = self.table_gesture.verticalHeader()
        # 设置行高
        vertical_header.setSectionResizeMode(qt.QHeaderView.ResizeMode.Fixed)
        vertical_header.setDefaultSectionSize(40)

        for i in range(self.table_gesture_qtmodel.rowCount()):
            gesture_id = gesture_table.body[i].gesture_id
            self.add_delete_btn(i, gesture_id)

    def add_delete_btn(self, row: int, gesture_id: int):
        deleteBtn = qt.QPushButton('删除')
        deleteBtn.clicked.connect(lambda: self.btn_delete_click(gesture_id))

        layout = qt.QHBoxLayout()
        layout.addWidget(deleteBtn)
        
        widget = QWidget()
        widget.setLayout(layout)

        self.table_gesture.setIndexWidget(
            self.table_gesture_qtmodel.index(
                row, self.table_gesture_qtmodel.columnCount() - 1),
            widget)

    def btn_delete_click(self, gesture_id: int) -> None:
        gesture = self.db_client.get_gesture(gesture_id)
        result = qt.QMessageBox.question(self, "提示", "确定要删除手势'{}'的数据吗".format(gesture.name))
        if result != qt.QMessageBox.StandardButton.Yes:
            return
        self.db_client.delete_gesture(gesture_id)
        self.update_table_gesture()

    def on_tab_activated(self) -> None:
        self.update_table_gesture()