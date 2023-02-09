from typing import List, Callable
import typing
from PyQt5.QtWidgets import QWidget
import PyQt5.QtWidgets as qt
import PyQt5.QtGui as qtgui
import PyQt5.QtCore as core
from PyQt5 import uic

from database import ops, schema

def _handle_default() -> None:
    pass

class DialogEditConfig(qt.QDialog):
    def __init__(
        self, 
        db_client: ops.DBClient, 
        operation_id: int, 
        result_handler: Callable[[], None] = _handle_default,
        ) -> None:
        super().__init__()
        self.db_client = db_client
        self.handle_result = result_handler
        ui = uic.loadUi("./ui/dialog_edit_config.ui", self)
        self.init_ui_elem(ui)
        self.operation_id = operation_id
        self.operation = self.db_client.get_operation(operation_id)
        self.update_elem_content()

        self.bind_slot()

    def init_ui_elem(self, ui) -> None:
        self.input_operation_name: qt.QLineEdit = ui.input_operation_name
        self.input_operation_type: qt.QLineEdit = ui.input_operation_type
        self.input_extra_data: qt.QLineEdit = ui.input_extra_data
        self.cbox_gestures: qt.QComboBox = ui.cbox_gestures
        self.input_gestures: qt.QLineEdit = ui.input_gestures
        self.cbox_shape: qt.QComboBox = ui.cbox_shape
        # 确认按钮
        self.btn_bind: qt.QPushButton = ui.btn_bind

    def update_elem_content(self) -> None:
        operation = self.operation
        self.input_operation_name.setText(operation.name)
        self.input_operation_type.setText(operation.operation_type.type_name)
        self.input_extra_data.setText(operation.extra_data)
        self.update_gestures()
        self.update_cbox_shape()

    def update_gestures(self) -> None:
        operation = self.operation
        # cbox
        self.cbox_gestures.clear()
        gestures = self.db_client.get_gesture_name_list()
        self.cbox_gestures.addItems(gestures)
        if len(operation.gesture_seqence) > 0:
            self.cbox_gestures.setCurrentText(operation.gesture_seqence[-1].gesture.name)
        # input
        gesture_list_str = "+".join(list(map(lambda g: g.gesture.name,
                                    operation.gesture_seqence)))
        self.input_gestures.setText(gesture_list_str)

    def update_cbox_shape(self) -> None:
        operation = self.operation
        self.cbox_shape.addItems(['', '三角形', '正方形', '圆形'])
        print(operation.shape)
        if operation.shape == None:
            return
        self.cbox_shape.setCurrentText(operation.shape.name)
    
    def bind_slot(self) -> None:
        self.cbox_gestures.activated[str].connect( # type: ignore
            self.cbox_gestures_activated
        )
        self.btn_bind.clicked.connect(self.btn_bind_clicked)
        
    def cbox_gestures_activated(self, text: str) -> None:
        input_text = self.input_gestures.text()
        if input_text != "":
            input_text += "+"
        input_text += text
        self.input_gestures.setText(input_text)

    def btn_bind_clicked(self) -> None:
        operation_name = self.input_operation_name.text()
        extra_data = self.input_extra_data.text()
        self.db_client.update_operation(self.operation_id, operation_name, extra_data)

        gestures = self.input_gestures.text()
        gesture_list = gestures.split('+')
        self.db_client.operation_gestures_binding(self.operation_id, gesture_list)

        shape_name = self.cbox_shape.currentText()
        self.db_client.operation_shape_binding(self.operation_id, shape_name)

        self.handle_result()
        self.close()

