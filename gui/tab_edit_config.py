from PyQt5.QtWidgets import QWidget
import PyQt5.QtWidgets as qt
import PyQt5.QtGui as qtgui
from PyQt5 import uic

from database import ops
from .interface import TabActivationListener


class TabEditConfig(QWidget, TabActivationListener):
    def __init__(self, db_client:ops.DBClient) -> None:
        super().__init__()
        self.db_client = db_client
        ui = uic.loadUi("./ui/tab_edit_config.ui", self)
        self.init_ui_elem(ui)

    def init_ui_elem(self, ui) -> None:
        self.input_operation_name: qt.QLineEdit = ui.input_operation_name
        self.cbox_op_type: qt.QComboBox = ui.cbox_op_type
        self.input_extra_data: qt.QLineEdit = ui.input_extra_data
        self.btn_add_operation: qt.QPushButton = ui.btn_add_operation

        self.cbox_operations: qt.QComboBox = ui.cbox_operations
        self.cbox_gestures: qt.QComboBox = ui.cbox_gestures
        self.input_gestures: qt.QLineEdit = ui.input_gestures
        self.btn_bind_op_gesture: qt.QPushButton = ui.btn_bind_op_gesture

        self.table_config: qt.QTableView = ui.table_config

        self.bind_slot()

        self.update_cbox_operations_show()
        self.update_cbox_gestures_show()
    
    def bind_slot(self) -> None:
        self.btn_add_operation.clicked.connect(self.btn_add_operation_click)
        self.cbox_gestures.activated[str].connect(self.cbox_gestures_activated) #type: ignore
        self.btn_bind_op_gesture.clicked.connect(self.btn_bind_op_gesture_click)

    def btn_add_operation_click(self) -> None:
        operation_name = self.input_operation_name.text()
        type_name = self.cbox_op_type.currentText()
        extra_data = self.input_extra_data.text()
        self.db_client.add_operation(operation_name, type_name, extra_data)
        self.update_cbox_operations_show()

    def focusInEvent(self, e: qtgui.QFocusEvent) -> None:
        print("focus")
        return super().focusInEvent(e)

    def update_cbox_operations_show(self) -> None:
        operations = self.db_client.get_operations()
        operation_name_list = list(map(lambda op: op.name, operations))
        self.cbox_operations.clear()
        self.cbox_operations.addItems(operation_name_list)

    def update_cbox_gestures_show(self) -> None:
        gestures = self.db_client.get_gesture_name_list()
        self.cbox_gestures.clear()
        self.cbox_gestures.addItems(gestures)

    def cbox_gestures_activated(self, text:str) -> None:
        input_text = self.input_gestures.text()
        if input_text != "":
            input_text += ", "
        input_text += text
        self.input_gestures.setText(input_text)

    def btn_bind_op_gesture_click(self) -> None:
        operation_name = self.cbox_operations.currentText()
        gestures = self.input_gestures.text()
        gesture_list = gestures.split(", ")
        self.db_client.operation_gestures_binding(operation_name, gesture_list)

    def on_tab_activated(self):
        self.update_cbox_gestures_show()
        self.update_cbox_operations_show()