from PyQt5.QtWidgets import QWidget
import PyQt5.QtWidgets as qt
from PyQt5 import uic


class TabEditConfig(QWidget):
    def init_ui_elem(self, ui) -> None:
        self.cbox_op_type: qt.QComboBox = ui.cbox_op_type
        self.input_extra_data: qt.QLineEdit = ui.input_extra_data
        self.input_extra_data: qt.QLineEdit = ui.input_extra_data
        self.cbox_gesture: qt.QComboBox = ui.cbox_gesture
        self.btn_add_op_gesture: qt.QPushButton = ui.btn_add_op_gesture

        self.table_config: qt.QTableView = ui.table_config

    def __init__(self) -> None:
        super().__init__()
        ui = uic.loadUi("./ui/tab_edit_config.ui", self)
        self.init_ui_elem(ui)