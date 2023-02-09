from typing import List
from PyQt5.QtWidgets import QWidget
import PyQt5.QtWidgets as qt
import PyQt5.QtGui as qtgui
import PyQt5.QtCore as core
from PyQt5 import uic

from database import ops
from gui.interface import TabActivationListener
import gui


def table2qt_model(table: ops.OperationTable) -> qtgui.QStandardItemModel:
    model = qtgui.QStandardItemModel()
    model.setHorizontalHeaderLabels(table.header)
    table_array = table.get_body_array()
    for row in table_array:
        qt_row: List[qtgui.QStandardItem] = []
        for col in row:
            qt_row.append(qtgui.QStandardItem(col))
        model.appendRow(qt_row)
    return model


class TabEditConfig(QWidget, TabActivationListener):
    def __init__(self, db_client: ops.DBClient) -> None:
        super().__init__()
        self.db_client = db_client
        ui = uic.loadUi("./ui/tab_edit_config.ui", self)
        self.init_ui_elem(ui)

    def init_ui_elem(self, ui) -> None:
        self.input_operation_name: qt.QLineEdit = ui.input_operation_name
        self.cbox_op_type: qt.QComboBox = ui.cbox_op_type
        self.input_extra_data: qt.QLineEdit = ui.input_extra_data
        self.btn_add_operation: qt.QPushButton = ui.btn_add_operation

        self.table_config: qt.QTableView = ui.table_config
        self.update_table_config()

        self.bind_slot()

    def update_table_config(self) -> None:
        table_data = self.db_client.get_operation_table()
        self.table_config_qtmodel = table2qt_model(table_data)
        self.table_config.setModel(self.table_config_qtmodel)
        vertical_header = self.table_config.verticalHeader()
        # 设置行高
        vertical_header.setSectionResizeMode(qt.QHeaderView.ResizeMode.Fixed)
        vertical_header.setDefaultSectionSize(40)
        # 自适应列宽
        self.table_config.horizontalHeader().\
            setSectionResizeMode(
                qt.QHeaderView.ResizeMode.Stretch
            )

        for i in range(self.table_config_qtmodel.rowCount()):
            operation_id = table_data.body[i].operation_id
            self.add_edit_delete_btn(i, operation_id)

    def bind_slot(self) -> None:
        self.btn_add_operation.clicked.connect(self.btn_add_operation_click)

    def btn_add_operation_click(self) -> None:
        operation_name = self.input_operation_name.text()
        type_name = self.cbox_op_type.currentText()
        extra_data = self.input_extra_data.text()
        self.db_client.add_operation(operation_name, type_name, extra_data)
        self.update_table_config()

    def add_edit_delete_btn(self, row: int, operation_id: int) -> None:
        editBtn = qt.QPushButton('编辑操作')
        editBtn.setStyleSheet(''' text-align : center;
                                    background-color : NavajoWhite;
                                    height : 30px;
                                    border-style: outset;
                                    font : 13px  ''')
        editBtn.clicked.connect(lambda: self.btn_edit_click(operation_id))
        deleteBtn = qt.QPushButton('删除操作')
        deleteBtn.setStyleSheet(''' text-align : center;
                                    background-color : LightCoral;
                                    height : 30px;
                                    border-style: outset;
                                    font : 13px; ''')
        deleteBtn.clicked.connect(lambda: self.btn_delete_click(operation_id))
        layout = qt.QHBoxLayout()
        layout.addWidget(editBtn)
        layout.addWidget(deleteBtn)
        widget = QWidget()
        widget.setLayout(layout)
        self.table_config.setIndexWidget(
            self.table_config_qtmodel.index(
                row, self.table_config_qtmodel.columnCount() - 1),
            widget)

    def btn_edit_click(self, operation_id: int) -> None:
        dialog = gui.DialogEditConfig(self.db_client, operation_id, self.handle_dialog_result)
        dialog.exec()
    
    def btn_delete_click(self, operation_id: int) -> None:
        operation = self.db_client.get_operation(operation_id)
        result = qt.QMessageBox.question(self, "提示", "确定要删除操作'{}'吗".format(operation.name))
        if result != qt.QMessageBox.StandardButton.Yes:
            return
        self.db_client.delete_operation(operation_id)
        self.update_table_config()

    def handle_dialog_result(self) -> None:
        self.update_table_config()

    def on_tab_activated(self) -> None:
        self.update_table_config()