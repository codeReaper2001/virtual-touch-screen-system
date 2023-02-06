from PyQt5.QtWidgets import QApplication, QWidget
import PyQt5.QtWidgets as qt
import sys

import util
from database import ops
import gui


class MyWindow(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.ui = self.init_ui()
        self.setGeometry(0, 0, 800, 600)
        
    def init_ui(self):
        db_client = ops.DBClient("./database/db/data.db")
        detector = util.HandDetector(maxHands=1)
        fps_calc = util.FPSCalculator()
        
        tabWidget = qt.QTabWidget(self)
        tabWidget.addTab(gui.TabGenDataset(
            db_client, detector, fps_calc), "添加新手势数据集")
        tabWidget.addTab(gui.TabTrainModel(
            db_client, detector, fps_calc, "./model/sign_classifier/keypoint_classifier_app.h5"), "训练模型")
        tabWidget.addTab(gui.TabEditConfig(), "配置手势动作")
        layout = qt.QVBoxLayout()
        layout.addWidget(tabWidget)
        self.setLayout(layout)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    # w = TabGenDataset()
    w = MyWindow()

    w.show()  # type: ignore

    app.exec()
