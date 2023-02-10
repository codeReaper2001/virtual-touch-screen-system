import sys
from PyQt5.QtWidgets import QApplication, QWidget
import PyQt5.QtWidgets as qt

import util
from database import ops
import gui
from gui.interface import TabActivationListener


class MyWindow(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.ui = self.init_ui()
        self.setGeometry(0, 0, 800, 600)
        
    def init_ui(self):
        db_client = ops.DBClient("./database/db/data.db")
        detector = util.HandDetector(maxHands=1)
        fps_calc = util.FPSCalculator()

        gesture_model_path = "./model/sign_classifier/sign_classifier_app.h5"
        
        tab_widget = qt.QTabWidget(self)
        self.tabs = [
            gui.TabApp(db_client, detector, gesture_model_path),
            gui.TabGenDataset(
                db_client, detector, fps_calc
            ), 
            gui.TabTrainModel(
                db_client, detector, fps_calc, gesture_model_path
            ),
            gui.TabEditConfig(db_client),
        ]
        tab_widget.addTab(self.tabs[0], "app")
        tab_widget.addTab(self.tabs[1], "添加新手势数据集")
        tab_widget.addTab(self.tabs[2], "训练模型")
        tab_widget.addTab(self.tabs[3], "配置手势动作")
        tab_widget.currentChanged.connect(self.tab_widget_change)
        layout = qt.QVBoxLayout()
        layout.addWidget(tab_widget)
        self.setLayout(layout)

    def tab_widget_change(self, idx:int):
        listener: TabActivationListener = self.tabs[idx]
        listener.on_tab_activated()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    w = MyWindow()
    # db_client = ops.DBClient("./database/db/data.db")
    # # detector = util.HandDetector(maxHands=1)
    # # w = gui.TabApp(db_client, detector, "./model/sign_classifier/sign_classifier_app.h5")
    # w = gui.TabEditConfig(db_client)

    w.show()

    app.exec()
