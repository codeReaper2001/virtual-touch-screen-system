import cv2
import PyQt5.QtGui as gui
import PyQt5.Qt as qt

import util
from gui.tab_widget.tab_gen_dataset import Counter


def show_fps(fps_calc:util.FPSCalculator, img:cv2.Mat):
    fps = fps_calc.get_fps()
    if fps:
        cv2.putText(img, str(int(fps)), (20, 50),
                    cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

def show_img(label_capture: qt.QLabel, img: cv2.Mat):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    showImage = gui.QImage(
        img.data, img.shape[1], img.shape[0], gui.QImage.Format_RGB888)
    label_capture.setPixmap(gui.QPixmap.fromImage(showImage))

def show_count(counter: Counter, img: cv2.Mat):
    display = "{}/{}".format(counter.cnt, counter.target)
    cv2.putText(img, display, (20, 100),
                    cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)