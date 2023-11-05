import cv2
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QApplication, QWidget
from PyQt5.QtCore import Qt
# from Inference import Inference
# from Capture import CaptureThread
from queue import Queue
from PyQt5.QtGui import QPixmap, QPainter
import time

from view.CamItem import CamItem
from thread.Capture import CaptureThread
from thread.Inference import Inference

class CamItemController(CamItem):
    def __init__(self, url,trackingAl):
        super().__init__()
        self.setupUi()
        self.url = url
        print(self.url)
        self.cap = CaptureThread(self.url)
        self.InferenceThread = Inference(self.cap.cap_queue, trackingAl)
        self.frame = None
        
        
    def Run(self):
        self.cap.start()
        self.InferenceThread.start()

        self.InferenceThread.signal.connect(self.display)

    

    def display(self, cv_img):
        # qt_img = self.convert_cv_qt(cv_img)  

        # self.label.setPixmap(qt_img)
        # print("----------------------------")
        self.frame = cv_img

    def paintEvent(self, event):

        if self.frame is not None:
            # self.old_frame = current_frame
            rgb_img = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
            qt_img = QPixmap.fromImage(
                QtGui.QImage(rgb_img.data, rgb_img.shape[1], rgb_img.shape[0], QtGui.QImage.Format_RGB888)).scaled(
                self.label.width(), self.label.height())
            self.label.setPixmap(qt_img)
        self.update()
        time.sleep(0.001)

    def convert_cv_qt(self, cv_img):
        rgb_image =cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        # fps, rgb_image = fpsReader.update(rgb_image,pos=(50,80),color=(0,255,0),scale=5,thickness=5)
        h,w,ch = rgb_image.shape
        bytes_per_line = ch * w 
        convert_to_Qt_format = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(1080, 720, Qt.KeepAspectRatio)
        return QtGui.QPixmap.fromImage(p) 