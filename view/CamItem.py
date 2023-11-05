import cv2
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QApplication, QWidget
from PyQt5.QtCore import Qt
# from Inference import Inference
# from Capture import CaptureThread
from queue import Queue
from PyQt5.QtGui import QPixmap, QPainter
import time


class CamItem(QWidget):

    def __init__(self):
        super().__init__()
        # self.setupUi()

        
    def setupUi(self):
        self.setObjectName("Form")
        self.resize(405, 315)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.sizePolicy().hasHeightForWidth())
        self.setSizePolicy(sizePolicy)
        self.label = QtWidgets.QLabel(self)
        self.label.setGeometry(QtCore.QRect(0, 0, 401, 311))
        self.label.setText("")
        # self.label.setPixmap(QtGui.QPixmap("../Project_team/test.jpg"))
        self.label.setScaledContents(True)
        self.label.setObjectName("label")

        self.retranslateUi()
        QtCore.QMetaObject.connectSlotsByName(self)

    def retranslateUi(self):
        _translate = QtCore.QCoreApplication.translate
        self.setWindowTitle(_translate("self", "self"))

    def resizeEvent(self, event):
        new_size = event.size()
        new_frame_size = new_size.width() -15, new_size.height() -5
        self.label.resize(*new_frame_size)