
import cv2

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QWidget,QApplication, QMainWindow, QFileDialog, QMessageBox, QDesktopWidget , QVBoxLayout, QHBoxLayout, QSizePolicy, QLineEdit
from PyQt5.QtCore import pyqtSignal, QObject, Qt
from PyQt5.QtGui import QImage, QPixmap

from  queue import Queue
# from setup import Ui_Form
# from frame import Frame
import link_cam
# from Capture import CaptureThread
# from Inference import Inference



class Ui_MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        

    def setupUi(self):
        # MainWindow.setObjectName("MainWindow")
        self.resize(800, 600)
        self.centralwidget = QtWidgets.QWidget(self)
        self.centralwidget.setObjectName("centralwidget")

        self.layout_main = QtWidgets.QGridLayout(self.centralwidget)
        self.layout_main.setObjectName("layout_main")

        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(20, 210, 191, 41))
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.label.setFont(font)
        self.label.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(40, 280, 141, 141))
        self.label_2.setStyleSheet("background-color: rgb(7, 7, 7)")
        self.label_2.setText("")
        self.label_2.setScaledContents(False)
        self.label_2.setObjectName("label_2")
        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setGeometry(QtCore.QRect(40, 440, 141, 41))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.pushButton.setFont(font)
        self.pushButton.setObjectName("pushButton")
        self.pushButton_2 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_2.setGeometry(QtCore.QRect(30, 20, 161, 51))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.pushButton_2.setFont(font)
        self.pushButton_2.setObjectName("pushButton_2")
        self.pushButton_3 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_3.setGeometry(QtCore.QRect(30, 80, 161, 51))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.pushButton_3.setFont(font)
        self.pushButton_3.setObjectName("pushButton_3")
        self.pushButton_4 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_4.setGeometry(QtCore.QRect(30, 140, 161, 51))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.pushButton_4.setFont(font)
        self.pushButton_4.setObjectName("pushButton_4")
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setGeometry(QtCore.QRect(90, 310, 41, 81))
        font = QtGui.QFont()
        font.setPointSize(36)
        self.label_3.setFont(font)
        self.label_3.setStyleSheet("color: white")
        self.label_3.setObjectName("label_3")
        self.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(self)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 800, 26))
        self.menubar.setObjectName("menubar")
        self.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(self)
        self.statusbar.setObjectName("statusbar")
        self.setStatusBar(self.statusbar)
        
        self.frame_cameras = QtWidgets.QFrame(self.centralwidget)
        self.frame_layout = QtWidgets.QVBoxLayout()
        self.frame_cameras.setLayout(self.frame_layout)
        self.frame_cameras.move(230, 20)
        self.frame_cameras.resize(300, 200)



        
        self.frame_cameras.setStyleSheet("border: 2px solid black;")
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.frame_cameras.sizePolicy().hasHeightForWidth())
        self.frame_cameras.setSizePolicy(sizePolicy)
        self.frame_cameras.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_cameras.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_cameras.setObjectName("frame_cameras")   


        self.grid_layout = QtWidgets.QGridLayout(self.frame_cameras)
        self.frame_layout.addLayout(self.grid_layout)

        self.retranslateUi()
        QtCore.QMetaObject.connectSlotsByName(self)


        # self.pushButton_2.clicked.connect(self.setup)
        # self.pushButton.clicked.connect(self.test)
        
    def retranslateUi(self):
        _translate = QtCore.QCoreApplication.translate
        self.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.label.setText(_translate("MainWindow", "Số lượng camera"))
        self.pushButton.setText(_translate("MainWindow", "Start"))
        self.pushButton_2.setText(_translate("MainWindow", "Settup"))
        self.pushButton_3.setText(_translate("MainWindow", "Add Cam"))
        self.pushButton_4.setText(_translate("MainWindow", "Config Window"))
        self.label_3.setText(_translate("MainWindow", "0"))
    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Return or event.key() == Qt.Key_Enter:
            self.pushButton.click()
            
    
    def resizeEvent(self, event):
        new_size = event.size()
        new_frame_size = new_size.width() -270, new_size.height() -50
        self.frame_cameras.resize(*new_frame_size)
        grid_rect = self.grid_layout.geometry()
        grid_width = grid_rect.width()
        grid_height = grid_rect.height()
        num_rows = self.grid_layout.rowCount()
        num_cols = self.grid_layout.columnCount()
        if self.cols != 0 and new_size.height() != 0:
            frame_width = grid_width // self.cols
            frame_height = grid_height // self.rows

            # Thiết lập lại kích thước của các Frame
            for frame in self.list_cam:
                frame.setFixedSize(frame_width, frame_height)