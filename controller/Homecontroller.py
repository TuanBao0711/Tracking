from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QWidget,QApplication, QMainWindow, QFileDialog, QMessageBox, QDesktopWidget , QVBoxLayout, QHBoxLayout, QSizePolicy, QLineEdit
from PyQt5.QtCore import pyqtSignal, QObject, Qt
from PyQt5.QtGui import QImage, QPixmap
import cv2

from view.Setup import SetupView 
from view.Home import Ui_MainWindow
from view.CamItem import CamItem
from controller.SetupController import SetupController
from controller.CamItemcontroller import CamItemController
import link_cam


class Homecontroller(Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi()
        self.list_cam = []
        self.list_url_cam = []
        self.list_queue = []
        self.list_queue_Infer = []
        self.rows = 0
        self.cols = 0
        self.slCam = 0
        self.trackingAl = None

        self.link_cam = link_cam.load_link()
    
        self.pushButton_2.clicked.connect(self.setup)
        self.pushButton.clicked.connect(self.run)
        
    def setup(self):
        self.setupWindow = QMainWindow()
        self.setupUI = SetupController()
        self.setupUI.setupUi(self.setupWindow)
        self.setupWindow.show()
        self.setupUI.connect_signals()
        self.setupUI.slCamSignal.connect(self.Config)
        
    def Config(self, rows, cols, slCam, trackingAl):
        for frame in self.list_cam:
            self.grid_layout.removeWidget(frame)
            frame.deleteLater()
        self.list_cam = []
        
        self.trackingAl = trackingAl
        self.rows = rows
        self.cols = cols
        self.slCam = slCam
        self.label_3.setText(str(slCam))
        self.setFrame()
        
        
    def setFrame(self):    
        self.grid_layout.setHorizontalSpacing(10)
        self.grid_layout.setVerticalSpacing(10)
        self.grid_layout.setContentsMargins(5, 5, 5, 5)

        for i in range(self.slCam):  
            cam = 'cam'+str(i+1)
            # frame.label.setPixmap(QtGui.QPixmap(self.link_cam[cam]))
            url_video = self.link_cam[cam]
            self.list_url_cam.append(url_video)
            Cam = CamItemController(url_video,self.trackingAl)
            self.list_cam.append(Cam)
            video_Cap = cv2.VideoCapture(url_video)
            video_Cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = video_Cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # Tạo QImage từ khung hình
                height, width, channel = frame.shape
                bytes_per_line = 3 * width
                q_image = QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888)
                Cam.label.setPixmap(QtGui.QPixmap(q_image))

            self.grid_layout.addWidget(Cam, i//self.cols, i%self.cols)
        
        self.pushButton.setDefault(True)
        
        
    def run(self):
        for i in range(self.slCam):
            self.list_cam[i].Run()