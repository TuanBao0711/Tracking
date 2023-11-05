from view.Setup import SetupView

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QWidget,QApplication, QMainWindow, QFileDialog, QMessageBox, QDesktopWidget , QVBoxLayout, QHBoxLayout, QSizePolicy, QLineEdit
from PyQt5.QtCore import pyqtSignal, QObject
import math
from sympy import isprime


def recommend_row_col(index):
    if index >2 and isprime(index):
        index += 1
    dictionary = {}
    for i in range(1, index + 1):
        if index % i == 0:
            dictionary[i + index // i] = [i, index // i]
    return dictionary[min(dictionary.keys())]



class SetupController(SetupView):
    slCamSignal = pyqtSignal(int,int,int, int)
    def __init__(self):
        super().__init__()
        self.rows = 0
        self.cols = 0
        
        
    def connect_signals(self):
        self.lineEdit.textChanged.connect(self.slot_cameras_count_changed)
        self.lineEdit.returnPressed.connect(self.pushButton.click)
        self.pushButton.clicked.connect(self.Confirm)
        self.pushButton_2.clicked.connect(self.Cancel)
    
    def slot_cameras_count_changed(self, text_):
        if text_.isdigit():
            cam_count = int(text_)
            if cam_count:
                self.rows, self.cols = recommend_row_col(cam_count)
                self.lineEdit_2.setText(str(self.rows))
                self.lineEdit_3.setText(str(self.cols))
        elif not text_:
            pass
        else:
            QtWidgets.QMessageBox.warning(None, "Cảnh báo", "Yêu cầu nhập số tự nhiên")

    def Confirm(self):
        if self.lineEdit.text().isdigit() and self.rows and self.cols:
            if self.rows and self.cols:
                self.slCamSignal.emit(int(self.lineEdit_3.text()), int(self.lineEdit_2.text()), int(self.lineEdit.text()), int(self.comboBox.currentIndex()))
                
            self.pushButton.parent().close()
        else:
            mess = QMessageBox()
            mess.setWindowTitle('Lỗi')
            mess.setText('Chưa nhập số Camera kìa thằng ngu!!!')
            mess.setIcon(QMessageBox.Warning)  #Critical, Warning, Information, Question
            mess.setStandardButtons(QMessageBox.Ok)
            mess.setDefaultButton(QMessageBox.Ok)
            x = mess.exec_()
    def Cancel(self):
        self.pushButton_2.parent().close()
    
    

