from PyQt5 import QtCore, QtGui, QtWidgets
from controller.Homecontroller import Homecontroller
import sys

if __name__ == "__main__":
    
    app = QtWidgets.QApplication(sys.argv)
    
    ui = Homecontroller()
    # ui.setupUi(MainWindow)
    ui.show()
    sys.exit(app.exec_())

        