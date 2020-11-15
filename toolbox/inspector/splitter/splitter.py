# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'splitter.ui'
#
# Created by: PyQt5 UI code generator 5.11.2
#
# WARNING! All changes made in this file will be lost!
import sys

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(800, 600)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.viewerScrollArea = QtWidgets.QScrollArea(self.centralwidget)
        self.viewerScrollArea.setGeometry(QtCore.QRect(0, 0, 500, 500))
        self.viewerScrollArea.setFrameShadow(QtWidgets.QFrame.Raised)
        self.viewerScrollArea.setWidgetResizable(True)
        self.viewerScrollArea.setObjectName("viewerScrollArea")
        self.scrollAreaWidgetContents = QtWidgets.QWidget()
        self.scrollAreaWidgetContents.setGeometry(QtCore.QRect(0, 0, 494, 494))
        self.scrollAreaWidgetContents.setObjectName("scrollAreaWidgetContents")
        self.ImageViewer = QtWidgets.QLabel(self.scrollAreaWidgetContents)
        self.ImageViewer.setGeometry(QtCore.QRect(0, 0, 500, 500))
        self.ImageViewer.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.ImageViewer.setFrameShape(QtWidgets.QFrame.Box)
        self.ImageViewer.setFrameShadow(QtWidgets.QFrame.Raised)
        self.ImageViewer.setLineWidth(1)
        self.ImageViewer.setTextFormat(QtCore.Qt.AutoText)
        self.ImageViewer.setAlignment(QtCore.Qt.AlignCenter)
        self.ImageViewer.setWordWrap(False)
        self.ImageViewer.setObjectName("ImageViewer")
        self.viewerScrollArea.setWidget(self.scrollAreaWidgetContents)
        self.subViewer = QtWidgets.QLabel(self.centralwidget)
        self.subViewer.setGeometry(QtCore.QRect(510, 5, 280, 280))
        self.subViewer.setText("")
        self.subViewer.setPixmap(QtGui.QPixmap("../../../../bird.png"))
        self.subViewer.setScaledContents(True)
        self.subViewer.setObjectName("subViewer")
        self.widget = QtWidgets.QWidget(self.centralwidget)
        self.widget.setGeometry(QtCore.QRect(510, 290, 281, 211))
        self.widget.setObjectName("widget")
        self.vBtnLayout = QtWidgets.QVBoxLayout(self.widget)
        self.vBtnLayout.setContentsMargins(0, 0, 0, 0)
        self.vBtnLayout.setObjectName("vBtnLayout")
        self.undoBtn = QtWidgets.QPushButton(self.widget)
        self.undoBtn.setObjectName("undoBtn")
        self.vBtnLayout.addWidget(self.undoBtn)
        self.nextBtn = QtWidgets.QPushButton(self.widget)
        self.nextBtn.setObjectName("nextBtn")
        self.vBtnLayout.addWidget(self.nextBtn)
        self.prevBtn = QtWidgets.QPushButton(self.widget)
        self.prevBtn.setObjectName("prevBtn")
        self.vBtnLayout.addWidget(self.prevBtn)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 800, 40))
        self.menubar.setObjectName("menubar")
        self.menuFile = QtWidgets.QMenu(self.menubar)
        self.menuFile.setObjectName("menuFile")
        self.menuShape = QtWidgets.QMenu(self.menubar)
        self.menuShape.setObjectName("menuShape")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.shape32x32 = QtWidgets.QAction(MainWindow)
        self.shape32x32.setCheckable(True)
        self.shape32x32.setObjectName("shape32x32")
        self.shape64x64 = QtWidgets.QAction(MainWindow)
        self.shape64x64.setCheckable(True)
        self.shape64x64.setObjectName("shape64x64")
        self.actionOpen_a_file = QtWidgets.QAction(MainWindow)
        self.actionOpen_a_file.setObjectName("actionOpen_a_file")
        self.actionOpen_a_directory = QtWidgets.QAction(MainWindow)
        self.actionOpen_a_directory.setObjectName("actionOpen_a_directory")
        self.menuFile.addAction(self.actionOpen_a_file)
        self.menuFile.addAction(self.actionOpen_a_directory)
        self.menuShape.addAction(self.shape32x32)
        self.menuShape.addAction(self.shape64x64)
        self.menubar.addAction(self.menuFile.menuAction())
        self.menubar.addAction(self.menuShape.menuAction())

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.ImageViewer.setText(_translate("MainWindow", "TextLabel"))
        self.undoBtn.setText(_translate("MainWindow", "Undo"))
        self.nextBtn.setText(_translate("MainWindow", "Next"))
        self.prevBtn.setText(_translate("MainWindow", "Previous"))
        self.menuFile.setTitle(_translate("MainWindow", "&File"))
        self.menuShape.setTitle(_translate("MainWindow", "S&hape"))
        self.shape32x32.setText(_translate("MainWindow", "32x32"))
        self.shape64x64.setText(_translate("MainWindow", "64x64"))
        self.actionOpen_a_file.setText(_translate("MainWindow", "Open a file"))
        self.actionOpen_a_directory.setText(_translate("MainWindow", "Open a directory"))


class MainWindow(Ui_MainWindow):
    def __init__(self):
        Ui_MainWindow.__init__(self)



if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)

    ui = Ui_MainWindow()
    w = QtWidgets.QMainWindow()
    ui.setupUi(w)
    w.show()

    sys.exit(app.exec_())
