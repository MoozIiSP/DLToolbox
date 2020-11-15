import sys

from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *


class Viewer(QLabel):
    pos: int = None
    drawPoint: bool = False
    drawLine: bool = False
    drawRect: bool = True
    drawTempRect: bool = False
    imageLoaded: bool = False
    boxes: list = []
    magic: int = 127
    __statusbar: QStatusBar = None

    def __init__(self):
        QLabel.__init__(self)

    def __init__(self, text) -> None:
        """usage"""
        QLabel.__init__(self, text)

    def paintEvent(self, a0: QPaintEvent):
        QLabel.paintEvent(self, a0)

        # Multi-Layer Painting
        painter = QPainter(self)
        painter.setPen(Qt.black)
        painter.setFont(QFont("Noto Sans", 21))
        if self.boxes:
            for box in self.boxes:
                x0 = box.topLeft().x()
                y0 = box.topLeft().y()
                x1 = box.bottomRight().x()
                y1 = box.bottomRight().y()
                print(x1-x0, y1-y0)
                painter.drawRect(box)
        if self.drawTempRect:
            painter.drawRect(self.rect_position())

    def mousePressEvent(self, ev: QMouseEvent):
        if ev.button() == Qt.LeftButton:
            self._showMessage("at ({:4d}, {:4d}).".format(ev.x(), ev.y()))
            if self.drawRect:
                # Get first point of rectangle.
                self.pos = ev.pos()
                self.drawTempRect = True
        elif ev.button() == Qt.RightButton:
            # Pop last element
            if self.boxes:
                self.boxes.pop()
            self.update()

    def mouseReleaseEvent(self, ev: QMouseEvent):
        if ev.button() == Qt.LeftButton:
            self._showMessage("at ({:4d}, {:4d}).".format(ev.x(), ev.y()))
            if self.drawRect:
                self.boxes.append(self.rect_position())
                # Clear All
                self.pos = None
                self.update()
                self.drawTempRect = False

    def mouseMoveEvent(self, ev: QMouseEvent):
        self._showMessage("at ({:4d}, {:4d}).".format(ev.x(), ev.y()))
        if self.drawRect:
            self.pos = ev.pos()
            self.update()

    def rect_position(self):
        if not self.underMouse():
            self._showMessage("Mouse not in Viewer.")
            return QRect(0, 0, 0, 0)
        x, y = self.pos.x(), self.pos.y()
        ex = self.magic // 2
        if self.magic % 2 == 0:
            return QRect(QPoint(x - ex, y - ex),
                         QPoint(x + ex, y + ex))
        else:
            return QRect(QPoint(x - ex, y - ex),
                         QPoint(x + ex + 1, y + ex + 1))

    def binding_statusbar(self, statusbar):
        self.__statusbar = statusbar

    def _showMessage(self, msg):
        self.__statusbar.showMessage(msg)

    def get_rect_list(self):
        return self.boxes


class MainWindow(QMainWindow):
    def __init__(self):
        QMainWindow.__init__(self)

        # Layout
        self.centralWidget = QWidget()
        self.centralWidget.setObjectName('CentralWidget')
        self.layout = QHBoxLayout()
        self.layout.setObjectName('MainLayout')
        self.sl_viewer = QVBoxLayout()
        self.sl_viewer.setObjectName('ViewerLayout')
        self.sl_watcher = QVBoxLayout()
        self.sl_watcher.setObjectName('WatcherLayout')

        # Widgets
        self.label = Viewer('This is a Label')
        self.btn = QPushButton('Load Image')
        self.roi_viewer = QLabel("Viewer")
        self.listView = QListView()

        # Misc
        self.btn.clicked.connect(self.btnPressEvent)
        self.label.binding_statusbar(self.statusBar())
        self.label.setText("Put a image on Screen.")

        # Setting
        # = Layout bindings
        self.setCentralWidget(self.centralWidget)
        self.centralWidget.setLayout(self.layout)
        self.layout.addLayout(self.sl_viewer)
        self.layout.addLayout(self.sl_watcher)
        # = Widget bindings
        self.sl_viewer.addWidget(self.label)
        self.sl_viewer.addWidget(self.btn)
        self.sl_watcher.addWidget(self.roi_viewer)
        self.sl_watcher.addWidget(self.listView)
        # = Oth
        self.label.setFixedSize(500, 500)
        self.label.setAlignment(Qt.AlignCenter)
        self.roi_viewer.setFixedSize(300, 300)
        self.roi_viewer.setAlignment(Qt.AlignCenter)

        self.statusBar().showMessage("Initialization finished.")

        self.im = QPixmap()

        # Meta-Object
        obj = self.label.metaObject()
        print(obj)
        print(obj.className())
        print(obj.property(0).type())


    # def mousePressEvent(self, a0: QMouseEvent):
    #     self.statusBar().showMessage("At ({:4d}, {:4d})".format(a0.x(),
    #                                                             a0.y()))
    #
    # def mouseReleaseEvent(self, a0: QMouseEvent):
    #     self.statusBar().showMessage("At ({:4d}, {:4d})".format(a0.x(),
    #                                                             a0.y()))
    #
    # def mouseMoveEvent(self, a0: QMouseEvent):
    #     self.statusBar().showMessage("At ({:4d}, {:4d})".format(a0.x(),
    #                                                             a0.y()))
    #     if a0.button() == Qt.LeftButton:
    #         rect = self.label._rect_position()
    #         self.roi_viewer.setPixmap(self.im.copy(rect))
    #         self.roi_viewer.update()

    def btnPressEvent(self):
        fn, t = QFileDialog().getOpenFileName(self,
                                              "Open a image file.",
                                              filter="Images (*.png)")
        if self.im.load(fn):
            # h, w = self.im.height(), self.im.width()
            # scale = 500 // min(h, w)
            self.label.setPixmap(self.im.scaled(500, 500, #QSize(h * scale, w * scale),
                                                Qt.KeepAspectRatio))


if __name__ == '__main__':
    app = QApplication(sys.argv)

    w = MainWindow()
    w.show()

    sys.exit(app.exec_())
