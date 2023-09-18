import sys

from PyQt5.QtWidgets import QMainWindow, QApplication, QVBoxLayout, QWidget

from krita_comfy.config import Config
from krita_comfy.pages.txt2img import Txt2ImgPage

"""Test App for GUI components
    This file is not part of the Krita plugin.
    this is a way to invoke the plugin pages
    outside of Krita to test and debug them directly.
 """

class Test_Window(QMainWindow):
    def __init__(self):
        super().__init__()
        self.cfg = Config()
        self.setCentralWidget(QWidget())
        layout = QVBoxLayout(self.centralWidget())
        layout.setContentsMargins(0, 0, 0, 0)
        self.page = Txt2ImgPage()
        layout.addWidget(self.page)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = Test_Window()
    window.show()
    sys.exit(app.exec_())
