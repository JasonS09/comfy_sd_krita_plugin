from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QApplication, QFileDialog, QPushButton, QVBoxLayout, QHBoxLayout
from ..widgets import QLabel

class ImageLoaderLayout(QVBoxLayout):
    def __init__(self, *args, **kwargs):
        super(ImageLoaderLayout, self).__init__(*args, **kwargs)

        self.scaled_preview = QLabel()
        self.scaled_preview.setAlignment(Qt.AlignCenter)
        self.scaled_preview.setMinimumWidth(256)
        self.import_button = QPushButton('Import image')
        self.paste_button = QPushButton('Paste image')
        self.clear_button = QPushButton('Clear')
        self.original_pixmap = QPixmap()

        button_layout = QHBoxLayout()
        button_layout.addWidget(self.import_button)
        button_layout.addWidget(self.paste_button)

        self.addLayout(button_layout)
        self.addWidget(self.clear_button)
        self.addWidget(self.scaled_preview)

        self.import_button.released.connect(self.load_image)
        self.paste_button.released.connect(self.paste_image)
        self.clear_button.released.connect(self.clear_image)

    def get_pixmap(self):
        return self.original_pixmap

    def set_pixmap(self, pixmap):
        # store the full size pixmap
        self.original_pixmap = pixmap
        max_width = self.scaled_preview.width()
        if pixmap.width() > max_width:
            pixmap = pixmap.scaledToWidth(max_width, Qt.SmoothTransformation)
        self.scaled_preview.setPixmap(pixmap)

    def load_image(self):
        file_name, _ = QFileDialog.getOpenFileName(self.import_button, 'Open File', '', 'Image Files (*.png *.jpg *.bmp)')
        if file_name:
            pixmap = QPixmap(file_name)
            self.set_pixmap(pixmap)

    def paste_image(self):
        pixmap = QPixmap(QApplication.clipboard().pixmap())
        self.set_pixmap(pixmap)

    def clear_image(self):
        self.set_pixmap(QPixmap())


        