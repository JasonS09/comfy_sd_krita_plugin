from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QPushButton, QVBoxLayout, QWidget

from ..script import script
from ..widgets import QLabel, StatusBar
from ..defaults import (
    STATE_WAIT
)


class PreviewPage(QWidget):
    name = "Live Preview"

    def __init__(self, *args, **kwargs):
        super(PreviewPage, self).__init__(*args, **kwargs)

        self.status_bar = StatusBar()
        self.preview = QLabel()
        self.interrupt_btn = QPushButton("Interrupt")

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.status_bar)
        layout.addWidget(self.interrupt_btn)
        layout.addWidget(self.preview)
        layout.addStretch()
        self.setLayout(layout)

    def cfg_init(self):
        pass

    def _update_image(self, image_mime, image_bytes):
        try:
            image = QImage.fromData(image_bytes, image_mime)
            self.preview.setPixmap(QPixmap.fromImage(image))
        except:
            pass

    def _clear_image(self, status):
        try:
            if status == STATE_WAIT:
                self.preview.clear()
        except:
            pass

    def cfg_connect(self):
        script.status_changed.connect(lambda s: self.status_bar.set_status(s))
        script.status_changed.connect(self._clear_image)
        script.preview_received.connect(self._update_image)
        self.interrupt_btn.released.connect(lambda: script.action_interrupt())
