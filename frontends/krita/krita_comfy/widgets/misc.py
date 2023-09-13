from PyQt5.QtWidgets import QLabel as _QLabel
from PyQt5.QtCore import Qt


class QLabel(_QLabel):
    """QLabel with overwritten default behaviours."""

    def __init__(self, *args, **kwargs):
        super(QLabel, self).__init__(*args, **kwargs)

        self.setOpenExternalLinks(True)
        self.setWordWrap(True)
        self.setTextFormat(Qt.TextFormat.RichText)
