from ..defaults import (
    STATE_LOADING,
    STATE_DONE,
    STATE_INIT,
    STATE_INTERRUPT,
    STATE_READY,
    STATE_URLERROR,
)
from .misc import QLabel


class StatusBar(QLabel):
    def __init__(self, *args, **kwargs):
        super(StatusBar, self).__init__(*args, **kwargs)
        self.set_status(STATE_INIT)

    def set_status(self, s):
        if s == STATE_READY and STATE_URLERROR not in self.text():
            return
        if (s == STATE_LOADING or s == STATE_DONE) and STATE_INTERRUPT == self.text():
            return

        self.setText(f"<b>Status:</b> {s}")
