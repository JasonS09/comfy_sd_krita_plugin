from krita import QPushButton, QVBoxLayout, QWidget

from ..script import script
from ..widgets import QCheckBox, QComboBoxLayout, QLabel, StatusBar, QSpinBoxLayout

class UpscalePage(QWidget):
    name = "Upscale"

    def __init__(self, *args, **kwargs):
        super(UpscalePage, self).__init__(*args, **kwargs)

        self.status_bar = StatusBar()

        self.upscaler_layout = QComboBoxLayout(
            script.cfg, 
            ["None"] + script.cfg("upscaler_methods_list", "QStringList") + script.cfg("upscaler_model_list", "QStringList"), 
            "upscale_upscaler_name", 
            label="Upscaler:"
        )

        self.upscale_by = QSpinBoxLayout(
            script.cfg, "upscale_upscale_by", label="Scale by:", min=1, max=8
        )

        self.note = QLabel(
            """
NOTE:<br/>
 - txt2img & img2img will use the <em>Quick Config</em> Upscaler when needing to scale up.<br/>
            """
        )
        self.note.setWordWrap(True)

        self.btn = QPushButton("Start upscaling")

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)

        layout.addWidget(self.status_bar)
        layout.addWidget(self.note)
        layout.addLayout(self.upscaler_layout)
        layout.addLayout(self.upscale_by)
        layout.addWidget(self.btn)
        layout.addStretch()

        self.setLayout(layout)

    def cfg_init(self):
        self.upscaler_layout.cfg_init()
        self.upscale_by.cfg_init()
        self.note.setVisible(not script.cfg("minimize_ui", bool))

    def cfg_connect(self):
        self.upscaler_layout.cfg_connect()
        self.upscale_by.cfg_connect()
        self.btn.released.connect(lambda: script.action_simple_upscale())
        script.status_changed.connect(lambda s: self.status_bar.set_status(s))
