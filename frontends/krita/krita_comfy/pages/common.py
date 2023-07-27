from krita import QHBoxLayout, QPushButton, QVBoxLayout, QWidget

from ..script import script
from ..widgets import QCheckBox, QComboBoxLayout, QLabel, QSpinBoxLayout

# Notes:
# - move tiling mode to config?
# - move upscaler/face restorer to config?


class SDCommonPage(QWidget):
    name = "SD Common Options"

    def __init__(self, *args, **kwargs):
        super(SDCommonPage, self).__init__(*args, **kwargs)

        self.title = QLabel("<em>Quick Config</em>")

        # Model list
        self.sd_model_layout = QComboBoxLayout(
            script.cfg, "sd_model_list", "sd_model", label="SD model:"
        )

        # VAE list
        self.sd_vae_layout = QComboBoxLayout( 
            script.cfg, "sd_vae_list", "sd_vae", label="VAE:"
        )

        # Clip skip
        self.clip_skip_layout = QSpinBoxLayout(
            script.cfg, "clip_skip", label="Clip skip:", min=1, max=12, step=1
        )

        # batch size & count
        self.batch_size_layout = QSpinBoxLayout(
            script.cfg, "sd_batch_size", label="Batch size:", min=1, max=9999, step=1
        )
        batch_layout = QHBoxLayout()
        batch_layout.addLayout(self.batch_size_layout)

        # base/max size adjustment
        self.base_size_layout = QSpinBoxLayout(
            script.cfg, "sd_base_size", label="Base size:", min=64, max=8192, step=64
        )
        self.max_size_layout = QSpinBoxLayout(
            script.cfg, "sd_max_size", label="Max size:", min=64, max=8192, step=64
        )
        size_layout = QHBoxLayout()
        size_layout.addLayout(self.base_size_layout)
        size_layout.addLayout(self.max_size_layout)

        # global upscaler
        self.upscaler_layout = QComboBoxLayout(
            script.cfg, 
            ["None"] + script.cfg("upscaler_methods_list", "QStringList") + script.cfg("upscaler_model_list", "QStringList"), 
            "upscaler_name", 
            label="Upscaler:"
        )

        self.sddebz = QCheckBox(
            script.cfg, "disable_sddebz_highres", "Disable base/max size"
        )

        checkboxes_layout = QHBoxLayout()
        checkboxes_layout.addWidget(self.sddebz)

        # Interrupt button
        self.interrupt_btn = QPushButton("Interrupt")

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)

        layout.addWidget(self.title)
        layout.addLayout(self.sd_model_layout)
        layout.addLayout(self.sd_vae_layout)
        layout.addLayout(self.clip_skip_layout)
        layout.addLayout(batch_layout)
        layout.addLayout(checkboxes_layout)
        layout.addLayout(size_layout)
        layout.addLayout(self.upscaler_layout)
        layout.addWidget(self.interrupt_btn)
        layout.addStretch()

        self.setLayout(layout)

    def cfg_init(self):
        self.sd_model_layout.cfg_init()
        self.sd_vae_layout.cfg_init()
        self.clip_skip_layout.cfg_init()
        self.batch_size_layout.cfg_init()
        self.base_size_layout.cfg_init()
        self.max_size_layout.cfg_init()
        self.upscaler_layout.cfg_init()
        self.sddebz.cfg_init()

        self.title.setVisible(not script.cfg("minimize_ui", bool))

    def cfg_connect(self):
        self.sd_model_layout.cfg_connect()
        self.sd_vae_layout.cfg_connect()
        self.clip_skip_layout.cfg_connect()
        self.batch_size_layout.cfg_connect()
        self.base_size_layout.cfg_connect()
        self.max_size_layout.cfg_connect()
        self.upscaler_layout.cfg_connect()
        self.sddebz.cfg_connect()

        # hide base/max size when disabled
        def toggle_sddebz_highres(visible):
            self.base_size_layout.qspin.setVisible(visible)
            self.base_size_layout.qlabel.setVisible(visible)
            self.max_size_layout.qspin.setVisible(visible)
            self.max_size_layout.qlabel.setVisible(visible)
            self.upscaler_layout.qcombo.setVisible(visible)
            self.upscaler_layout.qlabel.setVisible(visible)

        self.sddebz.toggled.connect(lambda b: toggle_sddebz_highres(not b))
        toggle_sddebz_highres(not script.cfg("disable_sddebz_highres", bool))

        self.interrupt_btn.released.connect(lambda: script.action_interrupt())
