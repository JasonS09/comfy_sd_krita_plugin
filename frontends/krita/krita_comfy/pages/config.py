from functools import partial

from PyQt5.QtWidgets import QHBoxLayout, QLineEdit, QPushButton, QVBoxLayout, QWidget

from ..defaults import DEFAULTS
from ..script import script
from ..utils import reset_docker_layout
from ..widgets import QCheckBox, QLabel, QLineEditLayout, StatusBar


class ConfigPage(QWidget):
    name = "SD Plugin Config"

    def __init__(self, *args, **kwargs):
        super(ConfigPage, self).__init__(*args, **kwargs)

        self.status_bar = StatusBar()

        self.base_url = QLineEdit()
        self.base_url_reset = QPushButton("Default")
        inline1 = QHBoxLayout()
        inline1.addWidget(self.base_url)
        inline1.addWidget(self.base_url_reset)

        # Plugin settings
        self.create_mask_layer = QCheckBox(
            script.cfg, "create_mask_layer", "Add transparency mask"
        )
        self.save_temp_images = QCheckBox(
            script.cfg, "save_temp_images", "Save images for debug"
        )
        self.fix_aspect_ratio = QCheckBox(
            script.cfg, "fix_aspect_ratio", "Adjust selection aspect ratio"
        )
        self.minimize_ui = QCheckBox(script.cfg, "minimize_ui", "Squeeze the UI")
        self.alt_docker = QCheckBox(
            script.cfg, "alt_dock_behavior", "Alt Docker Behaviour"
        )
        self.hide_layers = QCheckBox(script.cfg, "hide_layers", "Auto hide layers")

        self.refresh_btn = QPushButton("Auto-Refresh Options Now")
        self.restore_defaults = QPushButton("Restore Defaults")

        self.info_label = QLabel()
        self.info_label.setOpenExternalLinks(True)
        self.info_label.setWordWrap(True)

        # scroll_area = QScrollArea()

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout_inner = QVBoxLayout()
        layout_inner.setContentsMargins(0, 0, 0, 0)

        layout_inner.addWidget(QLabel("<em>Plugin settings:</em>"))
        layout_inner.addWidget(self.minimize_ui)
        layout_inner.addWidget(self.alt_docker)
        layout_inner.addWidget(self.fix_aspect_ratio)
        layout_inner.addWidget(self.create_mask_layer)
        layout_inner.addWidget(self.hide_layers)
        layout_inner.addWidget(self.save_temp_images)

        layout_inner.addWidget(QLabel("<em>Backend/webUI settings:</em>"))

        # TODO: figure out how to set height of scroll area when there are too many options
        # or maybe an option search bar
        # scroll_area.setLayout(layout_inner)
        # scroll_area.setWidgetResizable(True)
        # layout.addWidget(scroll_area)
        layout.addWidget(self.status_bar)
        layout.addWidget(QLabel("<em>Backend url:</em>"))
        layout.addLayout(inline1)
        layout.addLayout(layout_inner)
        layout.addWidget(self.refresh_btn)
        layout.addWidget(self.restore_defaults)
        layout.addWidget(self.info_label)
        layout.addStretch()

        self.setLayout(layout)

    def cfg_init(self):
        # NOTE: update timer -> cfg_init, setText seems to reset cursor position so we prevent it
        base_url = script.cfg("base_url", str)
        if self.base_url.text() != base_url:
            self.base_url.setText(base_url)

        self.create_mask_layer.cfg_init()
        self.save_temp_images.cfg_init()
        self.minimize_ui.cfg_init()
        self.alt_docker.cfg_init()
        self.hide_layers.cfg_init()

        info_text = """
            <em>Tip:</em> Only a selected few backend/webUI settings are exposed above.<br/>
            <em>Tip:</em> You should look through & configure all the backend/webUI settings at least once.
            <br/><br/>
            <a href="{}" target="_blank">Configure all settings in webUI</a><br/>
            <a href="{}" target="_blank">Read the guide</a><br/>
            <a href="{}" target="_blank">Report bugs or suggest features</a>
            """.format(
            script.cfg("base_url", str),
            "https://github.com/JasonS09/comfy_sd_krita_plugin/wiki",
            "https://github.com/JasonS09/comfy_sd_krita_plugin/issues",
            )
        if script.cfg("minimize_ui", bool):
            info_text = "\n".join(info_text.split("\n")[-4:-1])
        self.info_label.setText(info_text)

    def cfg_connect(self):
        self.base_url.textChanged.connect(partial(script.cfg.set, "base_url"))
        # NOTE: this triggers on every keystroke; theres no focus lost signal...
        self.base_url.textChanged.connect(lambda: script.action_update_config())
        self.base_url_reset.released.connect(
            lambda: self.base_url.setText(DEFAULTS.base_url)
        )
        self.create_mask_layer.cfg_connect()
        self.save_temp_images.cfg_connect()
        self.fix_aspect_ratio.cfg_connect()
        self.minimize_ui.cfg_connect()
        self.alt_docker.cfg_connect()
        self.hide_layers.cfg_connect()

        def restore_defaults():
            script.restore_defaults()
            reset_docker_layout()
            script.cfg.set("first_setup", False)
            # retrieve list of available stuff again
            script.action_update_config()
            script.action_update_controlnet_config()

        self.refresh_btn.released.connect(lambda: script.action_update_config())
        self.restore_defaults.released.connect(restore_defaults)
        self.minimize_ui.toggled.connect(lambda _: script.config_updated.emit())
        self.alt_docker.toggled.connect(lambda _: script.config_updated.emit())
        script.status_changed.connect(lambda s: self.status_bar.set_status(s))
