from PyQt5.QtWidgets import QHBoxLayout, QPushButton

from ..script import script
from ..widgets import QCheckBox, QComboBoxLayout, TipsLayout
from .img_base import SDImgPageBase
from ..utils import get_workflow


class InpaintPage(SDImgPageBase):
    name = "Inpaint"

    def __init__(self, *args, **kwargs):
        super(InpaintPage, self).__init__(cfg_prefix="inpaint", *args, **kwargs)
        self.layout.addLayout(self.denoising_strength_layout)

        self.invert_mask = QCheckBox(script.cfg, "inpaint_invert_mask", "Invert mask")
        self.auto_generate_mask = QCheckBox(script.cfg, "inpaint_auto_generate_mask", "Auto generate transparency mask")

        inline1 = QHBoxLayout()
        inline1.addWidget(self.invert_mask)
        inline1.addWidget(self.auto_generate_mask)

        self.fill_layout = QComboBoxLayout(
            script.cfg, "inpaint_fill_list", "inpaint_fill", label="Inpaint fill:"
        )

        inline2 = QHBoxLayout()

        self.tips = TipsLayout(
            [
                "Ensure the inpaint layer is selected.",
                "Select what the model will see when inpainting. <em>Inpaint full res</em> is unnecessary.",
            ]
        )
        self.tips2 = TipsLayout(
            [
                '<a href="https://github.com/Interpause/auto-sd-paint-ext/wiki/Usage-Guide#inpainting" target="_blank">Inpaint Full Res & Mask Blur is obsolete; Click for new method.</a>'
            ],
            prefix="",
        )
        self.btn = QPushButton("Start inpaint")
        self.get_workflow_btn = QPushButton("Get workflow")

        self.layout.addLayout(self.fill_layout)
        self.layout.addLayout(inline1)
        self.layout.addLayout(inline2)
        self.layout.addWidget(self.btn)
        self.layout.addWidget(self.get_workflow_btn)
        self.layout.addLayout(self.tips2)
        self.layout.addLayout(self.tips)
        self.layout.addStretch()

    def cfg_init(self):
        super(InpaintPage, self).cfg_init()
        self.fill_layout.cfg_init()
        self.invert_mask.cfg_init()
        self.auto_generate_mask.cfg_init()

        self.tips.setVisible(not script.cfg("minimize_ui", bool))

    def cfg_connect(self):
        super(InpaintPage, self).cfg_connect()
        self.fill_layout.cfg_connect()
        self.invert_mask.cfg_connect()
        self.auto_generate_mask.cfg_connect()

        self.btn.released.connect(lambda: script.action_inpaint())
        self.get_workflow_btn.released.connect(
            lambda: get_workflow(script.cfg, script.action_get_workflow, "inpaint")
        )