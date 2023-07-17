from krita import QHBoxLayout, QPushButton

from ..script import script
from ..widgets import TipsLayout
from .img_base import SDImgPageBase


class Txt2ImgPage(SDImgPageBase):
    name = "Txt2Img"

    def __init__(self, *args, **kwargs):
        super(Txt2ImgPage, self).__init__(cfg_prefix="txt2img", *args, **kwargs)

        self.denoising_strength_layout.qlabel.setText("Denoising Strength (highres fix):")
        inline_layout = QHBoxLayout()
        inline_layout.addLayout(self.denoising_strength_layout)

        self.tips = TipsLayout(
            ["Set base_size & max_size for highres fix to work."]
        )

        self.btn = QPushButton("Start txt2img")

        self.layout.addLayout(inline_layout)
        self.layout.addWidget(self.btn)
        self.layout.addLayout(self.tips)
        self.layout.addStretch()

    def cfg_init(self):
        super(Txt2ImgPage, self).cfg_init()
        self.tips.setVisible(not script.cfg("minimize_ui", bool))

    def cfg_connect(self):
        super(Txt2ImgPage, self).cfg_connect()

        def toggle_highres(enabled):
            # hide/show denoising strength
            self.denoising_strength_layout.qlabel.setVisible(enabled)
            self.denoising_strength_layout.qspin.setVisible(enabled)

        toggle_highres(not script.cfg("disable_sddebz_highres", bool))

        self.btn.released.connect(lambda: script.action_txt2img())
