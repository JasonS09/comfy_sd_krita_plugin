from PyQt5.QtWidgets import QHBoxLayout, QVBoxLayout, QWidget, QPushButton

from ..script import script
from ..widgets import (
    QComboBoxLayout,
    QLineEditLayout,
    QPromptLayout,
    QSpinBoxLayout,
    StatusBar,
    QCheckBox,
    TipsLayout
)

from ..prompt import PromptBase
from ..utils import get_workflow, get_mode

class GeneratePage(QWidget):
    name = "Generate"
    def __init__(self, *args, **kwargs):
        super(GeneratePage, self).__init__(*args, **kwargs)

        self.status_bar = StatusBar()

        self.custom_workflow = QCheckBox(
            script.cfg, "custom_workflow", label="Enable custom workflow"
        )

        self.inpaint_tip = TipsLayout(
            ["Inpaint will use the current layer as mask.",
            '<a href="https://github.com/JasonS09/comfy_sd_krita_plugin/wiki/Inpainting" target="_blank">Check inpainting guide</a>']
        )

        self.inpaint = QCheckBox(
            script.cfg, "inpaint", label="Inpaint"
        )

        self.prompt_layout = QPromptLayout(
            script.cfg, "prompt", "negative_prompt"
        )

        self.prompt_layer_load = QPushButton("Load Prompt from Layer")

        self.seed_layout = QLineEditLayout(
            script.cfg, "seed", label="Seed:", placeholder="Random"
        )

        self.sampler_layout = QComboBoxLayout(
            script.cfg,
            "sampler_list",
            "sampler",
            "Sampler:"
        )

        self.scheduler_layout = QComboBoxLayout(
            script.cfg,
            "scheduler_list",
            "scheduler",
            "Scheduler:"
        )

        self.steps_layout = QSpinBoxLayout(
            script.cfg, "steps", label="Steps:", min=1, max=9999, step=1
        )
        self.cfg_scale_layout = QSpinBoxLayout(
            script.cfg,
            "cfg_scale",
            label="CFG scale:",
            min=1.0,
            max=9999.0
        )

        self.invert_mask = QCheckBox(script.cfg, "inpaint_invert_mask", "Invert mask")
        self.auto_generate_mask = QCheckBox(script.cfg, "inpaint_auto_generate_mask", "Auto generate transparency mask")
        self.fill_layout = QComboBoxLayout(
            script.cfg, "inpaint_fill_list", "inpaint_fill", label="Inpaint fill:"
        )

        self.denoise_tip = TipsLayout(
            ["Set 1 denoise to use txt2img. Otherwise img2img workflow will be run. This does not apply when inpaint is selected."]
        )

        self.denoising_strength_layout = QSpinBoxLayout(
            script.cfg,
            "denoising_strength",
            label="Denoising strength:",
            step=0.01,
        )

        self.get_workflow_btn = QPushButton("Get workflow")
        self.btn = QPushButton("Start Generating")

        inline_layout = QHBoxLayout()
        inline_layout.addLayout(self.steps_layout)
        inline_layout.addLayout(self.cfg_scale_layout)

        inline_2_layout = QHBoxLayout()
        inline_2_layout.addWidget(self.custom_workflow)
        inline_2_layout.addWidget(self.inpaint)

        inline_3_layout = QHBoxLayout()
        inline_3_layout.addWidget(self.invert_mask)
        inline_3_layout.addWidget(self.auto_generate_mask)

        self.layout = layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)

        layout.addWidget(self.status_bar)
        layout.addLayout(self.inpaint_tip)
        layout.addLayout(inline_2_layout)
        layout.addLayout(self.prompt_layout)
        layout.addWidget(self.prompt_layer_load)
        layout.addLayout(self.seed_layout)
        layout.addLayout(self.sampler_layout)
        layout.addLayout(self.scheduler_layout)
        layout.addLayout(inline_layout)
        layout.addLayout(inline_3_layout)
        layout.addLayout(self.fill_layout)
        layout.addLayout(self.denoise_tip)
        layout.addLayout(self.denoising_strength_layout)
        layout.addWidget(self.get_workflow_btn)
        layout.addWidget(self.btn)

        self.setLayout(layout)

    def update_prompt(self, info: PromptBase):
        if info is None:
            return
        self.prompt_layout.qedit_prompt.setPlainText(info.pos_prompt)
        self.prompt_layout.qedit_neg_prompt.setPlainText(info.neg_prompt)
        self.sampler_layout.qcombo.setEditText(info.sampler)
        self.scheduler_layout.qcombo.setEditText(info.scheduler)
        self.seed_layout.qedit.setText(str(info.seed))
        self.cfg_scale_layout.qspin.setValue(info.cfg)
        self.steps_layout.qspin.setValue(info.steps)

    def toggle_inpaint(self, visible):
        self.inpaint_tip.setVisible(visible)
        self.fill_layout.qcombo.setVisible(visible)
        self.fill_layout.qlabel.setVisible(visible)
        self.invert_mask.setVisible(visible)
        self.auto_generate_mask.setVisible(visible)

    def generate(self):
        if script.cfg("inpaint", bool):
            script.action_inpaint()
        elif script.cfg("denoising_strength", float) < 1:
            script.action_img2img()
        else:
            script.action_txt2img()

    def cfg_init(self):
        self.custom_workflow.cfg_init()
        self.inpaint.cfg_init()
        self.prompt_layout.cfg_init()
        self.seed_layout.cfg_init()
        self.sampler_layout.cfg_init()
        self.scheduler_layout.cfg_init()
        self.steps_layout.cfg_init()
        self.cfg_scale_layout.cfg_init()
        self.denoising_strength_layout.cfg_init()
        self.fill_layout.cfg_init()
        self.invert_mask.cfg_init()
        self.auto_generate_mask.cfg_init()

        self.toggle_inpaint(self.inpaint.isChecked())

    def cfg_connect(self):
        self.custom_workflow.cfg_connect()
        self.inpaint.cfg_connect()
        self.prompt_layout.cfg_connect()
        self.prompt_layer_load.released.connect(lambda: self.update_prompt(script.prompt_from_selection()))
        self.seed_layout.cfg_connect()
        self.sampler_layout.cfg_connect()
        self.scheduler_layout.cfg_connect()
        self.steps_layout.cfg_connect()
        self.cfg_scale_layout.cfg_connect()
        self.denoising_strength_layout.cfg_connect()
        self.fill_layout.cfg_connect()
        self.invert_mask.cfg_connect()
        self.auto_generate_mask.cfg_connect()

        script.status_changed.connect(lambda s: self.status_bar.set_status(s))
        self.inpaint.toggled.connect(lambda i: self.toggle_inpaint(i))
        self.btn.released.connect(self.generate)
        self.get_workflow_btn.released.connect(lambda: get_workflow(script.cfg, script.action_get_workflow, get_mode(script.cfg)))
