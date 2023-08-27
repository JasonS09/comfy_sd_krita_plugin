from krita import QHBoxLayout, QVBoxLayout, QWidget, QPushButton

from ..script import script
from ..widgets import (
    QComboBoxLayout,
    QLineEditLayout,
    QPromptLayout,
    QSpinBoxLayout,
    StatusBar,
    QCheckBox
)

from ..prompt import PromptBase

class SDImgPageBase(QWidget):
    def __init__(self, cfg_prefix: str, *args, **kwargs):
        super(SDImgPageBase, self).__init__(*args, **kwargs)

        self.status_bar = StatusBar()

        self.custom_workflow = QCheckBox(
            script.cfg, f"{cfg_prefix}_custom_workflow", label="Enable custom workflow"
        )

        self.prompt_layout = QPromptLayout(
            script.cfg, f"{cfg_prefix}_prompt", f"{cfg_prefix}_negative_prompt"
        )

        self.prompt_layer_load = QPushButton("Load Prompt from Layer")

        self.seed_layout = QLineEditLayout(
            script.cfg, f"{cfg_prefix}_seed", label="Seed:", placeholder="Random"
        )

        self.sampler_layout = QComboBoxLayout(
            script.cfg,
            f"{cfg_prefix}_sampler_list",
            f"{cfg_prefix}_sampler",
            label="Sampler:",
        )

        self.scheduler_layout = QComboBoxLayout(
            script.cfg,
            f"{cfg_prefix}_scheduler_list",
            f"{cfg_prefix}_scheduler",
            label="Scheduler:",
        )

        self.steps_layout = QSpinBoxLayout(
            script.cfg, f"{cfg_prefix}_steps", label="Steps:", min=1, max=9999, step=1
        )
        self.cfg_scale_layout = QSpinBoxLayout(
            script.cfg,
            f"{cfg_prefix}_cfg_scale",
            label="CFG scale:",
            min=1.0,
            max=9999.0,
        )

        inline_layout = QHBoxLayout()
        inline_layout.addLayout(self.steps_layout)
        inline_layout.addLayout(self.cfg_scale_layout)

        self.layout = layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)

        layout.addWidget(self.status_bar)
        layout.addWidget(self.custom_workflow)
        layout.addLayout(self.prompt_layout)
        layout.addWidget(self.prompt_layer_load)
        layout.addLayout(self.seed_layout)
        layout.addLayout(self.sampler_layout)
        layout.addLayout(self.scheduler_layout)
        layout.addLayout(inline_layout)

        self.setLayout(layout)

        # not added so inheritants can place it wherever they want
        self.denoising_strength_layout = QSpinBoxLayout(
            script.cfg,
            f"{cfg_prefix}_denoising_strength",
            label="Denoising strength:",
            step=0.01,
        )

    def update_prompt(self, info: PromptBase):
        self.prompt_layout.qedit_prompt.setPlainText(info.pos_prompt)
        self.prompt_layout.qedit_neg_prompt.setPlainText(info.neg_prompt)
        self.sampler_layout.qcombo.setEditText(info.sampler)
        self.scheduler_layout.qcombo.setEditText(info.scheduler)
        self.seed_layout.qedit.setText(str(info.seed))
        self.cfg_scale_layout.qspin.setValue(info.cfg)
        self.steps_layout.qspin.setValue(info.steps)

    def cfg_init(self):
        self.custom_workflow.cfg_init()
        self.prompt_layout.cfg_init()
        self.seed_layout.cfg_init()
        self.sampler_layout.cfg_init()
        self.scheduler_layout.cfg_init()
        self.steps_layout.cfg_init()
        self.cfg_scale_layout.cfg_init()
        self.denoising_strength_layout.cfg_init()

    def cfg_connect(self):
        self.custom_workflow.cfg_connect()
        self.prompt_layout.cfg_connect()
        self.prompt_layer_load.released.connect(lambda: self.update_prompt(script.prompt_from_selection()))
        self.seed_layout.cfg_connect()
        self.sampler_layout.cfg_connect()
        self.scheduler_layout.cfg_connect()
        self.steps_layout.cfg_connect()
        self.cfg_scale_layout.cfg_connect()
        self.denoising_strength_layout.cfg_connect()

        script.status_changed.connect(lambda s: self.status_bar.set_status(s))
