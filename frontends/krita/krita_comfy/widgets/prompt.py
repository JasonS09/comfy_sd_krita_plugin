import re
from PyQt5.QtGui import QSyntaxHighlighter, QColor, QColorConstants
from PyQt5.QtWidgets import QPlainTextEdit, QSizePolicy, QVBoxLayout
from ..config import Config
from ..utils import auto_complete_LoRA, auto_complete_embedding, re_lora, re_embedding


class QPromptHighLighter(QSyntaxHighlighter):
    def __init__(self, cfg: Config, *args, **kwargs):
        super(QSyntaxHighlighter, self).__init__(*args, **kwargs)

        self.cfg = cfg
        self.highlighters = [
            [re_lora, auto_complete_LoRA],
            [re_embedding, auto_complete_embedding]]

    def highlightBlock(self, line):
        for expression, func in self.highlighters:
            for m in re.finditer(expression, line):
                valid, names = func(self.cfg, m.group(1))
                color: QColor = QColorConstants.Green
                color = color if valid else QColorConstants.Red
                color = color if len(names) < 2 else QColorConstants.Yellow
                self.setFormat(m.start(0), m.end(0), color)


class QPromptEdit(QPlainTextEdit):
    def __init__(self, placeholder="Enter prompt...", num_lines=5, *args, **kwargs):
        super(QPromptEdit, self).__init__(*args, **kwargs)
        self.setPlaceholderText(placeholder)
        self.setFixedHeight(self.fontMetrics().lineSpacing() * num_lines)
        self.setSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.Maximum)


class QPromptLayout(QVBoxLayout):
    prompt_label: str = "Prompt:"
    neg_prompt_label: str = "Negative prompt:"

    def __init__(
        self, cfg: Config, prompt_cfg: str, neg_prompt_cfg: str, *args, **kwargs
    ):
        """Layout for prompt and negative prompt.

        Args:
            cfg (Config): Config to connect to.
            prompt_cfg (str): Config key to read/write prompt to.
            neg_prompt_cfg (str): Config key to read/write negative prompt to.
        """
        super(QPromptLayout, self).__init__(*args, **kwargs)

        # Used to connect to config stored in script
        self.cfg = cfg
        self.prompt_cfg = prompt_cfg
        self.neg_prompt_cfg = neg_prompt_cfg

        self.qedit_prompt = QPromptEdit(placeholder=self.prompt_label)
        self.qedit_neg_prompt = QPromptEdit(placeholder=self.neg_prompt_label)

        self.addWidget(self.qedit_prompt)
        self.addWidget(self.qedit_neg_prompt)

    def cfg_init(self):
        # NOTE: update timer -> cfg_init, setText seems to reset cursor position so we prevent it
        prompt = self.cfg(self.prompt_cfg, str)
        neg_prompt = self.cfg(self.neg_prompt_cfg, str)
        if self.qedit_prompt.toPlainText() != prompt:
            self.qedit_prompt.setPlainText(prompt)
        if self.qedit_neg_prompt.toPlainText() != neg_prompt:
            self.qedit_neg_prompt.setPlainText(neg_prompt)

    def cfg_connect(self):
        self.qedit_prompt.textChanged.connect(
            lambda: self.cfg.set(self.prompt_cfg, self.qedit_prompt.toPlainText())
        )
        self.qedit_neg_prompt.textChanged.connect(
            lambda: self.cfg.set(
                self.neg_prompt_cfg, self.qedit_neg_prompt.toPlainText()
            )
        )
