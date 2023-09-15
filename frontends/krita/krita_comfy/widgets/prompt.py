from PyQt5.QtCore import Qt, QRect
from PyQt5.QtGui import QFocusEvent, QKeyEvent, QTextCursor
from PyQt5.QtWidgets import QPlainTextEdit, QSizePolicy, QVBoxLayout

from ..config import Config
from .prompt_complete import QPromptCompleter, QPromptHighLighter


class QPromptEdit(QPlainTextEdit):
    completer: QPromptCompleter = None

    def __init__(self, placeholder="Enter prompt...", num_lines=5, *args, **kwargs):
        super(QPromptEdit, self).__init__(*args, **kwargs)
        self.setPlaceholderText(placeholder)
        self.setFixedHeight(self.fontMetrics().lineSpacing() * num_lines)
        self.setSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.Maximum)

    def setCompleter(self, completer: QPromptCompleter):
        if self.completer:
            self.disconnect(self.completer)
        if not completer:
            return

        completer.setWidget(self)
        self.completer = completer
        self.completer.insertText.connect(self.insertCompletion)

    def insertCompletion(self, text):
        tc = self.textCursor()
        extra = (len(text) - len(self.completer.completionPrefix()))
        tc.movePosition(QTextCursor.Left)
        tc.movePosition(QTextCursor.EndOfWord)
        tc.insertText(text[-extra:])
        self.setTextCursor(tc)
        self.completer.popup().hide()

    def focusInEvent(self, e: QFocusEvent) -> None:
        if self.completer:
            self.completer.setWidget(self)
        QPlainTextEdit.focusInEvent(self, e)

    def keyPressEvent(self, e: QKeyEvent) -> None:
        if self.completer is None:
            QPlainTextEdit.keyPressEvent(self, e)
            return

        if e.key() == Qt.Key_Tab and self.completer.popup().isVisible():
            self.completer.insertText.emit(self.completer.getSelected())
            return

        QPlainTextEdit.keyPressEvent(self, e)
        tc: QTextCursor = self.textCursor()
        tc.select(QTextCursor.WordUnderCursor)
        cr: QRect = self.cursorRect()
        self.completer.try_auto_complete(tc, cr)


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

        # Create AutoCompleter
        self.completer = QPromptCompleter(cfg)
        self.completer_neg = QPromptCompleter(cfg)

        self.qedit_prompt = QPromptEdit(placeholder=self.prompt_label)
        self.qedit_neg_prompt = QPromptEdit(placeholder=self.neg_prompt_label)

        # Connect AutoCompleter
        self.qedit_prompt.setCompleter(self.completer)
        self.qedit_neg_prompt.setCompleter(self.completer_neg)

        # Connect Highlighter
        self.highlighter = QPromptHighLighter(cfg, self.qedit_prompt.document())
        self.neg_highlighter = QPromptHighLighter(cfg, self.qedit_neg_prompt.document())

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
