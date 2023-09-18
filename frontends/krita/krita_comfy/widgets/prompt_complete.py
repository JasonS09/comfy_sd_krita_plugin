import re
from typing import List, Tuple, Optional
from dataclasses import dataclass, field

from PyQt5.QtCore import Qt, pyqtSignal, QStringListModel, QRect
from PyQt5.QtGui import QSyntaxHighlighter, QColor, QTextCursor
from PyQt5.QtWidgets import QCompleter

from ..config import Config
from ..utils import (
    fuzzy_match_LoRA,
    fuzzy_match_embedding,
    get_autocomplete_lora_list,
    re_embedding,
    re_lora,
    re_lora_start,
)


@dataclass
class SDCompleter:
    matcher: re.Pattern
    cfg: Config
    autocomplete_suffix: str
    autocomplete_list: List[str] = field(default_factory=str)

    def update(self):
        return


class LoraCompleter(SDCompleter):
    def __init__(self, cfg: Config):
        super().__init__(
            re.compile(re_lora_start + '$'),
            cfg,
            ":0.5>"
        )
        self.update()

    def update(self):
        self.autocomplete_list = get_autocomplete_lora_list(self.cfg)


class EmbeddingCompleter(SDCompleter):
    def __init__(self, cfg: Config):
        super().__init__(
            re.compile(re_embedding + '$'),
            cfg,
            " ",

        )
        self.update()

    def update(self):
        self.autocomplete_list = self.cfg("sd_embedding_list", str)


class QPromptHighLighter(QSyntaxHighlighter):
    def __init__(self, cfg: Config, *args, **kwargs):
        super(QSyntaxHighlighter, self).__init__(*args, **kwargs)

        self.cfg = cfg
        self.highlighters = [
            [re_lora, fuzzy_match_LoRA],
            [re_embedding, fuzzy_match_embedding]]

    def highlightBlock(self, line):
        for expression, func in self.highlighters:
            for m in re.finditer(expression, line):
                valid, names = func(self.cfg, m.group(1))
                color: QColor = QColor(Qt.green)
                color = color if valid else QColor(Qt.red)
                color = color if len(names) < 2 else QColor(Qt.yellow)
                self.setFormat(m.start(0), m.end(0)-m.start(0), color)


class QPromptCompleter(QCompleter):
    insertText = pyqtSignal(str)
    lastSelected: str = None
    model: QStringListModel = None

    def __init__(self, cfg: Config, parent=None):
        super().__init__(parent)
        self.model = QStringListModel()
        self.setModel(self.model)
        self.setCompletionMode(QCompleter.PopupCompletion)
        self.setCaseSensitivity(Qt.CaseInsensitive)
        self.highlighted.connect(self.setHighlighted)
        self.setMaxVisibleItems(5)
        self.cfg = cfg
        self.rules: List[SDCompleter] = [
            LoraCompleter(cfg),
            EmbeddingCompleter(cfg)]
        self.rule: Optional[SDCompleter] = None

    def hide_popup(self):
        self.rule = None
        self.popup().hide()

    def is_popup_visible(self):
        return self.popup().isVisible()

    def set_string_list(self, keys: List[str]):
        self.setCompletionMode(QCompleter.PopupCompletion)
        self.model.setStringList(keys)

    def setHighlighted(self, text):
        self.lastSelected = text

    def getSelected(self):
        extra = "" if self.rule is None else self.rule.autocomplete_suffix
        return self.lastSelected + extra

    def try_auto_complete(self, tc: QTextCursor, cr: QRect):
        # Get the line
        tc.select(QTextCursor.LineUnderCursor)
        line = tc.selectedText()

        match: Optional[re.Match] = None
        # match last occurrence of rule on the line
        for r in self.rules:
            match = re.search(r.matcher, line)
            if match is not None:
                self.rule = r
                break

        if match is not None:
            # Select first match group so we can replace it
            tc.movePosition(QTextCursor.StartOfLine, mode=QTextCursor.MoveAnchor)
            tc.movePosition(QTextCursor.Right, mode=QTextCursor.MoveAnchor, n=match.span(1)[0])
            tc.movePosition(QTextCursor.Right, mode=QTextCursor.KeepAnchor, n=match.span(1)[1])
            # update the completer for with the list for this rule
            self.rule.update()
            self.set_string_list(self.rule.autocomplete_list)
            self.setCompletionPrefix(tc.selectedText())
            popup = self.popup()
            popup.setCurrentIndex(self.completionModel().index(0, 0))
            cr.setWidth(popup.sizeHintForColumn(0)
                        + popup.verticalScrollBar().sizeHint().width())
            self.complete(cr)
        else:
            self.hide_popup()
