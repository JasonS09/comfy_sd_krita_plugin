import json
from krita import (
    QWidget,
    QPushButton,
    QVBoxLayout,
    QFileDialog
)
from ..widgets import(
    StatusBar,
    QComboBoxLayout,
    QPromptEdit
)
from ..script import script

class WorkflowPage(QWidget):
    name = "Workflow"

    def __init__(self, *args, **kwargs):
        super(WorkflowPage, self).__init__(*args, **kwargs)
        self.status_bar = StatusBar()
        self.import_workflow = QPushButton("Import Workflow")
        self.workflow_to = QComboBoxLayout(
            script.cfg, "workflow_to_list", "workflow_to", "Workflow to:"
        )
        self.workflow = QPromptEdit("", 20)
        self.run_this_workflow = QPushButton("Run This Workflow")
        layout = QVBoxLayout()
        layout.addWidget(self.status_bar)
        layout.addWidget(self.import_workflow)
        layout.addLayout(self.workflow_to)
        layout.addWidget(self.workflow)
        layout.addWidget(self.run_this_workflow)
        self.setLayout(layout)

        self.prefix = script.cfg.get("workflow_to", str)

    def import_json(self, filename):
        # Open the json file and read its contents
        with open(filename, "r") as f:
            data = json.load(f)

        if "prompt" in data.keys():
            data = data["prompt"]
        # Convert the data to a string
        text = json.dumps((data), indent=4)
        # Set the text of the workflow editor
        self.workflow.setPlainText(text)
    
    def on_import_workflow_release(self):
        # Get the filename from a file dialog
        filename, _ = QFileDialog.getOpenFileName(self, "Import Workflow", "", "JSON files (*.json)")
        # If the user selected a file, import it
        if filename:
            self.import_json(filename)
    
    def update_prefix(self):
        self.prefix = script.cfg.get("workflow_to", str)
        self.workflow.setPlainText(script.cfg(f"{self.prefix}_workflow", str))

    def cfg_init(self):
        self.workflow_to.cfg_init()
        self.workflow_to.qcombo.setEditText(self.prefix)
        if self.workflow.toPlainText() != script.cfg(f"{self.prefix}_workflow", str):
            self.update_prefix()

    def cfg_connect(self):
        self.workflow_to.cfg_connect()
        self.import_workflow.released.connect(self.on_import_workflow_release)
        self.run_this_workflow.released.connect(
            lambda: script.action_run_workflow(script.cfg(f"{self.prefix}_workflow", str))
        )
        self.workflow.textChanged.connect(
            lambda: script.cfg.set(f"{self.prefix}_workflow", self.workflow.toPlainText())
        )
        self.workflow_to.qcombo.editTextChanged.connect(self.update_prefix)
        script.status_changed.connect(lambda s: self.status_bar.set_status(s))
