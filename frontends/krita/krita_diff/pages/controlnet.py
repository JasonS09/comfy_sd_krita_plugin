from krita import (
    QApplication, 
    QPixmap, 
    QImage, 
    QPushButton, 
    QWidget, 
    QVBoxLayout, 
    QHBoxLayout, 
    QStackedLayout, 
    Qt
)
from functools import partial
from ..script import script
from ..widgets import (
    QLabel, 
    StatusBar, 
    ImageLoaderLayout, 
    QCheckBox, 
    TipsLayout, 
    QComboBoxLayout, 
    QSpinBoxLayout
)
from ..utils import img_to_b64, b64_to_img, clear_layout

class ControlNetPage(QWidget):                                                      
    name = "ControlNet"

    def __init__(self, *args, **kwargs):
        super(ControlNetPage, self).__init__(*args, **kwargs)
        self.status_bar = StatusBar()
        self.controlnet_unit = QComboBoxLayout(
            script.cfg, "controlnet_unit_list", "controlnet_unit", label="Unit:"
        )
        self.controlnet_unit_layout_list = list(ControlNetUnitSettings(i) 
                                                for i in range(len(script.cfg("controlnet_unit_list"))))

        self.units_stacked_layout = QStackedLayout()
        
        for unit_layout in self.controlnet_unit_layout_list:
            self.units_stacked_layout.addWidget(unit_layout)

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.status_bar)
        layout.addLayout(self.controlnet_unit)
        layout.addLayout(self.units_stacked_layout)
        self.setLayout(layout)

    def controlnet_unit_changed(self, selected: str):
        self.units_stacked_layout.setCurrentIndex(int(selected))

    def cfg_init(self):
        self.controlnet_unit.cfg_init()
        
        for controlnet_unit_layout in self.controlnet_unit_layout_list:
            controlnet_unit_layout.cfg_init()
        
        self.controlnet_unit_changed(self.controlnet_unit.qcombo.currentText())

    def cfg_connect(self):
        self.controlnet_unit.cfg_connect()

        for controlnet_unit_layout in self.controlnet_unit_layout_list:
            controlnet_unit_layout.cfg_connect()

        self.controlnet_unit.qcombo.currentTextChanged.connect(self.controlnet_unit_changed)
        script.status_changed.connect(lambda s: self.status_bar.set_status(s))

class ControlNetUnitSettings(QWidget):    
    def __init__(self, cfg_unit_number: int = 0, *args, **kwargs):
        super(ControlNetUnitSettings, self).__init__(*args, **kwargs)           
        self.unit = cfg_unit_number
        self.preview_result = QPixmap() #This will help us to copy to clipboard the image with original dimensions.

        #Top checkbox
        self.enable = QCheckBox(
            script.cfg, f"controlnet{self.unit}_enable", f"Enable ControlNet {self.unit}"
        )

        self.image_loader = ImageLoaderLayout()
        input_image = script.cfg(f"controlnet{self.unit}_input_image", str)
        self.image_loader.preview.setPixmap(
            QPixmap.fromImage(b64_to_img(input_image) if input_image else QImage())
        )

        #Tips
        self.tips = TipsLayout(
            ["Selection will be used as input if no image has been uploaded or pasted."]
        )

        #Preprocessor list
        self.preprocessor_layout = QComboBoxLayout(
            script.cfg, "controlnet_preprocessor_list", f"controlnet{self.unit}_preprocessor", label="Preprocessor:"
        )

        #Model list
        self.model_layout = QComboBoxLayout(
            script.cfg, "controlnet_model_list", f"controlnet{self.unit}_model", label="Model:"
        )

        #Refresh button
        self.refresh_button = QPushButton("Refresh")

        self.weight_layout = QSpinBoxLayout(
            script.cfg, f"controlnet{self.unit}_weight", label="Weight:", min=0, max=2, step=0.05
        )
        self.guidance_start_layout = QSpinBoxLayout(
            script.cfg, f"controlnet{self.unit}_guidance_start", label="Guidance start:", min=0, max=1, step=0.01
        )
        self.guidance_end_layout = QSpinBoxLayout(
            script.cfg, f"controlnet{self.unit}_guidance_end", label="Guidance end:", min=0, max=1, step=0.01
        )

        # self.control_mode = QComboBoxLayout(
        #     script.cfg, "controlnet_control_mode_list", f"controlnet{self.unit}_control_mode", label="Control mode:"
        # )

        #Preprocessor settings
        self.preprocessor_settings_layout = QVBoxLayout()

        #Preview annotator
        self.annotator_preview = QLabel()
        self.annotator_preview.setAlignment(Qt.AlignCenter)
        self.annotator_preview_button = QPushButton("Preview annotator")
        self.annotator_clear_button = QPushButton("Clear preview")
        self.copy_result_button = QPushButton("Copy result to clipboard")

        guidance_layout = QHBoxLayout()
        guidance_layout.addLayout(self.guidance_start_layout)
        guidance_layout.addLayout(self.guidance_end_layout)

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.enable)
        layout.addLayout(self.image_loader)
        layout.addLayout(self.tips)
        layout.addLayout(self.preprocessor_layout)
        layout.addLayout(self.model_layout)
        layout.addWidget(self.refresh_button)
        layout.addLayout(self.preprocessor_settings_layout)
        layout.addLayout(self.weight_layout)
        layout.addLayout(guidance_layout)
        #layout.addLayout(self.control_mode)
        layout.addWidget(self.annotator_preview)
        layout.addWidget(self.annotator_preview_button)
        layout.addWidget(self.copy_result_button)
        layout.addWidget(self.annotator_clear_button)
        layout.addStretch()

        self.setLayout(layout)
        self.add_preprocessor_options()

    def set_input(self, input, key, value):
        input.update({key: value})
        script.cfg.set(f"controlnet{self.unit}_inputs", input)
    
    def add_spinbox(self, key, default = None, **kwargs):
        label = key.capitalize().replace("_", " ")+":"
        widget = QSpinBoxLayout(script.cfg, "", label=label, **kwargs)
        if default is not None:
            widget.qspin.setValue(default)
        widget.qspin.valueChanged.connect(
            lambda value: self.set_input(dict(script.cfg(f"controlnet{self.unit}_inputs", dict)), key, value)
        )
        self.preprocessor_settings_layout.addLayout(widget)
    
    def add_combobox(self, key, default = None, options = None, **kwargs):
        label = key.capitalize().replace("_", " ")+":"
        widget = QComboBoxLayout(script.cfg, [], "", label=label, **kwargs)
        if default is not None:
            widget.qcombo.setEditText(default)
        widget.qcombo.addItems(options)
        widget.qcombo.editTextChanged.connect(
            lambda value: self.set_input(dict(script.cfg(f"controlnet{self.unit}_inputs", dict)), key, value)
        )
        self.preprocessor_settings_layout.addLayout(widget)    

    def add_preprocessor_options(self):
        clear_layout(self.preprocessor_settings_layout)
        for preprocessor, inputs in script.cfg("controlnet_preprocessors_info", dict).items():
            if preprocessor == script.cfg(f"controlnet{self.unit}_preprocessor", str):
                for key, value in inputs.items():
                    if value[0] in ["IMAGE", "MASK", "LATENT", "MODEL"]:
                        continue
                    if value[0] == "INT" or value[0] == "FLOAT":
                        rest = value[1].copy()
                        rest.pop("default")
                        self.add_spinbox(key, value[1]["default"], **rest)
                    else:
                        self.add_combobox(key, value[1]["default"], value[0])

    def enable_changed(self, state):
        if state == 1 or state == 2:
            script.action_update_controlnet_config()

    def image_loaded(self):
        image = self.image_loader.preview.pixmap().toImage().convertToFormat(QImage.Format_RGBA8888)
        script.cfg.set(f"controlnet{self.unit}_input_image", img_to_b64(image)) 

    def annotator_preview_received(self, pixmap):
        self.preview_result = pixmap
        if pixmap.width() > self.annotator_preview.width():
            pixmap = pixmap.scaledToWidth(self.annotator_preview.width(), Qt.SmoothTransformation)
        self.annotator_preview.setPixmap(pixmap)
    
    def annotator_clear_button_released(self):
        self.annotator_preview.setPixmap(QPixmap())
        self.preview_result = QPixmap()

    def copy_result_released(self):
        if self.preview_result:
            clipboard = QApplication.clipboard()
            clipboard.setImage(self.preview_result.toImage())
           
    def cfg_init(self):  
        self.enable.cfg_init()
        self.preprocessor_layout.cfg_init()
        self.model_layout.cfg_init()
        self.weight_layout.cfg_init()
        self.guidance_start_layout.cfg_init()
        self.guidance_end_layout.cfg_init()

        if self.preprocessor_layout.qcombo.currentText() == "None":
            self.annotator_preview_button.setEnabled(False)
        else:
            self.annotator_preview_button.setEnabled(True)

    def cfg_connect(self):
        self.enable.cfg_connect()
        self.preprocessor_layout.cfg_connect()
        self.model_layout.cfg_connect()
        self.weight_layout.cfg_connect()
        self.guidance_start_layout.cfg_connect()
        self.guidance_end_layout.cfg_connect()
        self.image_loader.import_button.released.connect(self.image_loaded)
        self.image_loader.paste_button.released.connect(self.image_loaded)
        self.image_loader.clear_button.released.connect(
            partial(script.cfg.set, f"controlnet{self.unit}_input_image", "")
        )
        self.preprocessor_layout.qcombo.currentTextChanged.connect(
            lambda: self.annotator_preview_button.setEnabled(False) if 
                self.preprocessor_layout.qcombo.currentText() == "None" else self.annotator_preview_button.setEnabled(True)
        )
        self.preprocessor_layout.qcombo.currentTextChanged.connect(self.add_preprocessor_options)
        self.refresh_button.released.connect(lambda: script.action_update_controlnet_config())
        self.annotator_preview_button.released.connect(
            lambda: script.action_preview_controlnet_annotator()
        )
        self.copy_result_button.released.connect(self.copy_result_released)
        self.annotator_clear_button.released.connect(lambda: self.annotator_preview.setPixmap(QPixmap()))
        script.controlnet_preview_annotator_received.connect(self.annotator_preview_received)