from functools import partial

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtWidgets import QApplication, QPushButton, QWidget, QVBoxLayout, QHBoxLayout, QStackedLayout

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
        self.image_loader.set_pixmap(
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

        self.pixel_perfect = QCheckBox(
            script.cfg, f"controlnet{self.unit}_pixel_perfect", label="Pixel perfect:"
        )
        self.weight_layout = QSpinBoxLayout(
            script.cfg, f"controlnet{self.unit}_weight", label="Weight:", min=0, max=10, step=0.05
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
        self.annotator_preview.setMinimumWidth(256)
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
        layout.addWidget(self.pixel_perfect)
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

    def set_input(self, inputs, key, value):
        inputs.update({key: value})
        script.cfg.set(f"controlnet{self.unit}_inputs", inputs)
    
    def add_spinbox(self, key, value = None, **kwargs):
        label = key.capitalize().replace("_", " ")+":"
        inputs = dict(script.cfg(f"controlnet{self.unit}_inputs", dict))
        widget = QSpinBoxLayout(script.cfg, "", label=label, **kwargs)
        widget.setObjectName(key)
        if value is not None:
            widget.qspin.setValue(value)
        widget.qspin.valueChanged.connect(
            lambda value: self.set_input(dict(script.cfg(f"controlnet{self.unit}_inputs", dict)), key, value)
        )
        self.set_input(inputs, key, widget.qspin.value())
        self.preprocessor_settings_layout.addLayout(widget)
        if key == "resolution":
            self.pixel_perfect_handle_resolution(self.pixel_perfect.isChecked())
    
    def add_combobox(self, key, value = None, options = None, **kwargs):
        label = key.capitalize().replace("_", " ")+":"
        inputs = dict(script.cfg(f"controlnet{self.unit}_inputs", dict))
        widget = QComboBoxLayout(script.cfg, [], "", label=label, **kwargs)
        if value is not None:
            widget.qcombo.setEditText(value)
        widget.qcombo.addItems(options)
        widget.qcombo.editTextChanged.connect(
            lambda value: self.set_input(dict(script.cfg(f"controlnet{self.unit}_inputs", dict)), key, value)
        )
        self.set_input(inputs, key, widget.qcombo.currentText())
        self.preprocessor_settings_layout.addLayout(widget)

    def hide_weight_and_guidance(self):
        self.weight_layout.qspin.hide()
        self.weight_layout.qlabel.hide()
        self.guidance_start_layout.qspin.hide()
        self.guidance_start_layout.qlabel.hide()
        self.guidance_end_layout.qspin.hide()
        self.guidance_end_layout.qlabel.hide()

    def show_weight_and_guidance(self):
        self.weight_layout.qspin.show()
        self.weight_layout.qlabel.show()
        self.guidance_start_layout.qspin.show()
        self.guidance_start_layout.qlabel.show()
        self.guidance_end_layout.qspin.show()
        self.guidance_end_layout.qlabel.show()   

    def show_resolution(self):
        res_layout = self.preprocessor_settings_layout.findChild(QSpinBoxLayout, "resolution")
        if res_layout is not None:
            res_layout.qspin.show()
            res_layout.qlabel.show()

    def hide_resolution(self):
        res_layout = self.preprocessor_settings_layout.findChild(QSpinBoxLayout, "resolution")
        if res_layout is not None:
            res_layout.qspin.hide()
            res_layout.qlabel.hide()

    def pixel_perfect_handle_resolution(self, checked):
        if checked:
            self.hide_resolution()
        else:
            self.show_resolution()

    def add_preprocessor_options(self):
        clear_layout(self.preprocessor_settings_layout)
        script.cfg.set(f"controlnet{self.unit}_inputs", dict())
        if script.cfg(f"controlnet{self.unit}_preprocessor", str) == "Revision":
            self.hide_weight_and_guidance()
        else:
            self.show_weight_and_guidance()
        for preprocessor, info in script.cfg("controlnet_preprocessors_info", dict).items():
            preprocessor_inputs = script.cfg(f"controlnet{self.unit}_inputs", dict)
            if preprocessor == script.cfg(f"controlnet{self.unit}_preprocessor", str):
                for key, value in info["inputs"].items():
                    if value[0] in ["IMAGE", "MASK", "LATENT", "MODEL", "CLIP_VISION_OUTPUT", "CONDITIONING"]:
                        continue
                    if value[0] == "INT" or value[0] == "FLOAT":
                        rest = value[1].copy()
                        rest.pop("default")
                        self.add_spinbox(
                            key, 
                            preprocessor_inputs[key] if key in preprocessor_inputs else value[1]["default"], 
                            **rest
                        )
                    else:
                        self.add_combobox(
                            key, 
                            preprocessor_inputs[key] if key in preprocessor_inputs else value[1]["default"],
                            value[0]
                        )

    def enable_changed(self, state):
        if state == 1 or state == 2:
            script.action_update_controlnet_config()

    def image_loaded(self):
        image = self.image_loader.get_pixmap().toImage().convertToFormat(QImage.Format_RGBA8888)
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
        self.pixel_perfect.cfg_init()
        self.guidance_start_layout.cfg_init()
        self.guidance_end_layout.cfg_init()

        if self.preprocessor_layout.qcombo.currentText() == "None":
            self.annotator_preview_button.setEnabled(False)
        else:
            self.annotator_preview_button.setEnabled(True)

        self.pixel_perfect_handle_resolution(self.pixel_perfect.isChecked())

    def cfg_connect(self):
        self.enable.cfg_connect()
        self.preprocessor_layout.cfg_connect()
        self.model_layout.cfg_connect()
        self.weight_layout.cfg_connect()
        self.pixel_perfect.cfg_connect()
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
        self.pixel_perfect.toggled.connect(lambda t: self.pixel_perfect_handle_resolution(t))
        self.annotator_preview_button.released.connect(
            lambda: script.action_preview_controlnet_annotator()
        )
        self.copy_result_button.released.connect(self.copy_result_released)
        self.annotator_clear_button.released.connect(lambda: self.annotator_preview.setPixmap(QPixmap()))
        script.controlnet_preview_annotator_received.connect(self.annotator_preview_received)

