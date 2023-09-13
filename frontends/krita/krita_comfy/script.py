import itertools
import os
import time
import json

from PyQt5.QtCore import QObject, QRect, QTimer, Qt, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap

from krita import (
    Document,
    Krita,
    Node,
    Selection,
)

from .prompt import PromptResponse, PromptBase, Base64Image
from .client import Client
from .config import Config
from .defaults import (
    ADD_MASK_TIMEOUT,
    ERR_NO_DOCUMENT,
    ERR_MISSING_PROMPT,
    ERR_BACKEND,
    ERR_EMPTY_RESPONSE,
    ETA_REFRESH_INTERVAL,
    EXT_CFG_NAME,
    STATE_INTERRUPT,
    STATE_RESET_DEFAULT,
    STATE_WAIT,
    STATE_DONE
)
from .utils import (
    b64_to_img,
    find_optimal_selection_region,
    img_to_ba,
    save_img,
)


# Does it actually have to be a QObject?
# The only possible use I see is for event emitting
class Script(QObject):
    cfg: Config
    """config singleton"""
    client: Client
    """API client singleton"""
    status: str
    """Current status (shown in status bar)"""
    app: Krita
    """Krita's Application instance (KDE Application)"""
    doc: Document
    """Currently opened document if any"""
    node: Node
    """Currently selected layer in Krita"""
    selection: Selection
    """Selection region in Krita"""
    x: int
    """Left position of selection"""
    y: int
    """Top position of selection"""
    width: int
    """Width of selection"""
    height: int
    """Height of selection"""
    status_changed = pyqtSignal(str)
    config_updated = pyqtSignal()
    progress_update = pyqtSignal(object)
    controlnet_preview_annotator_received = pyqtSignal(QPixmap)

    def __init__(self):
        super(Script, self).__init__()
        # Persistent settings (should reload between Krita sessions)
        self.cfg = Config()
        # used for webUI scripts aka extensions not to be confused with their extensions
        self.ext_cfg = Config(name=EXT_CFG_NAME, model=None)
        self.client = Client(self.cfg, self.ext_cfg)
        self.client.status.connect(self.status_changed.emit)
        self.client.status.connect(self.stop_update_timer)
        self.client.config_updated.connect(self.config_updated.emit)
        self.eta_timer = QTimer()
        self.eta_timer.setInterval(ETA_REFRESH_INTERVAL)
        self.eta_timer.timeout.connect(lambda: self.action_update_eta())
        self.client.prompt_sent.connect(lambda: self.eta_timer.start())
        self.progress_update.connect(lambda p: self.update_status_bar_eta(p))
        # keep track of inserted layers to prevent accidental usage as inpaint mask
        self._inserted_layers = []

    def restore_defaults(self, if_empty=False):
        """Restore to default config."""
        self.cfg.restore_defaults(not if_empty)
        self.ext_cfg.config.remove("")

        if not if_empty:
            self.status_changed.emit(STATE_RESET_DEFAULT)

    def stop_update_timer(self, status):
        if status == STATE_DONE or self.client.interrupted \
            or not self.client.is_connected:
            if self.client.interrupted or not self.client.is_connected:
                    try:
                        self.client.status.disconnect(self.client.conn)
                    except TypeError:
                        pass # signal was not connected
            self.eta_timer.stop()

    def update_status_bar_eta(self, progress):
        if self.client.interrupted:
            self.client.interrupted = False
            return

        running = len(progress["queue_running"])
        if running > 0:
            self.status_changed.emit(f"Executing prompt... ({len(progress['queue_pending'])} in queue)")

    def prompt_from_selection(self) -> PromptBase:
        """Return prompt parameters from the selected layer or None"""
        self.update_selection()
        if not hasattr(self, 'node') or not self.node:
            return None

        prompt_info = PromptBase.from_json(self.node.name())
        if not prompt_info:
            self.status_changed.emit(ERR_MISSING_PROMPT)
        return prompt_info


    def update_selection(self):
        """Update references to key Krita objects as well as selection information."""
        self.app = Krita.instance()
        self.doc = self.app.activeDocument()

        # self.doc doesnt exist at app startup
        if not self.doc:
            self.status_changed.emit(ERR_NO_DOCUMENT)
            return

        self.node = self.doc.activeNode()
        self.selection = self.doc.selection()

        is_not_selected = (
            self.selection is None
            or self.selection.width() < 1
            or self.selection.height() < 1
        )
        if is_not_selected:
            self.x = 0
            self.y = 0
            self.width = self.doc.width()
            self.height = self.doc.height()
            self.selection = None  # for the two other cases of invalid selection
        else:
            self.x = self.selection.x()
            self.y = self.selection.y()
            self.width = self.selection.width()
            self.height = self.selection.height()

        assert (
            self.doc.colorDepth() == "U8"
        ), f'Only "8-bit integer/channel" supported, Document Color Depth: {self.doc.colorDepth()}'
        assert (
            self.doc.colorModel() == "RGBA"
        ), f'Only "RGB/Alpha" supported, Document Color Model: {self.doc.colorModel()}'

    def adjust_selection(self):
        """Adjust selection region to account for scaling and striding to prevent image stretch."""
        if self.selection is not None and self.cfg("fix_aspect_ratio", bool):
            x, y, width, height = find_optimal_selection_region(
                self.cfg("sd_base_size", int),
                self.cfg("sd_max_size", int),
                self.x,
                self.y,
                self.width,
                self.height,
                self.doc.width(),
                self.doc.height(),
            )

            self.x = x
            self.y = y
            self.width = width
            self.height = height

    def get_selection_image(self) -> QImage:
        """QImage of selection"""
        return QImage(
            self.doc.pixelData(self.x, self.y, self.width, self.height),
            self.width,
            self.height,
            QImage.Format_RGBA8888,
        ).rgbSwapped()

    def get_mask_image(self):
        """QImage of mask layer for inpainting"""
        if self.node.type() not in {"paintlayer", "filelayer"}:
            assert False, "Please select a valid layer to use as inpaint mask!"
        elif self.node in self._inserted_layers:
            assert False, "Selected layer was generated. Copy the layer if sure you want to use it as inpaint mask."

        mask = QImage(
            self.node.pixelData(self.x, self.y, self.width, self.height),
            self.width,
            self.height,
            QImage.Format_RGBA8888
        )

        transparency_mask = mask.convertToFormat(QImage.Format_Alpha8)
        transparency_mask.reinterpretAsFormat(QImage.Format_Grayscale8)
        transparency_mask = transparency_mask.convertToFormat(QImage.Format_RGBA8888)

        if not self.cfg("inpaint_invert_mask", bool):
            mask.invertPixels(QImage.InvertRgba) #Alpha channel is mask.
        else:
            transparency_mask.invertPixels()

        return mask.rgbSwapped(), transparency_mask.rgbSwapped()

    def img_inserter(self, x, y, width, height, inpaint=False, glayer=None, skip_check_pixel_data=False):
        """Return frozen image inserter to insert images as new layer."""
        # Selection may change before callback, so freeze selection region
        has_selection = self.selection is not None

        def create_layer(name: str):
            """Create new layer in document or group"""
            layer = self.doc.createNode(name, "paintLayer")
            parent = self.doc.rootNode()
            if glayer:
                glayer.addChildNode(layer, None)
            else:
                parent.addChildNode(layer, None)
            return layer
            
        def insert(layer_name: str, b64image: Base64Image):
            nonlocal x, y, width, height, has_selection
            # QImage.Format_RGB32 (4) is default format after decoding image
            # QImage.Format_RGBA8888 (17) is format used in Krita tutorial
            # both are compatible, & converting from 4 to 17 required a RGB swap
            # Likewise for 5 & 18 (their RGBA counterparts)
            image = b64image.img
            print(
                f"inserting {layer_name}: {image.width()}x{image.height()}, depth: {image.depth()}, format: {image.format()}"
            )

            # Image won't be scaled down ONLY if there is no selection; i.e. selecting whole image will scale down,
            # not selecting anything won't scale down, leading to the canvas being resized afterwards
            if (has_selection or inpaint) and (image.width() != width or image.height() != height):
                print(f"Rescaling image to selection: {width}x{height}")
                image = image.scaled(
                    width, height, transformMode=Qt.SmoothTransformation
                )

            # Resize (not scale!) canvas if image is larger (i.e. outpainting or Upscale was used)
            if image.width() > self.doc.width() or image.height() > self.doc.height():
                # NOTE:
                # - user's selection will be partially ignored if image is larger than canvas
                # - it is complex to scale/resize the image such that image fits in the newly scaled selection
                # - the canvas will still be resized even if the image fits after transparency masking
                print("Image is larger than canvas! Resizing...")
                new_width, new_height = self.doc.width(), self.doc.height()
                if image.width() > self.doc.width():
                    x, width, new_width = 0, image.width(), image.width()
                if image.height() > self.doc.height():
                    y, height, new_height = 0, image.height(), image.height()
                self.doc.resizeImage(0, 0, new_width, new_height)

            ba = img_to_ba(image)
            layer = create_layer(layer_name)
            # layer.setColorSpace() doesn't pernamently convert layer depth etc...

            # Don't fail silently for setPixelData(); fails if bit depth or number of channels mismatch
            if not skip_check_pixel_data:
                size = ba.size()
                expected = layer.pixelData(int(x), int(y), int(width), int(height)).size()
                assert expected == size, f"Raw data size: {size}, Expected size: {expected}"

            print(f"inserting at x: {x}, y: {y}, w: {width}, h: {height}")
            layer.setPixelData(ba, int(x), int(y), int(width), int(height))
            self._inserted_layers.append(layer)

            return layer

        return insert
    
    def basic_callback_crafter(self, mode):
        is_inpaint = mode == "inpaint"

        glayer = self.doc.createGroupLayer("Unnamed Group")
        self.doc.rootNode().addChildNode(glayer, None)

        insert = self.img_inserter(
            self.x, self.y, self.width, self.height, False, glayer, mode == "receive"
        )
        if not is_inpaint:
            mask_trigger = self.transparency_mask_inserter()

        def cb(response: PromptResponse):
            assert response is not None
            response.mode = mode
            # if is_upscale:
            #     insert(f"upscale", outputs[0])
            #     self.doc.refreshProjection()
            # else:
            if len(response.image_info) < 1:
                self.status_changed.emit(ERR_EMPTY_RESPONSE)
            try:
                layers = [
                    insert(name if name else f"{mode} {i + 1}", output)
                    for output, name, i in response.image_insert_list()
                ]
            except Exception as e:
                try:
                    self.client.images_received.disconnect(cb)
                except TypeError:
                    pass
                assert False, e

            if self.cfg("hide_layers", bool):
                for layer in layers[:-1]:
                    layer.setVisible(False)

            if glayer:
                glayer.setName(response.to_base_prompt_json())
            self.doc.refreshProjection()

            if not is_inpaint:
                mask_trigger(layers)

            self.client.images_received.disconnect(cb)
        
        return cb, glayer
    
    def check_controlnet_enabled(self):
        for i in range(len(self.cfg("controlnet_unit_list", "QStringList"))):
            if self.cfg(f"controlnet{i}_enable", bool):
                return True
            
    def get_controlnet_input_images(self, selected):
        input_images = dict()

        for i in range(len(self.cfg("controlnet_unit_list", "QStringList"))):    
            if self.cfg(f"controlnet{i}_enable", bool):
                input_image = b64_to_img(self.cfg(f"controlnet{i}_input_image", str)) if \
                    self.cfg(f"controlnet{i}_input_image", str) else selected
                    
                input_images.update({f"{i}": input_image})

        return input_images

    def apply_txt2img(self):
        cb, glayer = self.basic_callback_crafter("txt2img")

        sel_image = self.get_selection_image()
        self.client.post_txt2img(
            cb, self.width, self.height, sel_image,
            self.get_controlnet_input_images(sel_image)
        )

    def apply_img2img(self, is_inpaint):
        cb, glayer = self.basic_callback_crafter("img2img") if not is_inpaint else \
            self.basic_callback_crafter("inpaint")

        mask_image, transparency_mask = self.get_mask_image() if is_inpaint else (None, None)

        if is_inpaint and self.cfg("inpaint_auto_generate_mask", bool) and mask_image is not None:
            # auto-hide mask layer before getting selection image
            self.node.setVisible(False)
            self.inpaint_transparency_mask_inserter(glayer, transparency_mask)
            self.doc.refreshProjection()

        sel_image = self.get_selection_image()

        if is_inpaint:
            self.client.post_inpaint(
                cb, sel_image, mask_image, self.width, self.height, 
                self.get_controlnet_input_images(sel_image))
        else:
            self.client.post_img2img(
                cb, sel_image, self.width, self.height, self.get_controlnet_input_images(sel_image))
    
    def inpaint_transparency_mask_inserter(self, glayer, transparency_mask):
        orig_selection = self.selection.duplicate() if self.selection else None
        create_mask = self.cfg("create_mask_layer", bool)
        add_mask_action = self.app.action("add_new_transparency_mask")
        merge_mask_action = self.app.action("flatten_layer")

        if orig_selection:
            sx = orig_selection.x()
            sy = orig_selection.y()
            sw = orig_selection.width()
            sh = orig_selection.height()
        else:
            sx = 0
            sy = 0
            sw = self.doc.width()
            sh = self.doc.height()

        # must convert mask to single channel format
        gray_mask = transparency_mask.convertToFormat(QImage.Format_Grayscale8)
        
        mw = gray_mask.width()
        mh = gray_mask.height()
        # crop mask to the actual selection size
        crop_rect = QRect(int((mw - sw)/2), int((mh - sh)/2), int(sw), int(sh))
        crop_mask = gray_mask.copy(crop_rect)

        mask_ba = img_to_ba(crop_mask)

        # Why is sizeInBytes() different from width * height? Just... why?
        w = crop_mask.bytesPerLine() 
        h = crop_mask.sizeInBytes()/w

        mask_selection = Selection()
        mask_selection.setPixelData(mask_ba, int(sx), int(sy), int(w), int(h))

        def apply_mask_when_ready():
            # glayer will be selected when it is done being created
            if self.doc.activeNode() == glayer: 
                self.doc.setSelection(mask_selection)
                add_mask_action.trigger()
                self.doc.setSelection(orig_selection)
                timer.stop()

        timer = QTimer()
        timer.timeout.connect(apply_mask_when_ready)
        timer.start(50)

    def apply_controlnet_preview_annotator(self): 
        unit = self.cfg("controlnet_unit", str)
        if self.cfg(f"controlnet{unit}_input_image"):
            image = b64_to_img(self.cfg(f"controlnet{unit}_input_image"))
        else:
            image = self.get_selection_image()    

        def cb(response: PromptResponse):
            assert response is not None, ERR_BACKEND
            if len(response.image_info) < 1:
                self.status_changed.emit(ERR_EMPTY_RESPONSE)
            else:
                output = response.images[0]
                pixmap = QPixmap.fromImage(output.img)
                self.controlnet_preview_annotator_received.emit(pixmap)

        self.client.post_controlnet_preview(cb, image)

    def apply_simple_upscale(self):
        insert = self.img_inserter(self.x, self.y, self.width, self.height)
        sel_image = self.get_selection_image()

        path = os.path.join(self.cfg("sample_path", str), f"{int(time.time())}.png")
        if self.cfg("save_temp_images", bool):
            save_img(sel_image, path)

        def cb(response: PromptResponse):
            assert response is not None, ERR_BACKEND
            output = response.images[0]
            insert(f"upscale", output)
            self.doc.refreshProjection()
            self.client.images_received.disconnect(cb)

        self.client.post_upscale(cb, sel_image)

    def apply_get_workflow(self, mode):
        params = {}
        if mode != "inpaint":
            sel_image = self.get_selection_image()
            controlnet_input_images = self.get_controlnet_input_images(sel_image)
        if mode == "txt2img":
            params = self.client.post_txt2img(
                None, self.width, self.height, sel_image, controlnet_input_images
            )
        if mode == "img2img":
            params = self.client.post_img2img(
                None, sel_image, self.width, self.height, controlnet_input_images
            )
        if mode == "inpaint":
            mask_image, transparency_mask = self.get_mask_image()
            if self.node.visible():
                self.node.setVisible(False)
                self.doc.refreshProjection()
                sel_image = self.get_selection_image()
                self.node.setVisible(True)
                self.doc.refreshProjection()
            else:
                sel_image = self.get_selection_image()

            controlnet_input_images = self.get_controlnet_input_images(sel_image)
            params = self.client.post_inpaint(
                None, sel_image, mask_image, self.width, self.height, controlnet_input_images
            )
        if mode == "upscale":
            params = self.client.post_upscale(None, sel_image)
        return json.dumps(params, indent=4)
    
    def apply_run_workflow(self, workflow):
        # freeze selection region
        mode = self.cfg("workflow_to", str)
        is_inpaint = mode == "inpaint"
        #is_upscale = self.cfg("workflow_to", str) == "upscale"

        cb, glayer = self.basic_callback_crafter(mode)
        
        if is_inpaint and self.cfg("inpaint_auto_generate_mask", bool) and mask_image is not None:
            mask_image, transparency_mask = self.get_mask_image()
            self.node.setVisible(False)
            self.doc.refreshProjection()
            sel_image = self.get_selection_image()
            self.inpaint_transparency_mask_inserter(glayer, transparency_mask)
        else:
            mask_image = sel_image = self.get_selection_image()

        self.client.run_workflow(workflow, sel_image, mask_image, cb)

    def apply_get_last_images(self):

        cb, glayer = self.basic_callback_crafter("receive")

        self.client.receive_images("", skip_status_check=True)
        self.client.images_received.connect(cb)

    def transparency_mask_inserter(self):
        """Mask out extra regions due to adjust_selection()."""
        orig_selection = self.selection.duplicate() if self.selection else None
        create_mask = self.cfg("create_mask_layer", bool)

        add_mask_action = self.app.action("add_new_transparency_mask")
        merge_mask_action = self.app.action("flatten_layer")

        # This function is recursive to workaround race conditions when calling Krita's actions
        def add_mask(layers: list, cur_selection):
            if len(layers) < 1:
                self.doc.setSelection(cur_selection)  # reset to current selection
                return
            layer = layers.pop()

            orig_visible = layer.visible()
            orig_name = layer.name()

            def restore():
                # assume newly flattened layer is active
                result = self.doc.activeNode()
                result.setVisible(orig_visible)
                result.setName(orig_name)

                add_mask(layers, cur_selection)

            layer.setVisible(True)
            self.doc.setActiveNode(layer)
            self.doc.setSelection(orig_selection)
            add_mask_action.trigger()
                
            if create_mask:
                # collapse transparency mask by default
                layer.setCollapsed(True)
                layer.setVisible(orig_visible)
                QTimer.singleShot(
                    ADD_MASK_TIMEOUT, lambda: add_mask(layers, cur_selection)
                )
            else:
                # flatten transparency mask into layer
                merge_mask_action.trigger()
                QTimer.singleShot(ADD_MASK_TIMEOUT, lambda: restore())

        def trigger_mask_adding(layers: list):
            layers = layers[::-1]  # causes final active layer to be the top one

            def handle_mask():
                cur_selection = self.selection.duplicate() if self.selection else None
                add_mask(layers, cur_selection)

            QTimer.singleShot(ADD_MASK_TIMEOUT, lambda: handle_mask())

        return trigger_mask_adding

    # Actions
    def action_txt2img(self):
        self.status_changed.emit(STATE_WAIT)
        self.update_selection()
        if not self.doc:
            return
        self.adjust_selection()
        self.apply_txt2img()

    def action_img2img(self):
        self.status_changed.emit(STATE_WAIT)
        self.update_selection()
        if not self.doc:
            return
        self.adjust_selection()
        self.apply_img2img(False)

    def action_sd_upscale(self):
        assert False, "disabled"
        self.status_changed.emit(STATE_WAIT)
        self.update_selection()
        self.apply_img2img(mode=2)

    def action_inpaint(self):
        self.status_changed.emit(STATE_WAIT)
        self.update_selection()
        if not self.doc:
            return
        self.adjust_selection()
        self.apply_img2img(True)

    def action_simple_upscale(self):
        self.status_changed.emit(STATE_WAIT)
        self.update_selection()
        if not self.doc:
            return
        self.apply_simple_upscale()

    def action_update_config(self):
        """Update certain config/state from the backend."""
        self.client.get_config()
            
    def action_update_controlnet_config(self):
        """Update controlnet config from the backend."""
        self.client.get_controlnet_config()

    def action_preview_controlnet_annotator(self):
        self.status_changed.emit(STATE_WAIT)
        self.update_selection()
        if not self.doc:
            return
        self.adjust_selection()
        self.apply_controlnet_preview_annotator()

    def action_run_workflow(self, workflow):
        self.status_changed.emit(STATE_WAIT)
        self.update_selection()
        if not self.doc:
            return
        self.adjust_selection()
        self.apply_run_workflow(workflow)
            
    def action_update_controlnet_config(self):
        """Update controlnet config from the backend."""
        self.client.get_controlnet_config()

    def action_preview_controlnet_annotator(self):
        self.status_changed.emit(STATE_WAIT)
        self.update_selection()
        if not self.doc:
            return
        self.adjust_selection()
        self.apply_controlnet_preview_annotator()

    def action_interrupt(self):
        def cb(resp=None):
            self.client.interrupted = True
            self.status_changed.emit(STATE_INTERRUPT)

        self.client.post_interrupt(cb)

    def action_get_workflow(self, mode):
        self.update_selection()
        if not self.doc:
            return
        return self.apply_get_workflow(mode)
    
    def action_get_last_images(self):
        self.update_selection()
        if not self.doc:
            return
        self.adjust_selection()
        self.apply_get_last_images()

    def action_update_eta(self):
        self.client.check_progress(self.progress_update.emit)


script = Script()