import itertools
import json
import re
import copy
from dataclasses import dataclass, field
from enum import Enum, unique
from typing import List, Dict, Tuple

from PyQt5.QtCore import QByteArray, QSize, QBuffer, QIODevice
from PyQt5.QtGui import QImage

from .defaults import (
    DEFAULT_NODE_IDS
    , PROMPT
    , PRUNED_DATA
)


def _default(self, obj):
    return getattr(obj.__class__, "to_json", _default.default)(obj)


_default.default = json.JSONEncoder().default
json.JSONEncoder.default = _default


def get_members(obj):
    return [attr for attr in dir(obj) if not callable(getattr(obj, attr)) and not attr.startswith("__")]


def bind_or_none(data: Dict, keys: List[str]):
    node = data
    for k in keys:
        if node is not None:
            node = node.get(k, None)
        else:
            break
    return node


def print_attributes(obj):
    output = ""
    members = get_members(obj)
    output = obj.__class__.__name__ + " ["
    first: bool = True
    for m in members:
        if not first:
            output += ", "
        first = False
        output += str(m) + ": "
        if isinstance(getattr(obj, m), str):
            output += '"' + str(getattr(obj, m)).replace('\n', '') + '"'
        else:
            output += str(getattr(obj, m))
    output += "]"
    return output


def update_json(json_obj, key_to_update, new_value):
    if isinstance(json_obj, dict):
        for key, value in json_obj.items():
            if key == key_to_update:
                json_obj[key] = new_value
            else:
                update_json(value, key_to_update, new_value)
    elif isinstance(json_obj, list):
        for item in json_obj:
            update_json(item, key_to_update, new_value)


def add_loras_from_history(history):
    node_name = DEFAULT_NODE_IDS["LoraLoader"]
    output = ""
    lora_loader_count = 1
    try:
        nodes = history["prompt"][2]
        node_inputs = nodes[f"{node_name}+{lora_loader_count}"]["inputs"]
        while True:
            lora_name = re.sub(".safetensors", "", node_inputs["lora_name"])
            lora_weight = node_inputs["strength_model"]
            # < > are hidden from the lable names but are still in the string
            # I think krita is trying to hide XML from the user
            output += f"\n<lora:{lora_name}:{lora_weight}>"
            lora_loader_count += 1
            node_inputs = nodes[f"{node_name}+{lora_loader_count}"]["inputs"]
    except (KeyError, ValueError, IndexError):
        return output


@unique
class PromptMode(str, Enum):
    empty = "empty"
    txt2img = "txt2img"
    img2img = "img2img"
    inpaint = "inpaint"
    upscale = "upscale"


@dataclass
class Base64Image:
    """
    This is a container for QImage.
    it offers the functionality to convert to base64 if needed
    and then keep the result to avoid unnecessary conversions.
    if you modify the internal QImage re-create this object.
    """
    img: QImage = field(default_factory=QImage)
    size: QSize = field(default_factory=QSize)
    __b64str: str = None

    def get_b64str(self):
        if self.__b64str is None:
            ba: QByteArray = QByteArray()
            buffer: QBuffer = QBuffer(ba)
            buffer.open(QIODevice.WriteOnly)
            self.img.save(buffer, "PNG", 0)
            self.__b64str = ba.toBase64().data().decode("utf-8")
        return self.__b64str

    @staticmethod
    def from_image(img: QImage):
        self = Base64Image()
        self.img = img
        self.size = img.size()
        return self

    @staticmethod
    def from_b64(b64str: str):
        self = Base64Image()
        ba = QByteArray.fromBase64(b64str.encode("utf-8"))
        self.img = QImage.fromData(ba)
        self.size = self.img.size()
        return self

    def __bool__(self):
        return len(self.__b64str) > 0

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return "{}: {}".format(self.__class__.__name__, self.size)

    def to_json(self) -> str:
        return json.dumps(self.__b64str)


@dataclass
class ControlNetUnit:
    enable: bool = False
    preprocessor: str = "None"
    model: str = "None"
    weight: float = 1.0
    guidance_start: float = 0
    guidance_end: float = 1
    input_image: Base64Image = None
    # controlnet0_inputs Unused?

    def __bool__(self):
        return self.enable

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        if self.enable:
            return print_attributes(self)
        else:
            return str(self.enable)


@dataclass
class PromptBase:
    """
        This class is supposed to represent minimal amount of data related to a prompt.
        We don't want the image data here for that use the Response class
    """
    mode: PromptMode = PromptMode.empty
    sd_model: str = None
    clip_skip: int = None
    seed: int = None
    pos_prompt: str = None
    neg_prompt: str = None
    sampler: str = None
    scheduler: str = None
    steps: str = None
    cfg: float = None
    denoise: float = None

    def __str__(self):
        return print_attributes(self)

    def to_json(self) -> str:
        keys = get_members(PromptBase)
        return json.dumps({k: v for k, v in self.__dict__.items() if k in keys})

    @staticmethod
    def from_json(text: str):
        prompt = PromptBase()
        keys = get_members(PromptBase)
        # Skip text until json starts
        prompt_data = json.loads(text[text.find('{'):])
        for k in keys:
            setattr(prompt, k, prompt_data[k])
        return prompt


@dataclass
class PromptResponse(PromptBase):
    """This class represents a response from the ComfyUI server"""
    history: dict = field(default_factory=dict)
    image_info: List[Tuple[str, str, str]] = field(default_factory=list)  # name, subfolder, type
    images: List[Base64Image] = field(default_factory=list)

    @staticmethod
    def from_history_json(history: dict, images: List[QImage] = None, image_names: List[str] = None):
        ids = DEFAULT_NODE_IDS
        self = PromptResponse()
        self.history = copy.deepcopy(history)
        update_json(self.history, 'image', PRUNED_DATA)
        update_json(self.history, 'text', PROMPT)

        try:
            # Attempt to parse the history json
            # TODO: Can we get the Mode here?
            #  yes we could infer inpaint from the latent_image is SetLatentNoiseMask or VAEEncodeForInpaint
            #  or txt2img latent_image is "5" or EmptyLatentImage
            #  or img2img latent_image is VAEEncode
            workflow: dict = history["prompt"][2]
            output_name: dict = history["prompt"][4][0]
            # Model
            self.sd_model = bind_or_none(workflow, [ids["CheckpointLoaderSimple"], "inputs", "ckpt_name"])
            self.clip_skip = bind_or_none(workflow, [ids["ClipSetLastLayer"], "inputs", "stop_at_clip_layer"])

            # Prompt
            self.pos_prompt = bind_or_none(workflow, [ids["ClipTextEncode_pos"], "inputs", "text"])
            self.neg_prompt = bind_or_none(workflow, [ids["ClipTextEncode_neg"], "inputs", "text"])
            if self.pos_prompt is not None:
                self.pos_prompt = self.pos_prompt.strip()
                self.pos_prompt += add_loras_from_history(history)
            if self.neg_prompt is not None:
                self.neg_prompt = self.neg_prompt.strip()

            # KSampler
            ksampler_inputs = bind_or_none(workflow, [ids["KSampler"], "inputs"])
            ksampler_inputs_latent = bind_or_none(ksampler_inputs, ["latent_image"])
            if ksampler_inputs is not None:
                self.cfg = ksampler_inputs.get("cfg")
                self.sampler = ksampler_inputs.get("sampler_name")
                self.scheduler = ksampler_inputs.get("scheduler")
                self.seed = ksampler_inputs.get("seed")
                self.steps = ksampler_inputs.get("steps")
                # txt2img without hires always sets the denoise to 1
                if ksampler_inputs_latent is not None and not ksampler_inputs_latent[0] == ids["EmptyLatentImage"]:
                    self.denoise = round(ksampler_inputs.get("denoise"), 2)

            # KSampler_HiRes
            ksampler_hires_inputs = bind_or_none(workflow, [ids["KSampler_upscale"], "inputs"])
            if ksampler_hires_inputs is not None:
                self.denoise = round(ksampler_hires_inputs.get("denoise"), 2)
                self.steps += ksampler_hires_inputs.get("steps")

            if image_names is not None:
                for i in image_names:
                    self.image_info.append((i, "", ""))
            else:
                output_images_info = bind_or_none(history, ['outputs', output_name, 'images'])
                if output_images_info is not None:
                    for image in output_images_info:
                        self.image_info.append((image["filename"], image["subfolder"], image["type"]))

        except (KeyError, ValueError, IndexError):
            # Due to the way response can be used with other workflows
            # the above can fail to resolve continue and pass the images back
            pass

        if images is not None:
            self.images = [Base64Image.from_image(i) for i in images]
            assert len(self.images) == len(self.image_info)

        return self

    def to_base_prompt_json(self) -> str:
        return super().to_json()

    def to_json(self) -> str:
        keys = get_members(PromptResponse)
        return json.dumps({k: v for k, v in self.__dict__.items() if k in keys})

    def append_image(self, img: QImage):
        assert len(self.images) < len(self.image_info)
        self.images.append(Base64Image.from_image(img))

    def image_insert_list(self):
        assert len(self.images) == len(self.image_info)
        return zip(self.images, [i[0] for i in self.image_info], itertools.count())

    def image_names(self):
        return [i[0] for i in self.image_info]


@dataclass
class Prompt(PromptBase):
    """
    This class is supposed to represent all the information needed to
    issue a command to the backend or convert to a workflow
    Including base64 encoded image info.
    This class should be:
     - Simple to copy
     - Not link to or be dependent on the global state after construction
     - have helpers to:
        - Extract information
        - Export/import different formats
    """
    # base 64 encoded Images
    src_img: Base64Image = field(default_factory=Base64Image)
    mask_img: Base64Image = field(default_factory=Base64Image)
    # ControlNets
    controlnet = [ControlNetUnit() for i in range(10)]




