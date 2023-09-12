import json
import socket
import uuid
import re
from math import ceil, floor
from random import randint
from typing import Any
from urllib.error import URLError
from urllib.parse import urljoin, urlparse, urlencode
from urllib.request import Request, urlopen
import ssl

from krita import (
    Qt,
    QImage,
    QObject,
    QThread,
    pyqtSignal,
    QTimer
)

from .config import Config
from .defaults import (
    CURRENT_LAYER_AS_MASK,
    DEFAULT_NODE_IDS,
    ERR_BAD_URL,
    ERR_MISSING_NODE,
    ERR_NO_CONNECTION,
    LAST_LOADED_LORA,
    LONG_TIMEOUT,
    NEGATIVE_PROMPT,
    PROMPT,
    PRUNED_DATA,
    ROUTE_PREFIX,
    SELECTED_IMAGE,
    SHORT_TIMEOUT,
    STATE_DONE,
    STATE_READY,
    STATE_URLERROR,
    THREADED,
)
from .utils import (
    bytewise_xor,
    img_to_b64,
    calculate_resized_image_dimensions
)

from .prompt import PromptResponse

# NOTE: backend queues up responses, so no explicit need to block multiple requests
# except to prevent user from spamming themselves

# TODO: tab showing all queued up requests (local plugin instance only)


def get_url(cfg: Config, route: str = ..., prefix: str = ROUTE_PREFIX):
    base = cfg("base_url", str)
    if not urlparse(base).scheme in {"http", "https"}:
        return None
    url = urljoin(base, prefix)
    if route is not ...:
        url = urljoin(url, route)
    # print("url:", url)
    return url

# krita doesn't reexport QtNetwork


class AsyncRequest(QObject):
    timeout = None
    finished = pyqtSignal()
    result = pyqtSignal(object)
    error = pyqtSignal(Exception)

    def __init__(
        self,
        base_url: str,
        data: Any = None,
        timeout: int = ...,
        method: str = ...,
        headers: dict = ...,
        key: str = None,
    ):
        """Create an AsyncRequest object.

        By default, AsyncRequest has no timeout, will infer whether it is "POST"
        or "GET" based on the presence of `data` and uses JSON to transmit. It
        also assumes the response is JSON.

        Args:
            url (str): URL to request from.
            data (Any, optional): Payload to send. Defaults to None.
            timeout (int, optional): Timeout for request. Defaults to `...`.
            method (str, optional): Which HTTP method to use. Defaults to `...`.
            key (Union[str, None], Optional): Key to use for encryption/decryption. Defaults to None.
            headers (dict, optional): dictionary of headers to send to request.
            is_upload: if set to true, request will be sent as multipart/form-data type.
        """
        super(AsyncRequest, self).__init__()
        self.url = base_url
        self.data = data
        self.headers = {} if headers is ... else headers

        self.key = None
        if isinstance(key, str) and key.strip() != "":
            self.key = key.strip().encode("utf-8")

        if self.key is not None:
            self.headers["X-Encrypted-Body"] = "XOR"
        if timeout is not ...:
            self.timeout = timeout
        if method is ...:
            self.method = "GET" if data is None else "POST"
        else:
            self.method = method
        if method == "POST":
            self.data = None if data is None else json.dumps(
                data).encode("utf-8")

            if self.data is not None and self.key is not None:
                # print(f"Encrypting with ${self.key}:\n{self.data}")
                self.data = bytewise_xor(self.data, self.key)
                # print(f"Encrypt Result:\n{self.data}")
                self.headers["Content-Type"] = "application/json"
                self.headers["Content-Length"] = str(len(self.data))
        self.headers["User-Agent"] = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36 Edg/115.0.1901.200"
        #self.headers["accept"] = "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7"
    def run(self):
        try:
            url = self.url
            ctx = None
            if url.startswith('https://'):
                ctx = ssl.create_default_context()
            if self.method == "GET":
                url = url if self.data is None else f"{self.url}?{urlencode(self.data)}"
            req = Request(url, headers=self.headers, method=self.method)
            with urlopen(req, self.data if self.method == "POST" else None, self.timeout, context=ctx) as res:
                data = res.read()
                enc_type = res.getheader("X-Encrypted-Body", None)
                assert enc_type in {"XOR", None}, "Unknown server encryption!"
                if enc_type == "XOR":
                    assert self.key, "Key needed to decrypt server response!"
                    print(f"Decrypting with ${self.key}:\n{data}")
                    data = bytewise_xor(data, self.key)
                self.result.emit(json.loads(data))
        except ValueError as e:
            self.result.emit(data)
        except Exception as e:
            self.error.emit(e)
            return
        finally:
            self.finished.emit()

    @classmethod
    def request(cls, *args, **kwargs):
        req = cls(*args, **kwargs)
        if THREADED:
            thread = QThread()
            # NOTE: need to keep reference to thread or it gets destroyed
            req.thread = thread
            req.moveToThread(thread)
            thread.started.connect(req.run)
            req.finished.connect(thread.quit)
            # NOTE: is this a memory leak?
            # For some reason, deleteLater occurs while thread is still running, resulting in crash
            # req.finished.connect(req.deleteLater)
            # thread.finished.connect(thread.deleteLater)
            return req, lambda: thread.start()
        else:
            return req, lambda: req.run()


class Client(QObject):
    status = pyqtSignal(str)
    config_updated = pyqtSignal()
    images_received = pyqtSignal(object)
    prompt_sent = pyqtSignal()

    lora_re = r"<lora:([=\[\] \\/\w\d.-]+):(\-?[\d.]+)>"

    def __init__(self, cfg: Config, ext_cfg: Config):
        """It is highly dependent on config's structure to the point it writes directly to it. :/"""
        super(Client, self).__init__()
        self.cfg = cfg
        self.ext_cfg = ext_cfg
        self.short_reqs = set()
        self.long_reqs = set()
        # NOTE: this is a hacky workaround for detecting if backend is reachable
        self.is_connected = False
        self.interrupted = False
        self.client_id = str(uuid.uuid4())
        self.conn = lambda s: None

    def handle_api_error(self, exc: Exception):
        """Handle exceptions that can occur while interacting with the backend."""
        self.is_connected = False
        try:
            # wtf python? socket raises an error that isnt an Exception??
            if isinstance(exc, socket.timeout):
                raise TimeoutError
            else:
                raise exc
        except URLError as e:
            self.status.emit(f"{STATE_URLERROR}: {e.reason}")
        except TimeoutError as e:
            self.status.emit(f"{STATE_URLERROR}: response timed out")
        except json.JSONDecodeError as e:
            self.status.emit(f"{STATE_URLERROR}: invalid JSON response")
        except ValueError as e:
            self.status.emit(f"{STATE_URLERROR}: Invalid backend URL")
        except ConnectionError as e:
            self.status.emit(
                f"{STATE_URLERROR}: connection error during request")
        except Exception as e:
            # self.status.emit(f"{STATE_URLERROR}: Unexpected Error")
            # self.status.emit(str(e))
            assert False, e

    def receive_images(self, status, prompt_id=None, skip_status_check=False):
        def on_history_received(history_res):
            def on_image_received(img):
                assert img is not None, "Backend Error, check terminal"
                qimage = QImage()
                qimage.loadFromData(img)
                response.append_image(qimage)
                # Check if all images are in the response before sending to the script.
                if len(response.images) == len(response.image_info):
                    self.images_received.emit(response)

            assert history_res is not None, "Backend Error, check terminal"
            history = history_res[prompt_id] if prompt_id is not None else history_res[list(history_res.keys())[-1]]
            response = PromptResponse.from_history_json(history)

            # Server error occured empty response
            if len(response.image_info) < 1:
                self.images_received.emit(response)

            for image in response.image_info:
                self.get_image(image[0], image[1], image[2], on_image_received)

        if status == STATE_DONE or skip_status_check:
            # Prevent undesired executions of this function.
            if status == STATE_DONE:
                try:
                    self.status.disconnect(self.conn)
                except TypeError:
                    pass # signal was not connected
            self.get_history(prompt_id, on_history_received)

    def queue_prompt(self, prompt, cb=None):
        p = {"prompt": prompt, "client_id": self.client_id}
        self.post("prompt", p, cb, is_long=False)

    def get_image(self, filename, subfolder, folder_type, cb=None):
        data = {"filename": filename,
                "subfolder": subfolder, "type": folder_type}
        self.get("view", cb, data=data)

    def get_history(self, prompt_id=None, cb=None):
        if prompt_id is not None:
            self.get(f"history/{prompt_id}", cb)
        else:
            self.get("history", cb)

    def check_progress(self, cb):
        def on_progress_checked(res):
            if not self.interrupted and len(res["queue_running"]) == 0 \
                    and self.is_connected:
                self.status.emit(STATE_DONE)

            cb(res)

        self.get("queue", on_progress_checked)

    def get_images(self, prompt, cb):
        def on_prompt_received(prompt_res):
            assert prompt_res is not None, "Backend Error, check terminal"
            self.prompt_sent.emit()
            prompt_id = prompt_res['prompt_id']
            self.conn = lambda s: self.receive_images(s, prompt_id)
            self.status.connect(self.conn)

        self.queue_prompt(prompt, on_prompt_received)
        self.images_received.connect(cb)

    def post(
        self, route, body, cb, base_url=..., is_long=True, ignore_no_connection=False, method="POST", headers={}
    ):
        if not ignore_no_connection and not self.is_connected:
            self.status.emit(ERR_NO_CONNECTION)
            return
        url = get_url(self.cfg, route) if base_url is ... else urljoin(
            base_url, route)
        if not url:
            self.status.emit(ERR_BAD_URL)
            return
        req, start = AsyncRequest.request(
            url,
            body,
            LONG_TIMEOUT if is_long else SHORT_TIMEOUT,
            key=self.cfg("encryption_key"),
            method=method,
            headers=headers
        )

        if is_long:
            self.long_reqs.add(req)
        else:
            self.short_reqs.add(req)

        def handler():
            self.long_reqs.discard(req)
            self.short_reqs.discard(req)
            if is_long and len(self.long_reqs) == 0:
                self.status.emit(STATE_DONE)

        req.result.connect(cb)
        req.error.connect(lambda e: self.handle_api_error(e))
        req.finished.connect(handler)
        start()

    def get(self, route, cb, data=None, base_url=..., is_long=False, ignore_no_connection=False):
        self.post(
            route,
            data,
            cb,
            base_url=base_url,
            is_long=is_long,
            ignore_no_connection=ignore_no_connection,
            method="GET"
        )

    def check_params(self, params, keys):
        "Check if the defined nodes exist in params"
        for key in keys:
            try:
                assert key in params
            except:
                self.status.emit(ERR_MISSING_NODE)
                return False
        return True

    def common_params(self):
        """Parameters used by most modes."""
        checkpointloadersimple_node = {
            "class_type": "CheckpointLoaderSimple",
            "inputs": {
                "ckpt_name": self.cfg("sd_model", str)
            }
        }
        vaedecode_node = {
            "class_type": "VAEDecode",
            "inputs": {
                "samples": [
                    DEFAULT_NODE_IDS["KSampler"],
                    0
                ],
                "vae": [
                    DEFAULT_NODE_IDS["CheckpointLoaderSimple"],
                    2
                ]
            }
        }
        saveimage_node = {
            "class_type": "SaveImage",
            "inputs": {
                "filename_prefix": "ComfyUI",
                "images": [
                    DEFAULT_NODE_IDS["VAEDecode"],
                    0
                ]
            }
        }
        clipsetlastlayer_node = {
            "class_type": "CLIPSetLastLayer",
            "inputs": {
                "stop_at_clip_layer": -self.cfg("clip_skip", int),
                "clip": [
                    DEFAULT_NODE_IDS["CheckpointLoaderSimple"],
                    1
                ]
            }
        }
        params = {
            DEFAULT_NODE_IDS["CheckpointLoaderSimple"]: checkpointloadersimple_node,
            DEFAULT_NODE_IDS["VAEDecode"]: vaedecode_node,
            DEFAULT_NODE_IDS["SaveImage"]: saveimage_node,
            DEFAULT_NODE_IDS["ClipSetLastLayer"]: clipsetlastlayer_node
        }

        def loadVAE():
            nonlocal params
            vaeloader_node = {
                "class_type": "VAELoader",
                "inputs": {
                    "vae_name": self.cfg("sd_vae", str)
                }
            }
            params.update({
                DEFAULT_NODE_IDS["VAELoader"]: vaeloader_node
            })
            params[DEFAULT_NODE_IDS["VAEDecode"]]["inputs"]["vae"] = [
                DEFAULT_NODE_IDS["VAELoader"],
                0
            ]

        if self.cfg("sd_vae", str) != "None"  \
                and self.cfg("sd_vae", str) in self.cfg("sd_vae_list", "QStringList"):
            loadVAE()

        return params

    def auto_complete_LoRA(self, name: str):
        lora_list = [ re.sub(".safetensors$", "", l, flags=re.I) for l in self.cfg("sd_lora_list", str)]
        viable_loras = [l for l in lora_list if re.search(re.escape(name)+"$", l, flags=re.I)]
        if (len(viable_loras) == 1 and name == viable_loras[0]):
            return name
        elif (len(viable_loras) == 1):
            print("Using Lora:", viable_loras[0])
            return viable_loras[0]
        elif (len(viable_loras) > 1):
            print("Ambiguous Lora:", name, "\nCould be:", viable_loras )
        return name



    def loadLoRAs(self, params, mode, connect_last_lora_outputs = True):
        clipsetlastlayer_id = DEFAULT_NODE_IDS["ClipSetLastLayer"]
        checkpointloadersimple_id = DEFAULT_NODE_IDS["CheckpointLoaderSimple"]
        ksampler_id = DEFAULT_NODE_IDS["KSampler"]
        ksampler_upscale_id = DEFAULT_NODE_IDS["KSampler_upscale"]
        cliptextencode_pos_id = DEFAULT_NODE_IDS["ClipTextEncode_pos"]
        cliptextencode_neg_id = DEFAULT_NODE_IDS["ClipTextEncode_neg"]

        if not connect_last_lora_outputs or self.check_params(params, [
            ksampler_id, cliptextencode_pos_id, cliptextencode_neg_id
            ]):
            # Initialize a counter to keep track of the number of nodes added
            lora_count = 0
            pos_prompt = self.cfg(f"{mode}_prompt", str)

            # Use a regular expression to find all the elements between < and > in the string
            matches = re.findall(self.lora_re, pos_prompt)

            # Remove LoRAs from prompt
            params[cliptextencode_pos_id]["inputs"]["text"] = re.sub(self.lora_re, "", pos_prompt)

            # Loop through the matches and create a node for each element
            for match in matches:
                # Extract the lora name and the strength number from the match
                lora_name = self.auto_complete_LoRA(match[0])
                strength_number = float(match[1])

                # Create a node dictionary with the class type, inputs, and outputs
                clip = clipsetlastlayer_id if clipsetlastlayer_id in params else checkpointloadersimple_id
                prev_lora_id = f"{DEFAULT_NODE_IDS['LoraLoader']}+{lora_count}"
                node = {
                    "class_type": "LoraLoader",
                    "inputs": {
                        "lora_name": f"{lora_name}.safetensors",
                        "strength_model": strength_number,
                        "strength_clip": strength_number,
                        # If this is the first node, use the default model and clip parameters
                        # Otherwise, use the previous node id as the model and clip parameters
                        "model": [
                            checkpointloadersimple_id if lora_count == 0 else prev_lora_id,
                            0
                        ],
                        "clip": [
                            clip if lora_count == 0 else prev_lora_id,
                            1 if lora_count > 0 or clipsetlastlayer_id not in params else 0
                        ]
                    }
                }

                # Generate a node id by adding the node count to the default node id
                node_id = f"{DEFAULT_NODE_IDS['LoraLoader']}+{lora_count+1}"

                # Add the node to the params dictionary with the node id as the key
                params.update({node_id: node})

                # Increment the node count by one
                lora_count += 1

            if lora_count > 0:
                last_lora_id = f"{DEFAULT_NODE_IDS['LoraLoader']}+{lora_count}"
                if connect_last_lora_outputs:
                    #Connect KSampler to last lora node.
                    params[ksampler_id]["inputs"]["model"] = [last_lora_id, 0]

                    #Connect KSampler for upscale (second pass) to last lora node if found.
                    if ksampler_upscale_id in params:
                        params[ksampler_upscale_id]["inputs"]["model"] = [last_lora_id, 0]

                    #Connect positive prompt to lora clip.
                    params[cliptextencode_pos_id]["inputs"]["clip"] = [last_lora_id, 1]

                    #Connect negative prompt to lora clip.
                    params[cliptextencode_neg_id]["inputs"]["clip"] = [last_lora_id, 1]

                return last_lora_id

    def upscale_latent(self, params, width, height, seed, cfg_prefix):
        ksampler_id = DEFAULT_NODE_IDS["KSampler"]
        vaedecode_id = DEFAULT_NODE_IDS["VAEDecode"]

        if self.check_params(params, [ksampler_id, vaedecode_id]):
            latentupscale_node = {
                "class_type": "LatentUpscale",
                "inputs": {
                    "upscale_method": self.cfg("upscaler_name", str),
                    "width": width,
                    "height": height,
                    "crop": "disabled",
                    "samples": [ksampler_id, 0]
                }
            }
            denoise = self.cfg(f"{cfg_prefix}_denoising_strength", float)
            ksampler_upscale_node = {
                "class_type": "KSampler",
                "inputs": {
                    "cfg": 8,
                    "denoise": denoise if denoise < 1 else 0.30,
                    "model": [
                        DEFAULT_NODE_IDS["CheckpointLoaderSimple"],
                        0
                    ],
                    "latent_image": [
                        DEFAULT_NODE_IDS["LatentUpscale"],
                        0
                    ],
                    "negative": [
                        DEFAULT_NODE_IDS["ClipTextEncode_neg"],
                        0
                    ],
                    "positive": [
                        DEFAULT_NODE_IDS["ClipTextEncode_pos"],
                        0
                    ],
                    "sampler_name": self.cfg(f"{cfg_prefix}_sampler", str),
                    "scheduler": self.cfg(f"{cfg_prefix}_scheduler", str),
                    "seed": seed,
                    "steps": ceil(self.cfg(f"{cfg_prefix}_steps", int)/2)
                }
            }
            params.update({
                DEFAULT_NODE_IDS["LatentUpscale"]: latentupscale_node,
                DEFAULT_NODE_IDS["KSampler_upscale"]: ksampler_upscale_node
            })
            params[vaedecode_id]["inputs"]["samples"] = [
                DEFAULT_NODE_IDS["KSampler_upscale"],
                0
            ]
            params[ksampler_id]["inputs"]["steps"] = ceil(self.cfg(f"{cfg_prefix}_steps", int)/2)

    def upscale_with_model(self, params, width, height, seed, mode):
        vae_id = DEFAULT_NODE_IDS["VAELoader"]
        vaedecode_id = DEFAULT_NODE_IDS["VAEDecode"]
        ksampler_id = DEFAULT_NODE_IDS["KSampler"]

        if self.check_params(params, [vaedecode_id, ksampler_id]):
            vaedecode_upscale_node = {
                "class_type": "VAEDecode",
                "inputs": {
                    "samples": [
                        DEFAULT_NODE_IDS["KSampler"],
                        0
                    ],
                    "vae": [
                        vae_id if vae_id in params else DEFAULT_NODE_IDS["CheckpointLoaderSimple"],
                        0 if vae_id in params else 2
                    ]
                }
            }
            upscalemodelloader_node = {
                "class_type": "UpscaleModelLoader",
                "inputs": {
                    "model_name": self.cfg("upscaler_name", str)
                }
            }
            imageupscalewithmodel_node = {
                "class_type": "ImageUpscaleWithModel",
                "inputs": {
                    "upscale_model": [
                        DEFAULT_NODE_IDS["UpscaleModelLoader"],
                        0
                    ],
                    "image": [
                        DEFAULT_NODE_IDS["VAEDecode_upscale"],
                        0
                    ]
                }
            }
            scaleimage_node = {
                "class_type": "ImageScale",
                "inputs": {
                    "upscale_method": "bilinear",
                    "width": width,
                    "height": height,
                    "crop": "disabled",
                    "image": [
                        DEFAULT_NODE_IDS["ImageUpscaleWithModel"],
                        0
                    ]
                }
            }
            vaeencode_upscale_node = {
                "class_type": "VAEEncode",
                "inputs": {
                    "pixels": [
                        DEFAULT_NODE_IDS["ImageScale"],
                        0
                    ],
                    "vae": [
                        vae_id if vae_id in params else DEFAULT_NODE_IDS["CheckpointLoaderSimple"],
                        0 if vae_id in params else 2
                    ]
                }
            }
            denoise = self.cfg(f"{mode}_denoising_strength", float)
            ksampler_upscale_node = {
                "class_type": "KSampler",
                "inputs": {
                    "cfg": 8,
                    "denoise": denoise if denoise < 1 else 0.30,
                    "model": [
                        DEFAULT_NODE_IDS["CheckpointLoaderSimple"],
                        0
                    ],
                    "latent_image": [
                        DEFAULT_NODE_IDS["VAEEncode_upscale"],
                        0
                    ],
                    "negative": [
                        DEFAULT_NODE_IDS["ClipTextEncode_neg"],
                        0
                    ],
                    "positive": [
                        DEFAULT_NODE_IDS["ClipTextEncode_pos"],
                        0
                    ],
                    "sampler_name": self.cfg(f"{mode}_sampler", str),
                    "scheduler": self.cfg(f"{mode}_scheduler", str),
                    "seed": seed,
                    "steps": ceil(self.cfg(f"{mode}_steps", int)/2)
                }
            }
            params.update({
                DEFAULT_NODE_IDS["VAEDecode_upscale"]: vaedecode_upscale_node,
                DEFAULT_NODE_IDS["UpscaleModelLoader"]: upscalemodelloader_node,
                DEFAULT_NODE_IDS["ImageUpscaleWithModel"]: imageupscalewithmodel_node,
                DEFAULT_NODE_IDS["ImageScale"]: scaleimage_node,
                DEFAULT_NODE_IDS["VAEEncode_upscale"]: vaeencode_upscale_node,
                DEFAULT_NODE_IDS["KSampler_upscale"]: ksampler_upscale_node
            })
            params[vaedecode_id]["inputs"]["samples"] = [
                DEFAULT_NODE_IDS["KSampler_upscale"],
                0
            ]
            params[ksampler_id]["inputs"]["steps"] = ceil(self.cfg(f"{mode}_steps", int)/2)
        
    def apply_controlnet(self, params, controlnet_src_imgs):
        ksampler_id = DEFAULT_NODE_IDS["KSampler"]
        if self.check_params(params, [ksampler_id]) and controlnet_src_imgs:
            prev = "" #chain positive conditioning
            prev_neg = "" #chain negative conditioning

            for i in range(len(self.cfg("controlnet_unit_list", "QStringList"))):
                if self.cfg(f"controlnet{i}_enable", bool):
                    prev, prev_neg = self.controlnet_unit_params(params, img_to_b64(
                        controlnet_src_imgs[str(i)]), i, prev, prev_neg)
            
            if prev != "":
                params[ksampler_id]["inputs"]["positive"] = [prev, 0]
            if prev_neg != "":
                params[ksampler_id]["inputs"]["negative"] = [prev_neg, 1]

    def controlnet_unit_params(self, params, image: str, unit: int, prev = "", prev_neg = ""):
        #Image loading
        preprocessor = self.cfg(f"controlnet{unit}_preprocessor", str)
        imageloader_prefix = DEFAULT_NODE_IDS["ControlNetImageLoader"]
        image_node_exists = False
        num_imageloader_nodes = 0
        for key, value in params.items():
            if key.startswith(imageloader_prefix):
                if value["inputs"]["image"] == image:
                    image_node_exists = True
                    break
                num_imageloader_nodes += 1

        imageloader_node_id = f"{imageloader_prefix}+{num_imageloader_nodes}"

        if not image_node_exists:
            controlnet_imageloader_node = {
                "class_type": "LoadBase64Image",
                "inputs": {
                    "image": image
                }
            }
            params.update({
                imageloader_node_id: controlnet_imageloader_node
            })

        if "Inpaint" in preprocessor:
            mask_node_id = DEFAULT_NODE_IDS["LoadBase64ImageMask"]
            mask_node = mask_node_id if mask_node_id in params else imageloader_node_id
            mask_node_output_num = 0 if mask_node_id in params else 1

        #model loading
        controlnetloader_prefix = DEFAULT_NODE_IDS["ControlNetLoader"]
        clipvisionloader_prefix = DEFAULT_NODE_IDS["CLIPVisionLoader"]
        model_node_exists = False
        is_controlnet = True #Or revision
        num_controlnetloader_nodes = 0
        for key, value in params.items():
            if key.startswith(controlnetloader_prefix) or key.startswith(clipvisionloader_prefix):
                try:
                    if value["inputs"]["control_net_name"] == self.cfg(f"controlnet{unit}_model", str):
                        model_node_exists = True
                        break
                except KeyError:
                    if value["inputs"]["clip_name"] == self.cfg(f"controlnet{unit}_model", str):
                        model_node_exists = True
                        is_controlnet = False
                        break
                num_controlnetloader_nodes += 1

        controlnetloader_node_id = f"{controlnetloader_prefix if is_controlnet else clipvisionloader_prefix}+{num_controlnetloader_nodes}"
        if not model_node_exists:
            if preprocessor == "Revision":
                controlnet_loader_node = {
                "class_type": "CLIPVisionLoader",
                "inputs": {
                    "clip_name": self.cfg(f"controlnet{unit}_model", str)
                }
            }
            else:
                controlnet_loader_node = {
                    "class_type": "ControlNetLoader",
                    "inputs": {
                        "control_net_name": self.cfg(f"controlnet{unit}_model", str)
                    }
                }
            params.update({
                controlnetloader_node_id: controlnet_loader_node
            })

        inputs = self.cfg(f"controlnet{unit}_inputs")
        if preprocessor not in ["None", "Revision"]:
            inputs.update({"image": [imageloader_node_id, 0]})

            if "Inpaint" in preprocessor:
                inputs.update({"mask": [mask_node, mask_node_output_num]})

            preprocessor_class = self.cfg("controlnet_preprocessors_info", dict)[preprocessor]["class"]
            preprocessor_node = {
                "class_type": preprocessor_class,
                "inputs": inputs
            }
            preprocessor_node_id = f"{preprocessor_class}+{unit}"
            params.update({
                preprocessor_node_id: preprocessor_node
            })
        #Apply controlnet or revision
        if preprocessor == "Revision":
            clip_vision_encode_node = {
                "class_type": "CLIPVisionEncode",
                "inputs": {
                    "clip_vision": [
                        controlnetloader_node_id,
                        0
                    ],
                    "image": [
                        imageloader_node_id,
                        0
                    ]
                }
            }
            clip_vision_encode_node_id = f"{DEFAULT_NODE_IDS['CLIPVisionEncode']}+{unit}"
            unclip_conditioning_node = {
                "class_type": "unCLIPConditioning",
                "inputs": {
                    "strength": inputs["strength"],
                    "noise_augmentation": inputs["noise_augmentation"],
                    "conditioning": [
                        prev if prev != "" else f"{DEFAULT_NODE_IDS['ClipTextEncode_pos']}",
                        0
                    ],
                    "clip_vision_output": [
                        clip_vision_encode_node_id,
                        0
                    ]
                }
            }
            id = f"{DEFAULT_NODE_IDS['unCLIPConditioning']}+{unit}"
            params.update({
                clip_vision_encode_node_id: clip_vision_encode_node,
                id: unclip_conditioning_node
            })
            return id, prev_neg #pos conditioning id, neg conditioning id

        apply_controlnet_node = {
            "class_type": "ControlNetApplyAdvanced",
            "inputs": {
                "strength": self.cfg(f"controlnet{unit}_weight", float),
                "start_percent": self.cfg(f"controlnet{unit}_guidance_start", float),
                "end_percent": self.cfg(f"controlnet{unit}_guidance_end", float),
                "positive": [
                    prev if prev != "" else f"{DEFAULT_NODE_IDS['ClipTextEncode_pos']}",
                    0
                ],
                "negative": [
                    prev_neg if prev_neg != "" else f"{DEFAULT_NODE_IDS['ClipTextEncode_neg']}",
                    1 if prev_neg != "" else 0
                ],
                "control_net": [
                    controlnetloader_node_id,
                    0
                ],
                "image": [
                    imageloader_node_id if preprocessor == "None" else preprocessor_node_id,
                    0    
                ]
            }
        }

        id = f"{DEFAULT_NODE_IDS['ControlNetApplyAdvanced']}+{unit}"
        params.update({id: apply_controlnet_node})
        return id, id #pos conditioning id, neg conditioning id
    
    def set_img2img_batch(self, params, vae_encode_id):
        def insert_image_batch_node(batch_size):
            nonlocal counter
            if batch_size == 1:
                return DEFAULT_NODE_IDS["LoadBase64Image"]
            image_batch_node = {
                "class_type": "ImageBatch",
                "inputs": {
                    "image1": [
                        insert_image_batch_node(batch_size - floor(batch_size/2)), 0
                    ],
                    "image2": [
                        insert_image_batch_node(batch_size - floor(batch_size/2) - batch_size%2), 0
                    ]
                }
            }
            image_batch_node_id = f"{DEFAULT_NODE_IDS['ImageBatch']}+{counter}"
            params.update({image_batch_node_id: image_batch_node})
            counter += 1
            return image_batch_node_id

        if self.check_params(params, [vae_encode_id]):
            batch_size = self.cfg("sd_batch_size", int)
            if batch_size > 1:
                counter = 0
                params[vae_encode_id]["inputs"]["pixels"] = [insert_image_batch_node(batch_size), 0]

    def set_controlnet_preprocessor_and_model_list(self, obj):
        def set_preprocessors():
            preprocessors_info = {}

            for o in obj.values():
                if "preprocessors" in o["category"].lower():
                    name = o["display_name"].replace("Preprocessor", "") if o["display_name"] != "" else o["name"].replace("Preprocessor", "")
                    preprocessors_info.update({
                        name: {
                            "class": o["name"], 
                            "inputs": obj[o["name"]]["input"]["required"]
                        }
                    })
                # Add revision 
                if o["name"] == "unCLIPConditioning":
                    preprocessors_info.update({
                        "Revision": {
                            "class": o["name"], 
                            "inputs": obj[o["name"]]["input"]["required"]
                        }
                    })

            self.cfg.set("controlnet_preprocessor_list", ["None"] + list(preprocessors_info.keys()))
            self.cfg.set("controlnet_preprocessors_info", preprocessors_info)

        def set_models():
            model_list = obj["ControlNetLoader"]["input"]["required"]["control_net_name"][0] + obj["CLIPVisionLoader"]["input"]["required"]["clip_name"][0]
            self.cfg.set("controlnet_model_list", ["None"] + model_list)

        set_preprocessors()
        set_models()

    def get_config(self):
        def check_response(obj, keys):
            def on_success():
                self.is_connected = True
                self.status.emit(STATE_READY)
                self.config_updated.emit()

            try:
                assert obj is not None
                for key in keys:
                    assert key in obj
                on_success()
                return True
            except:
                self.status.emit(
                    f"{STATE_URLERROR}: incompatible response, are you running the right API?"
                )
                print("Invalid Response:\n", obj)
                return False

        def on_get_response(obj):            
            def get_upscalers():
                node = obj["LatentUpscale"]["input"]["required"]
                self.cfg.set("upscaler_methods_list", node["upscale_method"][0])

            def get_upscaler_models():
                node = obj["UpscaleModelLoader"]["input"]["required"]
                self.cfg.set("upscaler_model_list", node["model_name"][0])

            def get_sampler_data():
                node = obj["KSampler"]["input"]["required"]
                self.cfg.set("txt2img_sampler_list", node["sampler_name"][0])
                self.cfg.set("img2img_sampler_list", node["sampler_name"][0])
                self.cfg.set("inpaint_sampler_list", node["sampler_name"][0])
                self.cfg.set("txt2img_scheduler_list", node["scheduler"][0])
                self.cfg.set("img2img_scheduler_list", node["scheduler"][0])
                self.cfg.set("inpaint_scheduler_list", node["scheduler"][0])

            def get_models():
                node = obj["CheckpointLoaderSimple"]["input"]["required"]
                self.cfg.set("sd_model_list", node["ckpt_name"][0])

            def get_VAE():
                node = obj["VAELoader"]["input"]["required"]
                self.cfg.set("sd_vae_list", ["None"] + node["vae_name"][0])

            def get_loras():
                node = obj["LoraLoader"]["input"]["required"]
                self.cfg.set("sd_lora_list", node["lora_name"][0])

            if check_response(obj, ["LatentUpscale", "UpscaleModelLoader",
                        "KSampler", "CheckpointLoaderSimple", 
                        "VAELoader", "ControlNetLoader", "CLIPVisionLoader", "LoraLoader"]):
                get_upscalers()
                get_upscaler_models()
                get_sampler_data()
                get_models()
                get_loras()
                get_VAE()
                self.set_controlnet_preprocessor_and_model_list(obj)

        self.get("/object_info", on_get_response, ignore_no_connection=True)

    def get_controlnet_config(self):
        '''Get models and modules for ControlNet'''
        def check_response(obj, keys = []):
            try:
                assert obj is not None
                for key in keys:
                    assert key in obj
                return True
            except:
                self.status.emit(
                    f"{STATE_URLERROR}: incompatible response, are you running the right API?"
                )
                print("Invalid Response:\n", obj)
                return False
        
        def on_get_response(obj):
            if check_response(obj, ["ControlNetLoader", "CLIPVisionLoader"]):
                self.set_controlnet_preprocessor_and_model_list(obj)

        self.get("object_info", on_get_response, ignore_no_connection=True)

    def get_workflow(self, params, mode):
        image_data = {mode: {}}
        for id, node in params.items():
            if node["class_type"] in ["LoadBase64Image", "LoadBase64ImageMask"]:
                image_data[mode][id] = {}
                image_data[mode][id].update({
                    "image": node["inputs"]["image"],
                })
                node["inputs"]["image"] = PRUNED_DATA
        self.cfg.set("workflow_img_data", image_data)
        return params
    
    def run_injected_custom_workflow(self, workflow, seed, mode, src_img, mask_img = None, controlnet_src_imgs = {},
                                     resized_width = None, resized_height = None, original_width = None, original_height = None):
        params = self.restore_params(json.loads(workflow), src_img, mask_img)
        ksampler_id = DEFAULT_NODE_IDS["KSampler"]
        positive_prompt_id =  DEFAULT_NODE_IDS["ClipTextEncode_pos"]
        negative_prompt_id =  DEFAULT_NODE_IDS["ClipTextEncode_neg"]
        image_scale_id = DEFAULT_NODE_IDS["ImageScale"]
        latent_upscale_id = DEFAULT_NODE_IDS["LatentUpscale"]
        model_loader_id = DEFAULT_NODE_IDS["CheckpointLoaderSimple"]
        upscale_model_loader_id = DEFAULT_NODE_IDS["UpscaleModelLoader"]
        ksampler_found =  ksampler_id in params
        positive_prompt_found = positive_prompt_id in params
        negative_prompt_found = negative_prompt_id in params
        image_scale_found = image_scale_id in params
        latent_upscale_found = latent_upscale_id in params
        model_loader_found =  model_loader_id in params
        upscale_model_loader_found = upscale_model_loader_id in params
        prompt = self.cfg(f"{mode}_prompt", str)
        negative_prompt = self.cfg(f"{mode}_negative_prompt", str)
        loras_loaded = False

        def remove_lora_from_prompt():
            pattern = self.lora_re
            return re.sub(pattern, "", prompt)

        def load_placeholder_data():
            # Define a function that takes a match object and returns a replacement string
            def replace_lora_pattern(match, last_lora_id = None):
                # Get the group inside the parentheses
                group = match.group(1)
                # If last_lora_id is not None, return its value
                if last_lora_id is not None:
                    return last_lora_id
                # Otherwise, return the group value
                else:
                    return group
                
            nonlocal loras_loaded, params
            str_params = ""
            if PROMPT in workflow:
                str_params = json.dumps(params)
                str_params = str_params.replace(PROMPT, remove_lora_from_prompt())
            
            if NEGATIVE_PROMPT in workflow:
                if str_params == "":
                    str_params = json.dumps(params)
                str_params = str_params.replace(NEGATIVE_PROMPT, negative_prompt)
            
            # Bring back the dict to load the loras
            if str_params != "":
                params = json.loads(str_params)
                str_params = ""
            
            if model_loader_found and positive_prompt_found:
                if re.search(LAST_LOADED_LORA, workflow):
                    last_lora_id = self.loadLoRAs(params, mode, False)       
                    str_params = json.dumps(params)
                    str_params = re.sub(
                        LAST_LOADED_LORA, lambda match: replace_lora_pattern(match, last_lora_id), str_params
                    )
                    loras_loaded = True

            return json.loads(str_params) if str_params != "" else params

        if model_loader_found:
            params[model_loader_id]["inputs"]["ckpt_name"] = self.cfg("sd_model", str)

        if mode == "txt2img" and DEFAULT_NODE_IDS["EmptyLatentImage"] in params:
            empty_latent_image_id =  DEFAULT_NODE_IDS["EmptyLatentImage"]
            params[empty_latent_image_id]["inputs"]["height"] = resized_height
            params[empty_latent_image_id]["inputs"]["width"] = resized_width
            params[empty_latent_image_id]["inputs"]["batch_size"] = self.cfg("sd_batch_size", int)

        if mode == "img2img" or mode == "inpaint":
            VAEEncode_id = DEFAULT_NODE_IDS["VAEEncode"] if DEFAULT_NODE_IDS["VAEEncode"] in params else DEFAULT_NODE_IDS["VAEEncodeForInpaint"]
            self.set_img2img_batch(params, VAEEncode_id)

        if ksampler_found:
            ksampler_inputs = params[ksampler_id]["inputs"]
            if "seed" in ksampler_inputs:
                ksampler_inputs["seed"] = seed
            ksampler_inputs["steps"] = self.cfg(f"{mode}_steps", int)
            ksampler_inputs["cfg"] = self.cfg(f"{mode}_cfg_scale", float)
            if "denoise" in ksampler_inputs and mode != "txt2img":
                ksampler_inputs["denoise"] = self.cfg(f"{mode}_denoising_strength", float)
            ksampler_inputs["sampler_name"] = self.cfg(f"{mode}_sampler", str)
            ksampler_inputs["scheduler"] = self.cfg(f"{mode}_scheduler", str)

        params = load_placeholder_data()

        if positive_prompt_found:
            params[positive_prompt_id]["inputs"]["text"] = remove_lora_from_prompt()

        if negative_prompt_found:
            params[negative_prompt_id]["inputs"]["text"] = negative_prompt
        
        if not loras_loaded and model_loader_found:
            self.loadLoRAs(params, mode)
        
        if positive_prompt_found and negative_prompt_found:
            self.apply_controlnet(params, controlnet_src_imgs)

        if upscale_model_loader_found:
            params[upscale_model_loader_id]["inputs"]["model_name"] = self.cfg("upscaler_name", str)

        if image_scale_found:
            params[image_scale_id]["inputs"]["height"] = original_height
            params[image_scale_id]["inputs"]["width"] = original_width
        
        if latent_upscale_found:
            params[latent_upscale_id]["inputs"]["height"] = original_height
            params[latent_upscale_id]["inputs"]["width"] = original_width

        return params

    def post_txt2img(self, cb, width, height, src_img = None, controlnet_src_imgs: dict = {}):
        """Uses official API. Leave controlnet_src_imgs empty to not use controlnet."""
        seed = (
            # Qt casts int as 32-bit int
            int(self.cfg("txt2img_seed", str))
            if not self.cfg("txt2img_seed", str).strip() == ""
            else randint(0, 18446744073709552000)
        )
        resized_width, resized_height = width, height
        disable_base_and_max_size = self.cfg(
            "disable_sddebz_highres", bool)

        if not disable_base_and_max_size:
            resized_width, resized_height = calculate_resized_image_dimensions(
                self.cfg("sd_base_size", int), self.cfg(
                    "sd_max_size", int), width, height
            )

        if not self.cfg("txt2img_custom_workflow", bool):
            params = self.common_params()
            ksampler_node = {
                "class_type": "KSampler",
                "inputs": {
                    "cfg": 8,
                    "denoise": 1,
                    "model": [
                        DEFAULT_NODE_IDS["CheckpointLoaderSimple"],
                        0
                    ],
                    "latent_image": [
                        DEFAULT_NODE_IDS["EmptyLatentImage"],
                        0
                    ],
                    "negative": [
                        DEFAULT_NODE_IDS["ClipTextEncode_neg"],
                        0
                    ],
                    "positive": [
                        DEFAULT_NODE_IDS["ClipTextEncode_pos"],
                        0
                    ],
                    "sampler_name": self.cfg("txt2img_sampler", str),
                    "scheduler": self.cfg("txt2img_scheduler", str),
                    "seed": seed,
                    "steps": self.cfg("txt2img_steps", int)
                }
            }
            emptylatentimage_node = {
                "class_type": "EmptyLatentImage",
                "inputs": {
                    "batch_size": self.cfg("sd_batch_size", int),
                    "height": resized_height,
                    "width": resized_width
                }
            }
            cliptextencode_pos_node = {
                "class_type": "CLIPTextEncode",
                "inputs": {
                    "clip": [
                        DEFAULT_NODE_IDS["ClipSetLastLayer"],
                        0
                    ],
                    "text": self.cfg("txt2img_prompt", str),
                }
            }
            cliptextencode_neg_node = {
                "class_type": "CLIPTextEncode",
                "inputs": {
                    "clip": [
                        DEFAULT_NODE_IDS["ClipSetLastLayer"],
                        0
                    ],
                    "text": self.cfg("txt2img_negative_prompt", str),
                }
            }

            params.update({
                DEFAULT_NODE_IDS["KSampler"]: ksampler_node,
                DEFAULT_NODE_IDS["EmptyLatentImage"]: emptylatentimage_node,
                DEFAULT_NODE_IDS["ClipTextEncode_pos"]: cliptextencode_pos_node,
                DEFAULT_NODE_IDS["ClipTextEncode_neg"]: cliptextencode_neg_node
            })

            upscaler_name = self.cfg("upscaler_name", str)
            if not disable_base_and_max_size and not upscaler_name == "None" and\
                (min(width, height) > self.cfg("sd_base_size", int)
                    or max(width, height) > self.cfg("sd_max_size", int)):
                upscaler_name = self.cfg("upscaler_name", str)
                if upscaler_name in self.cfg("upscaler_model_list", "QStringList"):
                    self.upscale_with_model(
                        params, width, height, seed, "txt2img")
                else:
                    self.upscale_latent(params, width, height, seed, "txt2img")

            self.loadLoRAs(params, "txt2img")
            self.apply_controlnet(params, controlnet_src_imgs)
        else:
            workflow = self.cfg("txt2img_workflow", str)
            params = self.run_injected_custom_workflow(
                workflow, seed, "txt2img", src_img, None, controlnet_src_imgs, resized_width, resized_height, width, height
            )

        if cb is None:
            return self.get_workflow(params, "txt2img")
        
        self.get_images(params, cb)

    def post_img2img(self, cb, src_img, width, height, controlnet_src_imgs: dict = {}):
        """Leave controlnet_src_imgs empty to not use controlnet."""
        seed = (
            # Qt casts int as 32-bit int
            int(self.cfg("img2img_seed", str))
            if not self.cfg("img2img_seed", str).strip() == ""
            else randint(0, 18446744073709552000)
        )
        resized_width, resized_height = width, height
        disable_base_and_max_size = self.cfg(
            "disable_sddebz_highres", bool)

        if not disable_base_and_max_size:
            resized_width, resized_height = calculate_resized_image_dimensions(
                self.cfg("sd_base_size", int), self.cfg(
                    "sd_max_size", int), width, height
            )

        src_img = src_img.scaled(resized_width, resized_height, Qt.KeepAspectRatio)

        if not self.cfg("img2img_custom_workflow", bool):
            params = self.common_params()
            loadimage_node = {
                "class_type": "LoadBase64Image",
                "inputs": {
                    "image": img_to_b64(src_img)
                }
            }
            vae_id = DEFAULT_NODE_IDS["VAELoader"]
            vaeencode_node = {
                "class_type": "VAEEncode",
                "inputs": {
                    "pixels": [
                        DEFAULT_NODE_IDS["LoadBase64Image"],
                        0
                    ],
                    "vae": [
                        vae_id if vae_id in params else DEFAULT_NODE_IDS["CheckpointLoaderSimple"],
                        0 if vae_id in params else 2
                    ]
                }
            }
            ksampler_node = {
                "class_type": "KSampler",
                "inputs": {
                    "cfg": 8,
                    "denoise": self.cfg("img2img_denoising_strength", float),
                    "model": [
                        DEFAULT_NODE_IDS["CheckpointLoaderSimple"],
                        0
                    ],
                    "latent_image": [
                        DEFAULT_NODE_IDS["VAEEncode"],
                        0
                    ],
                    "negative": [
                        DEFAULT_NODE_IDS["ClipTextEncode_neg"],
                        0
                    ],
                    "positive": [
                        DEFAULT_NODE_IDS["ClipTextEncode_pos"],
                        0
                    ],
                    "sampler_name": self.cfg("img2img_sampler", str),
                    "scheduler": self.cfg("img2img_scheduler", str),
                    "seed": seed,
                    "steps": self.cfg("img2img_steps", int)
                }
            }
            cliptextencode_pos_node = {
                "class_type": "CLIPTextEncode",
                "inputs": {
                    "clip": [
                        DEFAULT_NODE_IDS["ClipSetLastLayer"],
                        0
                    ],
                    "text": self.cfg("img2img_prompt", str),
                }
            }
            cliptextencode_neg_node = {
                "class_type": "CLIPTextEncode",
                "inputs": {
                    "clip": [
                        DEFAULT_NODE_IDS["ClipSetLastLayer"],
                        0
                    ],
                    "text": self.cfg("img2img_negative_prompt", str),
                }
            }

            params.update({
                DEFAULT_NODE_IDS["KSampler"]: ksampler_node,
                DEFAULT_NODE_IDS["LoadBase64Image"]: loadimage_node,
                DEFAULT_NODE_IDS["VAEEncode"]: vaeencode_node,
                DEFAULT_NODE_IDS["ClipTextEncode_pos"]: cliptextencode_pos_node,
                DEFAULT_NODE_IDS["ClipTextEncode_neg"]: cliptextencode_neg_node
            })

            self.set_img2img_batch(params, DEFAULT_NODE_IDS["VAEEncode"])

            upscaler_name = self.cfg("upscaler_name", str)
            if not disable_base_and_max_size and not upscaler_name == "None" and\
                (min(width, height) > self.cfg("sd_base_size", int)
                    or max(width, height) > self.cfg("sd_max_size", int)):
                if upscaler_name in self.cfg("upscaler_model_list", "QStringList"):
                    self.upscale_with_model(
                        params, width, height, seed, "img2img")
                else:
                    self.upscale_latent(params, width, height, seed, "img2img")
            
            self.loadLoRAs(params, "img2img")
            self.apply_controlnet(params, controlnet_src_imgs)
        else:
            params = self.run_injected_custom_workflow(
                self.cfg("img2img_workflow", str), seed, "img2img", src_img, None, 
                controlnet_src_imgs, resized_width, resized_height, width, height
            )

        if cb is None:
            return self.get_workflow(params, "img2img")
        
        self.get_images(params, cb)

    def post_inpaint(self, cb, src_img, mask_img, width, height, controlnet_src_imgs: dict = {}):
        """Leave controlnet_src_imgs empty to not use controlnet."""
        assert mask_img, "Inpaint layer is needed for inpainting!"
        seed = (
            # Qt casts int as 32-bit int
            int(self.cfg("inpaint_seed", str))
            if not self.cfg("inpaint_seed", str).strip() == ""
            else randint(0, 18446744073709552000)
        )
        resized_width, resized_height = width, height
        disable_base_and_max_size = self.cfg(
            "disable_sddebz_highres", bool)

        if not disable_base_and_max_size:
            resized_width, resized_height = calculate_resized_image_dimensions(
                self.cfg("sd_base_size", int), self.cfg(
                    "sd_max_size", int), width, height
            )
            src_img = src_img.scaled(resized_width, resized_height, Qt.KeepAspectRatio)
            mask_img = mask_img.scaled(resized_width, resized_height, Qt.KeepAspectRatio)

        if not self.cfg("inpaint_custom_workflow", bool):
            preserve = self.cfg("inpaint_fill", str) == "preserve"

            params = self.common_params()
            loadimage_node = {
                "class_type": "LoadBase64Image",
                "inputs": {
                    "image": img_to_b64(src_img)
                }
            }
            loadmask_node = {
                "class_type": "LoadBase64ImageMask",
                "inputs": {
                    "image": img_to_b64(mask_img),
                    "channel": "alpha"
                }
            }
            vae_id = DEFAULT_NODE_IDS["VAELoader"]
            if preserve:
                vaeencode_node = {
                    "class_type": "VAEEncode",
                    "inputs": {
                        "pixels": [
                            DEFAULT_NODE_IDS["LoadBase64Image"],
                            0
                        ],
                        "vae": [
                            vae_id if vae_id in params else DEFAULT_NODE_IDS["CheckpointLoaderSimple"],
                            0 if vae_id in params else 2
                        ]
                    }
                }
                setlatentnoisemask_node = {
                    "class_type": "SetLatentNoiseMask",
                    "inputs": {
                        "samples": [
                            DEFAULT_NODE_IDS["VAEEncode"],
                            0
                        ],
                        "mask": [
                            DEFAULT_NODE_IDS["LoadBase64ImageMask"],
                            0
                        ]
                    }
                }
                params.update({
                    DEFAULT_NODE_IDS["SetLatentNoiseMask"]: setlatentnoisemask_node
                })
            else:
                vaeencode_node = {
                    "class_type": "VAEEncodeForInpaint",
                    "inputs": {
                        "grow_mask_by": 6,
                        "pixels": [
                            DEFAULT_NODE_IDS["LoadBase64Image"],
                            0
                        ],
                        "mask": [
                            DEFAULT_NODE_IDS["LoadBase64ImageMask"],
                            0
                        ],
                        "vae": [
                            vae_id if vae_id in params else DEFAULT_NODE_IDS["CheckpointLoaderSimple"],
                            0 if vae_id in params else 2
                        ]
                    }
                }
            ksampler_node = {
                "class_type": "KSampler",
                "inputs": {
                    "cfg": 8,
                    "denoise": self.cfg("inpaint_denoising_strength", float),
                    "model": [
                        DEFAULT_NODE_IDS["CheckpointLoaderSimple"],
                        0
                    ],
                    "latent_image": [
                        DEFAULT_NODE_IDS["SetLatentNoiseMask"] if preserve else DEFAULT_NODE_IDS["VAEEncodeForInpaint"],
                        0
                    ],
                    "negative": [
                        DEFAULT_NODE_IDS["ClipTextEncode_neg"],
                        0
                    ],
                    "positive": [
                        DEFAULT_NODE_IDS["ClipTextEncode_pos"],
                        0
                    ],
                    "sampler_name": self.cfg("inpaint_sampler", str),
                    "scheduler": self.cfg("inpaint_scheduler", str),
                    "seed": seed,
                    "steps": self.cfg("inpaint_steps", int)
                }
            }
            cliptextencode_pos_node = {
                "class_type": "CLIPTextEncode",
                "inputs": {
                    "clip": [
                        DEFAULT_NODE_IDS["ClipSetLastLayer"],
                        0
                    ],
                    "text": self.cfg("inpaint_prompt", str),
                }
            }
            cliptextencode_neg_node = {
                "class_type": "CLIPTextEncode",
                "inputs": {
                    "clip": [
                        DEFAULT_NODE_IDS["ClipSetLastLayer"],
                        0
                    ],
                    "text": self.cfg("inpaint_negative_prompt", str),
                }
            }
            VAEEncode_id = DEFAULT_NODE_IDS["VAEEncode"] if preserve else DEFAULT_NODE_IDS["VAEEncodeForInpaint"]
            params.update({
                DEFAULT_NODE_IDS["KSampler"]: ksampler_node,
                DEFAULT_NODE_IDS["LoadBase64Image"]: loadimage_node,
                DEFAULT_NODE_IDS["LoadBase64ImageMask"]: loadmask_node,
                VAEEncode_id: vaeencode_node,
                DEFAULT_NODE_IDS["ClipTextEncode_pos"]: cliptextencode_pos_node,
                DEFAULT_NODE_IDS["ClipTextEncode_neg"]: cliptextencode_neg_node
            })

            self.set_img2img_batch(params, VAEEncode_id)

            upscaler_name = self.cfg("upscaler_name", str)
            if not disable_base_and_max_size and not upscaler_name == "None" and\
                (min(width, height) > self.cfg("sd_base_size", int)
                    or max(width, height) > self.cfg("sd_max_size", int)):  # Upscale image automatically.
                if upscaler_name in self.cfg("upscaler_model_list", "QStringList"):
                    # Set common upscaling nodes for model upscaling (eg. Ultrasharp).
                    self.upscale_with_model(
                        params, width, height, seed, "inpaint")
                    # Set a new latent noise mask for the second pass.
                    setlatentnoisemask_node = {
                        "class_type": "SetLatentNoiseMask",
                        "inputs": {
                            "samples": [
                                DEFAULT_NODE_IDS["VAEEncode_upscale"],
                                0
                            ],
                            "mask": [
                                DEFAULT_NODE_IDS["LoadBase64ImageMask"],
                                0
                            ]
                        }
                    }
                else:
                    self.upscale_latent(params, width, height, seed, "inpaint")
                    # Set a new latent noise mask for the second pass.
                    setlatentnoisemask_node = {
                        "class_type": "SetLatentNoiseMask",
                        "inputs": {
                            "samples": [
                                DEFAULT_NODE_IDS["LatentUpscale"],
                                0
                            ],
                            "mask": [
                                DEFAULT_NODE_IDS["LoadBase64ImageMask"],
                                0
                            ]
                        }
                    }
                # Register the second mask.
                params.update({
                    DEFAULT_NODE_IDS["SetLatentNoiseMask_upscale"]: setlatentnoisemask_node
                })
                # Connect second KSampler pass to the mask.
                params[DEFAULT_NODE_IDS["KSampler_upscale"]]["inputs"]["latent_image"] = [
                    DEFAULT_NODE_IDS["SetLatentNoiseMask_upscale"],
                    0
                ]
            self.loadLoRAs(params, "inpaint")
            self.apply_controlnet(params, controlnet_src_imgs)
        else:
            params = self.run_injected_custom_workflow(self.cfg("inpaint_workflow", str), seed,
                                     "inpaint", src_img, mask_img, controlnet_src_imgs, original_width=width,
                                     original_height=height)

        if cb is None:
            return self.get_workflow(params, "inpaint")

        self.get_images(params, cb)

    def post_upscale(self, cb, src_img):
        params = {}
        if not self.cfg("upscale_custom_workflow", bool):
            def upscale_latent():
                imagescale_node = {
                    "class_type": "ImageScaleBy",
                    "inputs": {
                        "upscale_method": self.cfg("upscale_upscaler_name", str),
                        "scale_by": self.cfg("upscale_upscale_by", float),
                        "image": [
                            DEFAULT_NODE_IDS["LoadBase64Image"],
                            0
                        ]
                    }
                }
                saveimage_node = {
                    "class_type": "SaveImage",
                    "inputs": {
                        "filename_prefix": "ComfyUI",
                        "images": [
                            DEFAULT_NODE_IDS["ImageScaleBy"],
                            0
                        ]
                    }
                }
                params.update({
                    DEFAULT_NODE_IDS["ImageScaleBy"]: imagescale_node,
                    DEFAULT_NODE_IDS["SaveImage"]: saveimage_node
                })

            def upscale_with_model():
                upscalemodelloader_node = {
                    "class_type": "UpscaleModelLoader",
                    "inputs": {
                        "model_name": self.cfg("upscale_upscaler_name", str)
                    }
                }
                imageupscalewithmodel_node = {
                    "class_type": "ImageUpscaleWithModel",
                    "inputs": {
                        "upscale_model": [
                            DEFAULT_NODE_IDS["UpscaleModelLoader"],
                            0
                        ],
                        "image": [
                            DEFAULT_NODE_IDS["LoadBase64Image"],
                            0
                        ]
                    }
                }
                imagescale_node = {
                    "class_type": "ImageScale",
                    "inputs": {
                        "upscale_method": "bilinear",
                        "width": src_img.width() * self.cfg("upscale_upscale_by", float),
                        "height": src_img.height() * self.cfg("upscale_upscale_by", float),
                        "crop": "disabled",
                        "image": [
                            DEFAULT_NODE_IDS["ImageUpscaleWithModel"],
                            0
                        ]
                    }
                }
                saveimage_node = {
                    "class_type": "SaveImage",
                    "inputs": {
                        "filename_prefix": "ComfyUI",
                        "images": [
                            DEFAULT_NODE_IDS["ImageScale"],
                            0
                        ]
                    }
                }
                params.update({
                    DEFAULT_NODE_IDS["UpscaleModelLoader"]: upscalemodelloader_node,
                    DEFAULT_NODE_IDS["ImageUpscaleWithModel"]: imageupscalewithmodel_node,
                    DEFAULT_NODE_IDS["ImageScale"]: imagescale_node,
                    DEFAULT_NODE_IDS["SaveImage"]: saveimage_node,
                })

            loadimage_node = {
                "class_type": "LoadBase64Image",
                "inputs": {
                    "image": img_to_b64(src_img)
                }
            }

            params = {
                DEFAULT_NODE_IDS["LoadBase64Image"]: loadimage_node
            }

            if self.cfg("upscale_upscaler_name", str) in self.cfg("upscaler_model_list", "QStringList"):
                upscale_with_model()
            else:
                upscale_latent()
        else:
            params = self.run_injected_custom_workflow(self.cfg("upscale_workflow", str), 0, "upscale", src_img)

        if cb is None:
            return self.get_workflow(params, "upscale")

        self.get_images(params, cb)

    def post_controlnet_preview(self, cb, src_img):
        unit = self.cfg("controlnet_unit", int)
        preprocessor = self.cfg(f"controlnet{unit}_preprocessor", str)
        try:
            assert "Inpaint" not in preprocessor
            assert "Reference" not in preprocessor
        except:
            self.status.emit(
                    "Preprocessor not supported for preview."
                )
            return

        loadimage_node = {
            "class_type": "LoadBase64Image",
            "inputs": {
                "image": img_to_b64(src_img)
            }
        }
        inputs = self.cfg(f"controlnet{unit}_inputs")
        inputs.update({"image": [
            DEFAULT_NODE_IDS['ControlNetImageLoader'], 0
        ]})
        preprocessor_class = self.cfg("controlnet_preprocessors_info", dict)[preprocessor]["class"]
        preprocessor_node = {
            "class_type": preprocessor_class,
            "inputs": inputs
        }
        saveimage_node = {
            "class_type": "SaveImage",
            "inputs": {
                "filename_prefix": "ComfyUI",
                "images": [
                    preprocessor_class,
                    0
                ]
            }
        }

        params = {
            DEFAULT_NODE_IDS['ControlNetImageLoader']: loadimage_node,
            preprocessor_class: preprocessor_node,
            DEFAULT_NODE_IDS['SaveImage']: saveimage_node
        }

        self.get_images(params, cb)

    def restore_params(self, params, src_img, mask=None):
        workflow_image_data = None
        if mask == None:
            mask = src_img

        mode = self.cfg("workflow_to", str)
        if mode in self.cfg("workflow_img_data", dict):
            workflow_image_data = self.cfg("workflow_img_data", dict)[mode]
        for node_id, node in params.items():
            if node["class_type"] in ["LoadBase64Image", "LoadBase64ImageMask"]:
                for input_key, input_value in node["inputs"].items():
                    if workflow_image_data is not None and input_value == PRUNED_DATA and \
                          node_id in workflow_image_data:
                        image_data = workflow_image_data[node_id]
                        node["inputs"][input_key] = image_data[input_key]
                    elif input_value == SELECTED_IMAGE:
                        # Replace the placeholder with the actual image data
                        node["inputs"][input_key] = img_to_b64(src_img)
                    elif input_value == CURRENT_LAYER_AS_MASK:
                        node["inputs"][input_key] = img_to_b64(mask)
        return params

    def run_workflow(self, workflow, src_img, mask, cb=None):
        params = self.restore_params(json.loads(workflow), src_img, mask)
        self.get_images(params, cb)

    def post_interrupt(self, cb):
        self.post("interrupt", {}, cb, is_long=False)

    # def get_progress(self, cb):
     #    self.get("progress", cb)