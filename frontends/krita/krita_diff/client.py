import json
import socket
import uuid
from math import ceil
from random import randint
from typing import Any
from urllib.error import URLError
from urllib.parse import urljoin, urlparse, urlencode
from urllib.request import Request, urlopen

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
    ERR_BAD_URL,
    ERR_NO_CONNECTION,
    LONG_TIMEOUT,
    OFFICIAL_ROUTE_PREFIX,
    ROUTE_PREFIX,
    CONTROLNET_ROUTE_PREFIX,
    SHORT_TIMEOUT,
    STATE_DONE,
    STATE_READY,
    STATE_URLERROR,
    THREADED,
    DEFAULT_NODE_IDS
)
from .utils import (
    bytewise_xor,
    img_to_b64,
    calculate_resized_image_dimensions
)

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

    def run(self):
        try:
            url = self.url
            if self.method == "GET":
                url = url if self.data is None else f"{self.url}?{urlencode(self.data)}"
            req = Request(url, headers=self.headers, method=self.method)
            with urlopen(req, self.data if self.method == "POST" else None, self.timeout) as res:
                data = res.read()
                enc_type = res.getheader("X-Encrypted-Body", None)
                assert enc_type in {"XOR", None}, "Unknown server encryption!"
                if enc_type == "XOR":
                    assert self.key, f"Key needed to decrypt server response!"
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

    def on_images_received(self, status, prompt_id):
        def craft_response(images, history, names):
            try:
                return {
                    "info": {
                        "prompt": history["prompt"][2][DEFAULT_NODE_IDS["ClipTextEncode_pos"]]["inputs"]["text"],
                        "negative_prompt": history["prompt"][2][DEFAULT_NODE_IDS["ClipTextEncode_neg"]]["inputs"]["text"],
                        "sd_model": history["prompt"][2][DEFAULT_NODE_IDS["CheckpointLoaderSimple"]]["inputs"]["ckpt_name"],
                        "sampler_name": history["prompt"][2][DEFAULT_NODE_IDS["KSampler"]]["inputs"]["sampler_name"],
                        "cfg_scale": history["prompt"][2][DEFAULT_NODE_IDS["KSampler"]]["inputs"]["cfg"],
                        "steps": history["prompt"][2][DEFAULT_NODE_IDS["KSampler"]]["inputs"]["steps"],
                        "all_names": names
                    },
                    "outputs": images
                }
            except KeyError:
                return {
                    "info": {},
                    # Assuming it's the result of a simple upscale.
                    "outputs": images[0]
                }

        def on_history_received(history_res):
            images_output = []

            def on_image_received(img):
                assert img is not None, "Backend Error, check terminal"
                qimage = QImage()
                qimage.loadFromData(img)
                images_output.append(img_to_b64(qimage))

            assert history_res is not None, "Backend Error, check terminal"

            history = history_res[prompt_id]
            names = []
            i = 0
            for node_id in history['outputs']:
                node_output = history['outputs'][node_id]
                if 'images' in node_output:
                    for image in node_output['images']:
                        i += 1
                        self.get_image(
                            image['filename'], image['subfolder'], image['type'], on_image_received)
                        names.append(image["filename"])

            # Check if all images are in the list before sending the response to script.
            def check_if_populated():
                if len(images_output) == i:
                    timer.stop()
                    response = craft_response(images_output, history, names)
                    self.images_received.emit(response)

            if i != 0:
                timer = QTimer()
                timer.timeout.connect(check_if_populated)
                timer.start(0.05)

        if status == STATE_DONE:
            # Prevent undesired executions of this function.
            self.status.disconnect(self.conn)
            self.get_history(prompt_id, on_history_received)

    def queue_prompt(self, prompt, cb=None):
        p = {"prompt": prompt, "client_id": self.client_id}
        self.post("prompt", p, cb, is_long=False)

    def get_image(self, filename, subfolder, folder_type, cb=None):
        data = {"filename": filename,
                "subfolder": subfolder, "type": folder_type}
        self.get("view", cb, data=data, is_long=False)

    def get_history(self, prompt_id, cb=None):
        self.get(f"history/{prompt_id}", cb, is_long=False)

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
            self.conn = lambda s: self.on_images_received(s, prompt_id)
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

    def common_params(self, width, height, controlnet_src_imgs):
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

        if self.cfg("sd_vae", str) != "Normal"  \
                and self.cfg("sd_vae", str) in self.cfg("sd_vae_list", "QStringList"):
            loadVAE()

        return params

        if controlnet_src_imgs:
            controlnet_units_param = list()

            for i in range(len(self.cfg("controlnet_unit_list", "QStringList"))):
                if self.cfg(f"controlnet{i}_enable", bool):
                    controlnet_units_param.append(
                        self.controlnet_unit_params(img_to_b64(
                            controlnet_src_imgs[str(i)]), i, width, height)
                    )
                else:
                    controlnet_units_param.append({"enabled": False})

            params["alwayson_scripts"].update({
                "controlnet": {
                    "args": controlnet_units_param
                }
            })

        return params

    def loadLoRAs(self, params, prefix):
        '''Call only when base prompt structure is already stored in params,
        otherwise will probably not work'''

        # Initialize a counter to keep track of the number of nodes added
        lora_count = 0

        # Use a regular expression to find all the elements between < and > in the string
        import re
        pattern = r"<lora:(\w+):([\d.]+)>"
        matches = re.findall(pattern, self.cfg(f"{prefix}_prompt", str))

        # Loop through the matches and create a node for each element
        for match in matches:
            # Extract the lora name and the strength number from the match
            lora_name = match[0]
            strength_number = float(match[1])

            # Create a node dictionary with the class type, inputs, and outputs
            node = {
                "class_type": "LoraLoader",
                "inputs": {
                    "lora_name": f"{lora_name}.safetensors",
                    "strength_model": strength_number,
                    "strength_clip": strength_number,
                    # If this is the first node, use the default model and clip parameters
                    # Otherwise, use the previous node id as the model and clip parameters
                    "model": [
                        DEFAULT_NODE_IDS[
                            "CheckpointLoaderSimple"] if lora_count == 0 else f"{DEFAULT_NODE_IDS['LoraLoader']}+{lora_count}",
                        0
                    ],
                    "clip": [
                        DEFAULT_NODE_IDS[
                            "ClipSetLastLayer"] if lora_count == 0 else f"{DEFAULT_NODE_IDS['LoraLoader']}+{lora_count}",
                        0 if lora_count == 0 else 1
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
            #Connect KSampler to last lora node.
            params[DEFAULT_NODE_IDS["KSampler"]]["inputs"]["model"] = [
                f"{DEFAULT_NODE_IDS['LoraLoader']}+{lora_count}",
                0
            ]

            #Connect KSampler for upscale (second pass) to last lora node if found.
            if DEFAULT_NODE_IDS["KSampler_upscale"] in params:
                params[DEFAULT_NODE_IDS["KSampler_upscale"]]["inputs"]["model"] = [
                    f"{DEFAULT_NODE_IDS['LoraLoader']}+{lora_count}",
                    0
                ]
            
            #Connect positive prompt to lora clip.
            params[DEFAULT_NODE_IDS["ClipTextEncode_pos"]]["inputs"]["clip"] = [
                f"{DEFAULT_NODE_IDS['LoraLoader']}+{lora_count}",
                1
            ]
            
            #Connect negative prompt to lora clip.
            params[DEFAULT_NODE_IDS["ClipTextEncode_neg"]]["inputs"]["clip"] = [
                f"{DEFAULT_NODE_IDS['LoraLoader']}+{lora_count}",
                1
            ]

    def upscale_latent(self, params, width, height, seed, cfg_prefix):
        '''Call only when base prompt structure is already stored in params,
        otherwise will probably not work'''
        latentupscale_node = {
            "class_type": "LatentUpscale",
            "inputs": {
                "upscale_method": self.cfg("upscaler_name", str),
                "width": width,
                "height": height,
                "crop": "disabled",
                "samples": [
                        DEFAULT_NODE_IDS["KSampler"],
                        0
                ]
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
        params[DEFAULT_NODE_IDS["VAEDecode"]]["inputs"]["samples"] = [
            DEFAULT_NODE_IDS["KSampler_upscale"],
            0
        ]
        params[DEFAULT_NODE_IDS["KSampler"]]["inputs"]["steps"] = ceil(
            self.cfg("inpaint_steps", int)/2)

    def upscale_with_model(self, params, width, height, seed, cfg_prefix):
        '''Call only when base prompt structure is already stored in params,
        otherwise will probably not work'''
        vae_id = DEFAULT_NODE_IDS["VAELoader"]
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
                "model_name": self.cfg(f"upscaler_name", str)
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
                "sampler_name": self.cfg(f"{cfg_prefix}_sampler", str),
                "scheduler": self.cfg(f"{cfg_prefix}_scheduler", str),
                "seed": seed,
                "steps": ceil(self.cfg(f"{cfg_prefix}_steps", int)/2)
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
        params[DEFAULT_NODE_IDS["VAEDecode"]]["inputs"]["samples"] = [
            DEFAULT_NODE_IDS["KSampler_upscale"],
            0
        ]
        params[DEFAULT_NODE_IDS["KSampler"]]["inputs"]["steps"] = ceil(
            self.cfg("inpaint_steps", int)/2)

    def controlnet_unit_params(self, image: str, unit: int, width: int, height: int):
        preprocessor_resolution = min(width, height) if self.cfg(f"controlnet{unit}_pixel_perfect", bool)  \
            else self.cfg(f"controlnet{unit}_preprocessor_resolution", int)
        params = dict(
            input_image=image,
            module=self.cfg(f"controlnet{unit}_preprocessor", str),
            model=self.cfg(f"controlnet{unit}_model", str),
            weight=self.cfg(f"controlnet{unit}_weight", float),
            lowvram=self.cfg(f"controlnet{unit}_low_vram", bool),
            processor_res=preprocessor_resolution,
            threshold_a=self.cfg(f"controlnet{unit}_threshold_a", float),
            threshold_b=self.cfg(f"controlnet{unit}_threshold_b", float),
            guidance_start=self.cfg(f"controlnet{unit}_guidance_start", float),
            guidance_end=self.cfg(f"controlnet{unit}_guidance_end", float),
            control_mode=self.cfg(f"controlnet{unit}_control_mode", str)
        )
        return params

    def get_config(self):
        def check_response(obj):
            def on_success():
                self.is_connected = True
                self.status.emit(STATE_READY)
                self.config_updated.emit()

            try:
                assert obj is not None
                on_success()
            except:
                self.status.emit(
                    f"{STATE_URLERROR}: incompatible response, are you running the right API?"
                )
                print("Invalid Response:\n", obj)
                return

        def on_get_upscalers(obj):  # Can only get latent upscalers for now.
            try:
                check_response(obj)
            except:
                return

            if "LatentUpscale" in obj:
                node = obj["LatentUpscale"]["input"]["required"]
                self.cfg.set("upscaler_methods_list",
                             node["upscale_method"][0])
            if "UpscaleModelLoader" in obj:
                node = obj["UpscaleModelLoader"]["input"]["required"]
                self.cfg.set("upscaler_model_list", node["model_name"][0])

        def on_get_sampler_data(obj):
            try:
                check_response(obj)
            except:
                return

            node = obj["KSampler"]["input"]["required"]
            self.cfg.set("txt2img_sampler_list", node["sampler_name"][0])
            self.cfg.set("img2img_sampler_list", node["sampler_name"][0])
            self.cfg.set("inpaint_sampler_list", node["sampler_name"][0])
            self.cfg.set("txt2img_scheduler_list", node["scheduler"][0])
            self.cfg.set("img2img_scheduler_list", node["scheduler"][0])
            self.cfg.set("inpaint_scheduler_list", node["scheduler"][0])

        def on_get_models(obj):
            try:
                check_response(obj)
            except:
                return

            node = obj["CheckpointLoaderSimple"]["input"]["required"]
            self.cfg.set("sd_model_list", node["ckpt_name"][0])

        def on_get_VAE(obj):
            try:
                check_response(obj)
            except:
                return

            node = obj["VAELoader"]["input"]["required"]
            self.cfg.set("sd_vae_list", ["None"] + node["vae_name"][0])

        self.get(f"/object_info/LatentUpscale",
                 on_get_upscalers, ignore_no_connection=True)
        self.get(f"/object_info/KSampler", on_get_sampler_data,
                 ignore_no_connection=True)
        self.get(f"/object_info/CheckpointLoaderSimple",
                 on_get_models, ignore_no_connection=True)
        self.get(f"/object_info/UpscaleModelLoader",
                 on_get_upscalers, ignore_no_connection=True)
        self.get(f"/object_info/VAELoader",
                 on_get_VAE, ignore_no_connection=True)

    def get_controlnet_config(self):
        '''Get models and modules for ControlNet'''
        def check_response(obj, key: str):
            try:
                assert key in obj
            except:
                self.status.emit(
                    f"{STATE_URLERROR}: incompatible response, are you running the right API?"
                )
                print("Invalid Response:\n", obj)
                return

        def set_model_list(obj):
            key = "model_list"
            check_response(obj, key)
            self.cfg.set("controlnet_model_list", ["None"] + obj[key])

        def set_preprocessor_list(obj):
            key = "module_list"
            check_response(obj, key)
            self.cfg.set("controlnet_preprocessor_list", obj[key])

        # Get controlnet API url
        url = get_url(self.cfg, prefix=CONTROLNET_ROUTE_PREFIX)
        self.get("model_list", set_model_list, base_url=url)
        self.get("module_list", set_preprocessor_list, base_url=url)

    def post_txt2img(self, cb, width, height, controlnet_src_imgs: dict = {}):
        """Uses official API. Leave controlnet_src_imgs empty to not use controlnet."""
        if not self.cfg("just_use_yaml", bool):
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

            params = self.common_params(
                resized_width, resized_height, controlnet_src_imgs)
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
        self.get_images(params, cb)

    def post_img2img(self, cb, src_img, width, height, controlnet_src_imgs: dict = {}):
        """Leave controlnet_src_imgs empty to not use controlnet."""
        if not self.cfg("just_use_yaml", bool):
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

            params = self.common_params(
                resized_width, resized_height, controlnet_src_imgs)
            loadimage_node = {
                "class_type": "LoadBase64Image",
                "inputs": {
                    "image": img_to_b64(src_img.scaled(resized_width, resized_height, Qt.KeepAspectRatio))
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
        self.get_images(params, cb)

    def post_inpaint(self, cb, src_img, mask_img, width, height, controlnet_src_imgs: dict = {}):
        """Leave controlnet_src_imgs empty to not use controlnet."""
        assert mask_img, "Inpaint layer is needed for inpainting!"
        preserve = self.cfg("inpaint_fill", str) == "preserve"
        if not self.cfg("just_use_yaml", bool):
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

            params = self.common_params(
                resized_width, resized_height, controlnet_src_imgs)
            loadimage_node = {
                "class_type": "LoadBase64Image",
                "inputs": {
                    "image": img_to_b64(src_img.scaled(resized_width, resized_height, Qt.KeepAspectRatio))
                }
            }
            loadmask_node = {
                "class_type": "LoadBase64ImageMask",
                "inputs": {
                    "image": img_to_b64(mask_img.scaled(resized_width, resized_height, Qt.KeepAspectRatio)),
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
        self.get_images(params, cb)

    def post_upscale(self, cb, src_img):
        params = {}

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

        self.get_images(params, cb)

    def post_controlnet_preview(self, cb, src_img, width, height):
        def get_pixel_perfect_preprocessor_resolution():
            if self.cfg("disable_sddebz_highres", bool):
                return min(width, height)

            resized_width, resized_height = calculate_resized_image_dimensions(
                self.cfg("sd_base_size", int), self.cfg(
                    "sd_max_size", int), width, height
            )
            return min(resized_width, resized_height)

        unit = self.cfg("controlnet_unit", str)
        preprocessor_resolution = get_pixel_perfect_preprocessor_resolution() if self.cfg(f"controlnet{unit}_pixel_perfect", bool)  \
            else self.cfg(f"controlnet{unit}_preprocessor_resolution", int)

        params = (
            {
                "controlnet_module": self.cfg(f"controlnet{unit}_preprocessor", str),
                "controlnet_input_images": [img_to_b64(src_img)],
                "controlnet_processor_res": preprocessor_resolution,
                "controlnet_threshold_a": self.cfg(f"controlnet{unit}_threshold_a", float),
                "controlnet_threshold_b": self.cfg(f"controlnet{unit}_threshold_b", float)
            }  # Not sure if it's necessary to make the just_use_yaml validation here
        )
        url = get_url(self.cfg, prefix=CONTROLNET_ROUTE_PREFIX)
        self.post("detect", params, cb, url)

    def post_interrupt(self, cb):
        self.post("interrupt", {}, cb, is_long=False)

    # def get_progress(self, cb):
     #    self.get("progress", cb)
