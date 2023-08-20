from dataclasses import dataclass, field
from typing import List, Dict

# set combo box to error msg instead of blank when cannot retrieve options from backend
ERROR_MSG = "Retrieval Failed"

# Used for status bar
STATE_READY = "Ready"
STATE_INIT = "Errors will be shown here"
STATE_URLERROR = "Network error"
STATE_RESET_DEFAULT = "All settings reset"
STATE_WAIT = "Please wait..."
STATE_DONE = "Done!"
STATE_INTERRUPT = "Interrupted!"

# Other currently hardcoded stuff
SHORT_TIMEOUT = 10
LONG_TIMEOUT = None  # requests that might take "forever", i.e., image generation with high batch count
REFRESH_INTERVAL = 3000  # 3 seconds between auto-config refresh
ETA_REFRESH_INTERVAL = 250  # milliseconds between eta refresh
CFG_FOLDER = "krita"  # which folder in ~/.config to store config
CFG_NAME = "krita_diff_plugin"  # name of config file
EXT_CFG_NAME = "krita_diff_plugin_scripts"  # name of config file
ADD_MASK_TIMEOUT = 50
THREADED = True
ROUTE_PREFIX = ""
OFFICIAL_ROUTE_PREFIX = "/sdapi/v1/"
CONTROLNET_ROUTE_PREFIX = "/controlnet/"
CONTROLNET_ROUTE_PREFIX = "/controlnet/"

# error messages
ERR_MISSING_CONFIG = "Report this bug, developer missed out a config key somewhere."
ERR_NO_DOCUMENT = "No document open yet!"
ERR_NO_CONNECTION = "Cannot reach backend!"
ERR_BAD_URL = "Invalid backend URL!"

# tab IDs
TAB_SDCOMMON = "krita_diff_sdcommon"
TAB_CONFIG = "krita_diff_config"
TAB_TXT2IMG = "krita_diff_txt2img"
TAB_IMG2IMG = "krita_diff_img2img"
TAB_INPAINT = "krita_diff_inpaint"
TAB_UPSCALE = "krita_diff_upscale"
TAB_WORKFLOW = "krita_diff_workflow"
TAB_CONTROLNET = "krita_diff_controlnet"
TAB_PREVIEW = "krita_diff_preview"

# Nodes

DEFAULT_NODE_IDS = {
    "KSampler": "3",
    "CheckpointLoaderSimple" : "4",
    "EmptyLatentImage": "5",
    "ClipTextEncode_pos": "6",
    "ClipTextEncode_neg": "7",
    "VAEDecode": "VAEDecode",
    "SaveImage": "SaveImage",
    "ClipSetLastLayer": "ClipSetLastLayer",
    "VAELoader": "VAELoader",

    #Upscale
    "UpscaleModelLoader": "UpscaleModelLoader",
    "ImageUpscaleWithModel": "ImageUpscaleWithModel",
    "LatentUpscale": "LatentUpscale",
    "KSampler_upscale": "KSampler_upscale",
    "VAEDecode_upscale": "VAEDecode_upscale",
    "ImageScale": "ImageScale",
    "ImageScaleBy": "ImageScaleBy",
    "VAEEncode_upscale": "VAEEncode_upscale",
    "SetLatentNoiseMask_upscale": "SetLatentNoiseMask_upscale",

    #Img2img
    "LoadBase64Image": "LoadBase64Image",
    "VAEEncode": "VAEEncode",

    #Inpaint
    "LoadBase64ImageMask": "LoadBase64ImageMask",
    "VAEEncodeForInpaint": "VAEEncodeForInpaint",
    "SetLatentNoiseMask": "SetLatentNoiseMask",

    #Lora
    "LoraLoader": "LoraLoader",

    #Controlnet
    "ControlNetLoader": "ControlNetLoader",
    "ControlNetImageLoader": "ControlNetImageLoader",
    "ControlNetApplyAdvanced": "ControlNetApplyAdvanced",

    #Revision
    "CLIPVisionLoader": "CLIPVisionLoader",
    "CLIPVisionEncode": "CLIPVisionEncode",
    "unCLIPConditioning": "unCLIPConditioning"
}

# Workflow data placeholders
PRUNED_DATA = "<<PrunedImageData>>"
SELECTED_IMAGE = "<<SelectedImage>>"
CURRENT_LAYER_AS_MASK = "<<CurrentLayerAsMask>>"
LAST_LOADED_LORA = r"<<LastLoadedLora\|(.+?)>>"
PROMPT = "<<Prompt>>"
NEGATIVE_PROMPT = "<<NegativePrompt>>"

@dataclass(frozen=True)
class Defaults:
    base_url: str = "http://127.0.0.1:8188"
    encryption_key: str = ""
    just_use_yaml: bool = False
    create_mask_layer: bool = True
    save_temp_images: bool = False
    fix_aspect_ratio: bool = True
    only_full_img_tiling: bool = True
    filter_nsfw: bool = False
    do_exact_steps: bool = True
    sample_path: str = "./tmp"
    minimize_ui: bool = False
    first_setup: bool = True  # only used for the initial docker layout
    alt_dock_behavior: bool = False
    hide_layers: bool = True
    no_groups: bool = False
    disable_sddebz_highres: bool = False

    sd_model_list: List[str] = field(default_factory=lambda: [ERROR_MSG])
    sd_model: str = "model.ckpt"
    sd_vae_list: List[str] = field(default_factory=lambda: [ERROR_MSG])
    sd_vae: str = "Automatic"
    clip_skip: int = 1
    sd_batch_size: int = 1
    sd_batch_count: int = 1
    sd_base_size: int = 512
    sd_max_size: int = 768
    sd_tiling: bool = False
    upscaler_list: List[str] = field(default_factory=lambda: [ERROR_MSG])
    upscaler_methods_list: List[str] = field(default_factory=lambda: [ERROR_MSG])
    upscaler_model_list: List[str] = field(default_factory=lambda: [ERROR_MSG])
    upscaler_name: str = "None"
    face_restorer_model_list: List[str] = field(default_factory=lambda: [ERROR_MSG])
    face_restorer_model: str = "None"
    codeformer_weight: float = 0.5
    include_grid: bool = False

    txt2img_prompt: str = ""
    txt2img_negative_prompt: str = ""
    txt2img_sampler_list: List[str] = field(default_factory=lambda: [ERROR_MSG])
    txt2img_sampler: str = "Euler a"
    txt2img_scheduler_list: List[str] = field(default_factory=lambda: [ERROR_MSG])
    txt2img_scheduler: str = "normal"
    txt2img_steps: int = 20
    txt2img_cfg_scale: float = 7.0
    txt2img_denoising_strength: float = 0.3
    txt2img_seed: str = ""
    txt2img_workflow: str = ""
    txt2img_custom_workflow: bool = False

    img2img_prompt: str = ""
    img2img_negative_prompt: str = ""
    img2img_sampler_list: List[str] = field(default_factory=lambda: [ERROR_MSG])
    img2img_sampler: str = "Euler a"
    img2img_scheduler_list: List[str] = field(default_factory=lambda: [ERROR_MSG])
    img2img_scheduler: str = "normal"
    img2img_steps: int = 20
    img2img_cfg_scale: float = 8.0
    img2img_denoising_strength: float = 0.5
    img2img_seed: str = ""
    img2img_color_correct: bool = False
    img2img_input_save_as: str = "input.png"
    img2img_workflow: str = ""
    img2img_custom_workflow: bool = False

    inpaint_prompt: str = ""
    inpaint_negative_prompt: str = ""
    inpaint_sampler_list: List[str] = field(default_factory=lambda: [ERROR_MSG])
    inpaint_sampler: str = "LMS"
    inpaint_scheduler_list: List[str] = field(default_factory=lambda: [ERROR_MSG])
    inpaint_scheduler: str = "normal"
    inpaint_steps: int = 20
    inpaint_cfg_scale: float = 8.0
    inpaint_denoising_strength: float = 0.40
    inpaint_seed: str = ""
    inpaint_invert_mask: bool = False
    # inpaint_mask_blur: int = 4
    inpaint_fill_list: List[str] = field(
        default_factory=lambda: ["preserve", "latent noise"]
    )
    inpaint_fill: str = "preserve"
    # inpaint_full_res: bool = False
    # inpaint_full_res_padding: int = 32
    inpaint_color_correct: bool = False
    inpaint_mask_weight: float = 1.0
    inpaint_workflow: str = ""
    inpaint_custom_workflow: bool = False

    upscale_upscaler_name: str = "None"
    upscale_upscale_by: float = 1.0
    upscale_workflow: str = ""
    upscale_custom_workflow: bool = False

    workflow_to_list: List[str] = field(default_factory=lambda: ["none", "txt2img", "img2img", "inpaint", "upscale"])
    workflow_to: str = "none"
    none_workflow: str = ""
    workflow_img_data: Dict[str, object] = field(default_factory=lambda: {})

    controlnet_unit: str = "0"
    controlnet_unit_list: List[str] = field(default_factory=lambda: list(str(i) for i in range(10)))
    controlnet_preprocessor_list: List[str] = field(default_factory=lambda: [ERROR_MSG])
    controlnet_model_list: List[str] = field(default_factory=lambda: [ERROR_MSG])
    controlnet_preprocessors_info: Dict[str, object] = field(default_factory=lambda: {})
    #controlnet_control_mode_list: List[str] = field(default_factory=lambda: ["Balanced", "My prompt is more important", "ControlNet is more important"])

    controlnet0_enable: bool = False
    controlnet0_preprocessor: str = "None" 
    controlnet0_model: str = "None"
    controlnet0_weight: float = 1.0
    controlnet0_guidance_start: float = 0
    controlnet0_guidance_end: float = 1
    controlnet0_inputs: Dict[str, object] = field(default_factory=lambda: {})
    controlnet0_input_image: str = ""
    #controlnet0_control_mode: str = "Balanced"

    controlnet1_enable: bool = False
    controlnet1_preprocessor: str = "None" 
    controlnet1_model: str = "None"
    controlnet1_weight: float = 1.0
    controlnet1_guidance_start: float = 0
    controlnet1_guidance_end: float = 1
    controlnet1_inputs: Dict[str, object] = field(default_factory=lambda: {})
    controlnet1_input_image: str = ""
    #controlnet1_control_mode: str = "Balanced"

    controlnet2_enable: bool = False
    controlnet2_preprocessor: str = "None"
    controlnet2_model: str = "None"
    controlnet2_weight: float = 1.0
    controlnet2_guidance_start: float = 0
    controlnet2_guidance_end: float = 1
    controlnet2_inputs: Dict[str, object] = field(default_factory=lambda: {})
    controlnet2_input_image: str = ""
    #controlnet2_control_mode: str = "Balanced"

    controlnet3_enable: bool = False
    controlnet3_preprocessor: str = "None"
    controlnet3_model: str = "None"
    controlnet3_weight: float = 1.0
    controlnet3_guidance_start: float = 0
    controlnet3_guidance_end: float = 1
    controlnet3_inputs: Dict[str, object] = field(default_factory=lambda: {})
    controlnet3_input_image: str = ""
    #controlnet3_control_mode: str = "Balanced"

    controlnet4_enable: bool = False
    controlnet4_preprocessor: str = "None"
    controlnet4_model: str = "None"
    controlnet4_weight: float = 1.0
    controlnet4_guidance_start: float = 0
    controlnet4_guidance_end: float = 1
    controlnet4_inputs: Dict[str, object] = field(default_factory=lambda: {})
    controlnet4_input_image: str = ""
    #controlnet4_control_mode: str = "Balanced"

    controlnet5_enable: bool = False
    controlnet5_preprocessor: str = "None"
    controlnet5_model: str = "None"
    controlnet5_weight: float = 1.0
    controlnet5_guidance_start: float = 0
    controlnet5_guidance_end: float = 1
    controlnet5_inputs: Dict[str, object] = field(default_factory=lambda: {})
    controlnet5_input_image: str = ""
    #controlnet5_control_mode: str = "Balanced"

    controlnet6_enable: bool = False
    controlnet6_preprocessor: str = "None"
    controlnet6_model: str = "None"
    controlnet6_weight: float = 1.0
    controlnet6_guidance_start: float = 0
    controlnet6_guidance_end: float = 1
    controlnet6_inputs: Dict[str, object] = field(default_factory=lambda: {})
    controlnet6_input_image: str = ""
    #controlnet6_control_mode: str = "Balanced"

    controlnet7_enable: bool = False
    controlnet7_preprocessor: str = "None"
    controlnet7_model: str = "None"
    controlnet7_weight: float = 1.0
    controlnet7_guidance_start: float = 0
    controlnet7_guidance_end: float = 1
    controlnet7_inputs: Dict[str, object] = field(default_factory=lambda: {})
    controlnet7_input_image: str = ""
    #controlnet7_control_mode: str = "Balanced"

    controlnet8_enable: bool = False
    controlnet8_preprocessor: str = "None"
    controlnet8_model: str = "None"
    controlnet8_weight: float = 1.0
    controlnet8_guidance_start: float = 0
    controlnet8_guidance_end: float = 1
    controlnet8_inputs: Dict[str, object] = field(default_factory=lambda: {})
    controlnet8_input_image: str = ""
    #controlnet8_control_mode: str = "Balanced"

    controlnet9_enable: bool = False
    controlnet9_preprocessor: str = "None"
    controlnet9_model: str = "None"
    controlnet9_weight: float = 1.0
    controlnet9_guidance_start: float = 0
    controlnet9_guidance_end: float = 1
    controlnet9_inputs: Dict[str, object] = field(default_factory=lambda: {})
    controlnet9_input_image: str = ""
    #controlnet9_control_mode: str = "Balanced"

DEFAULTS = Defaults()
