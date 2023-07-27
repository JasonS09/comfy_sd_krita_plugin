# Comfy SD Krita Plugin
Make AI art between canvas and nodes with Krita.

![Screenshot 2023-07-27 090424](https://github.com/JasonS09/comfy_sd_krita_plugin/assets/47617351/7eef5466-ee00-4554-b2eb-359c8ee65bf9)
This is a mutation from [auto-sd-paint-ext](https://github.com/Interpause/auto-sd-paint-ext), adapted to ComfyUI. With this plugin, you'll be able to take advantage of ComfyUI's best features while working on a canvas.

Supports:
- Basic txt2img.
- Basic img2img.
- Inpainting (with auto-generated transparency masks).
- Simple upscale and upscaling with model (like Ultrasharp).
- Multicontrolnet with preprocessors. Annotator preview also available.
- Lora loading with Automatic1111 format.
- TI embeddings.
- (Advanced) Run custom ComfyUI workflow from Krita.
- (Advanced) Modify the default behaviour of each mode by injecting your own custom workflows!
- Import latest ComfyUI generations with one click.

### How does it work?

This plugin offers a (somewhat) user friendly UI which can allow the user to make AI art in a rather simple way (for those that don't like the nodes), while also giving the option to use traditional digital art and edition tools to work on generated images. The plugin creates the necessary workflow under the hood to be run on an active ComfyUI instance. Generated images are then requested by the plugin and added to the canvas, and are also stored on your ComfyUI outputs directory. You are also given the possibility to work with your own custom workflows and inject them to the plugin to alter the behaviour of your generations.

### Requirements
- Krita (of course).
- ComfyUI (of course).
- [The custom nodes from this repo](https://github.com/JasonS09/comfy_sd_krita_plugin_nodes)
- [Fannovel16's ComfyUI controlnet preprocessors](https://github.com/Fannovel16/comfy_controlnet_preprocessors)

### Installation

#### The custom nodes
- Simply open a command line prompt and go to your ComfyUI's [custom nodes directory](https://github.com/comfyanonymous/ComfyUI/tree/master/custom_nodes).
- Run the following:
  `git clone https://github.com/JasonS09/comfy_sd_krita_plugin_nodes`
- Then run the following:
  `git clone https://github.com/Fannovel16/comfy_controlnet_preprocessors`

#### The plugin
- Clone this repo somewhere: `https://github.com/JasonS09/comfy_sd_krita_plugin`
- Locate `pykrita` folder (Open Krita -> Settings -> Manage Resources -> Open Resource Folder). `pykrita` should be in that directory.
- Open a new command prompt as Admin.
- Make a symlink of `krita_comfy` and `krita_comfy.desktop` present in this repository (in the frontends/krita directory).
Windows commands example:  
`mklink /d your-pykrita-dir\krita_comfy "your-local-repo-dir\frontends\krita\krita_comfy"`  
`mklink your-pykrita-dir\krita_comfy.desktop "your-local-repo-dir\frontends\krita\krita_comfy.desktop"`
- Your plugin is ready.

*Usage guide coming soon...*

Found this useful? [Buy me a coffee!](https://www.buymeacoffee.com/piratewolf09)
