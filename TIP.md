# 🖼️ Python Bindings for [`stable-diffusion.cpp`](https://github.com/leejet/stable-diffusion.cpp)

Simple Python bindings for **@leejet's** [`stable-diffusion.cpp`](https://github.com/leejet/stable-diffusion.cpp) library.

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![PyPi version](https://badgen.net/pypi/v/pywhispercpp)](https://pypi.org/project/stable-diffusion-cpp-python/)
[![Downloads](https://static.pepy.tech/badge/stable-diffusion-cpp-python)](https://pepy.tech/project/stable-diffusion-cpp-python)

This package provides:

- Low-level access to C API via `ctypes` interface.
- High-level Python API for Stable Diffusion, FLUX and Wan image/video generation.

## Installation

Requirements:

- Python 3.8+
- C compiler
  - Linux: gcc or clang
  - Windows: Visual Studio or MinGW
  - MacOS: Xcode

To install the package, run:

```bash
pip install stable-diffusion-cpp-python
```

This will also build `stable-diffusion.cpp` from source and install it alongside this python package.

If this fails, add `--verbose` to the `pip install` to see the full cmake build log.

### Installation Configuration

`stable-diffusion.cpp` supports a number of hardware acceleration backends to speed up inference as well as backend specific options. See the [stable-diffusion.cpp README](https://github.com/leejet/stable-diffusion.cpp#build) for a full list.

All `stable-diffusion.cpp` cmake build options can be set via the `CMAKE_ARGS` environment variable or via the `--config-settings / -C` cli flag during installation.

<details open>
<summary>Environment Variables</summary>

```bash
# Linux and Mac
CMAKE_ARGS="-DSD_CUDA=ON" pip install stable-diffusion-cpp-python
```

```powershell
# Windows
$env:CMAKE_ARGS="-DSD_CUDA=ON"
pip install stable-diffusion-cpp-python
```

</details>

<details>
<summary>CLI / requirements.txt</summary>

They can also be set via `pip install -C / --config-settings` command and saved to a `requirements.txt` file:

```bash
pip install --upgrade pip # ensure pip is up to date
pip install stable-diffusion-cpp-python -C cmake.args="-DSD_CUDA=ON"
```

```txt
# requirements.txt

stable-diffusion-cpp-python -C cmake.args="-DSD_CUDA=ON"
```

</details>

### Supported Backends

Below are some common backends, their build commands and any additional environment variables required.

<!-- CUDA -->
<details>
<summary>Using CUDA (CUBLAS)</summary>

This provides BLAS acceleration using the CUDA cores of your Nvidia GPU. Make sure you have the CUDA toolkit installed. You can download it from your Linux distro's package manager (e.g. `apt install nvidia-cuda-toolkit`) or from here: [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads). You can check your installed CUDA toolkit version by running `nvcc --version`.

- It is recommended you have at least 4 GB of VRAM.

```bash
CMAKE_ARGS="-DSD_CUDA=ON" pip install stable-diffusion-cpp-python
```

</details>

### Using Flash Attention

Enabling flash attention for the diffusion model reduces memory usage by varying amounts of MB, e.g.:

- **flux 768x768** ~600mb
- **SD2 768x768** ~1400mb

For most backends, it slows things down, but for cuda it generally speeds it up too.
At the moment, it is only supported for some models and some backends (like `cpu`, `cuda/rocm` and `metal`).

Run by passing `diffusion_flash_attn=True` to the `StableDiffusion` class and watch for:

```log
[INFO] stable-diffusion.cpp:312  - Using flash attention in the diffusion model
```

and the compute buffer shrink in the debug log:

```log
[DEBUG] ggml_extend.hpp:1004 - flux compute buffer size: 650.00 MB(VRAM)
```

## High-level API

The high-level API provides a simple managed interface through the `StableDiffusion` class.

Below is a short example demonstrating how to use the high-level API to generate a simple image:

### <u>Text to Image</u>

```python
from PIL import Image
from stable_diffusion_cpp import StableDiffusion

def progress_callback(step: int, steps: int, time: float):
    print("Completed step: {} of {}".format(step, steps))

def preview_callback(step: int, images: list[Image.Image], is_noisy: bool):
    images[0].save(f"preview/{step}.png")

stable_diffusion = StableDiffusion(
      model_path="../models/v1-5-pruned-emaonly.safetensors",
      # wtype="default", # Weight type (e.g. "q8_0", "f16", etc) (The "default" setting is automatically applied and determines the weight type of a model file)
)
output = stable_diffusion.generate_image(
      prompt="a lovely cat",
      width=512,
      height=512,
      progress_callback=progress_callback,
      # seed=1337, # Uncomment to set a specific seed (use -1 for a random seed)
      preview_method="proj",
      preview_interval=2,  # Call every 2 steps
      preview_callback=preview_callback,
)
output[0].save("output.png") # Output returned as list of PIL Images

# Model and generation paramaters accessible via .info
print(output[0].info)
```


#### <u>Z-Image</u>

Download the weights from the links below:

- Download `Z-Image-Turbo`
  - safetensors: https://huggingface.co/Comfy-Org/z_image_turbo/tree/main/split_files/diffusion_models
  - gguf: https://huggingface.co/leejet/Z-Image-Turbo-GGUF/tree/main
- Download `vae`
  - safetensors: https://huggingface.co/black-forest-labs/FLUX.1-schnell/tree/main
- Download `Qwen3 4b`
  - safetensors: https://huggingface.co/Comfy-Org/z_image_turbo/tree/main/split_files/text_encoders
  - gguf: https://huggingface.co/unsloth/Qwen3-4B-Instruct-2507-GGUF/tree/main

```python
from stable_diffusion_cpp import StableDiffusion

stable_diffusion = StableDiffusion(
      diffusion_model_path="../models/z_image_turbo-Q3_K.gguf",
      llm_path="../models/Qwen3-4B-Instruct-2507-Q4_K_M.gguf",
      vae_path="../models/ae.safetensors",
      offload_params_to_cpu=True,
      diffusion_flash_attn=True,
)

output = stable_diffusion.generate_image(
      prompt="A cinematic, melancholic photograph of a solitary hooded figure walking through a sprawling, rain-slicked metropolis at night. The city lights are a chaotic blur of neon orange and cool blue, reflecting on the wet asphalt. The scene evokes a sense of being a single component in a vast machine. Superimposed over the image in a sleek, modern, slightly glitched font is the philosophical quote: 'THE CITY IS A CIRCUIT BOARD, AND I AM A BROKEN TRANSISTOR.' -- moody, atmospheric, profound, dark academic",
      height=1024,
      width=512,
      cfg_scale=1.0,
)
```