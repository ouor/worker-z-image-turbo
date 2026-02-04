# 🖼️ Z-Image-Turbo RunPod Worker

This project runs **Z-Image-Turbo** as a serverless endpoint on RunPod, leveraging the high-performance C++ backend of `stable-diffusion.cpp` via Python bindings.

---

## ⚡ Features

- **High Performance**: Uses `stable-diffusion.cpp` with CUDA acceleration for lightning-fast inference.
- **Turbo Optimized**: Specifically tuned for Z-Image-Turbo models (GGUF quantization).
- **Simplified API**: Clean input parameters and Base64 output for easy integration.
- **Robustness**: Includes error handling for numerical instability and empty/solid-color outputs.

---

## 🚀 Usage

The worker accepts the following parameters through the RunPod API:

| Parameter             | Type    | Default | Required | Description                                                                 |
| :-------------------- | :------ | :------ | :------- | :-------------------------------------------------------------------------- |
| `prompt`              | `str`   | `None`  | **Yes**  | The text description of the image you want to generate.                     |
| `height`              | `int`   | `1024`  | No       | Height of the generated image (must be a multiple of 8).                    |
| `width`               | `int`   | `1024`  | No       | Width of the generated image (must be a multiple of 8).                     |
| `seed`                | `int`   | `43`    | No       | Random seed for reproducibility. Use `-1` for a random seed.                |
| `num_inference_steps` | `int`   | `9`     | No       | Number of denoising steps. Turbo models usually need 4-10 steps.            |
| `guidance_scale`      | `float` | `1.0`   | No       | Classifier-Free Guidance scale. Recommended `1.0` for Z-Image-Turbo.        |

---

## 📥 Example Request

```json
{
  "input": {
    "prompt": "A futuristic Tokyo cityscape at night, neon lights, rainy street reflectons, cinematic lighting, 8k resolution",
    "height": 1024,
    "width": 1024,
    "num_inference_steps": 9,
    "guidance_scale": 1.0,
    "seed": 12345
  }
}
```

## 📤 Example Response

```json
{
  "delayTime": 120,
  "executionTime": 850,
  "id": "job-id-123",
  "output": {
    "mime_type": "image/png",
    "data": "iVBORw0KGgoAAAANSUhEUgAABAAAAAQACAIAAADwf7zU..."
  },
  "status": "COMPLETED"
}
```

---

## 🛠️ Local Development & Build

### Requirements
- NVIDIA GPU with CUDA Toolkit installed.
- Docker & NVIDIA Container Toolkit.

### Build the Image
```bash
docker build -t runpod-worker-z-image .
```

### Advanced Configuration
The worker uses `stable-diffusion-cpp-python` with the following optimizations enabled by default:
- **Flash Attention**: Enabled for reduced VRAM and faster inference.
- **CUDA Acceleration**: Set during build via `CMAKE_ARGS="-DSD_CUDA=ON"`.

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
All models used belong to their respective creators on Hugging Face.
