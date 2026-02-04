# Z-Image-Turbo Worker

Run [Z-Image-Turbo](https://huggingface.co/leejet/Z-Image-Turbo-GGUF) as a serverless endpoint using `stable-diffusion.cpp` for high-performance, low-memory inference.

---

## 🚀 Key Features

- **Model**: Z-Image-Turbo (GGUF Quantized)
- **Backend**: [stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp) via Python bindings
- **Performance**: FPGA-like efficiency with CUDA acceleration and Flash Attention
- **Optimized**: Uses `Qwen3-4B` as LLM and `FLUX.1-schnell` VAE

---

## Usage

The worker accepts the following input parameters:

| Parameter             | Type    | Default | Required | Description                                                  |
| :-------------------- | :------ | :------ | :------- | :----------------------------------------------------------- |
| `prompt`              | `str`   | -       | **Yes**  | The text prompt describing the desired image.                 |
| `height`              | `int`   | `1024`  | No       | The height of the generated image.                           |
| `width`               | `int`   | `512`   | No       | The width of the generated image.                            |
| `seed`                | `int`   | `None`  | No       | Random seed for reproducibility. If `None`, random.          |
| `num_inference_steps` | `int`   | `8`     | No       | Number of denoising steps (Turbo models need fewer steps).   |
| `guidance_scale`      | `float` | `1.0`   | No       | CFG Scale. Recommended `1.0` for Z-Image-Turbo.              |
| `num_images`          | `int`   | `1`     | No       | Number of images to generate per prompt.                     |

### Example Request

```json
{
  "input": {
    "prompt": "A cinematic, melancholic photograph of a solitary hooded figure walking through a rain-slicked metropolis at night, neon lights reflecting on wet asphalt, cyberpunk atmosphere",
    "height": 1024,
    "width": 512,
    "num_inference_steps": 8,
    "guidance_scale": 1.0,
    "seed": 42
  }
}
```

### Example Output

```json
{
  "delayTime": 150,
  "executionTime": 2500,
  "id": "job-id-example",
  "output": {
    "image_url": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgA...",
    "images": [
      "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgA..."
    ],
    "seed": 42
  },
  "status": "COMPLETED"
}
```

## Development

This worker is built using:
- `stable-diffusion-cpp-python` for efficient inference
- `runpod` sdk for serverless handling
- GGUF Quantized models for VRAM optimization
