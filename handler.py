import os
import base64
from io import BytesIO

from stable_diffusion_cpp import StableDiffusion
import runpod
from runpod.serverless.utils import rp_upload, rp_cleanup
from runpod.serverless.utils.rp_validator import validate

from schemas import INPUT_SCHEMA

# Define model paths
MODELS_DIR = "/models"
DIFFUSION_MODEL_PATH = os.path.join(MODELS_DIR, "z_image_turbo-Q3_K.gguf")
LLM_PATH = os.path.join(MODELS_DIR, "Qwen3-4B-Instruct-2507-Q4_K_M.gguf")
VAE_PATH = os.path.join(MODELS_DIR, "ae.safetensors")

class ModelHandler:
    def __init__(self):
        self.sd = None
        self.load_model()

    def load_model(self):
        print("Loading Z-Image-Turbo models...")
        # Initialize StableDiffusion with Z-Image-Turbo configuration
        self.sd = StableDiffusion(
            diffusion_model_path=DIFFUSION_MODEL_PATH,
            llm_path=LLM_PATH,
            vae_path=VAE_PATH,
            offload_params_to_cpu=True, # Recommended for lower VRAM usage
            diffusion_flash_attn=True,  # Enable Flash Attention for speed/memory
            wtype="default",
        )
        print("Models loaded successfully.")

MODELS = ModelHandler()

def _save_and_upload_images(images, job_id):
    os.makedirs(f"/{job_id}", exist_ok=True)
    image_urls = []
    
    # images is a list of PIL Images
    for index, image in enumerate(images):
        image_path = os.path.join(f"/{job_id}", f"{index}.png")
        image.save(image_path)

        if os.environ.get("BUCKET_ENDPOINT_URL", False):
            image_url = rp_upload.upload_image(job_id, image_path)
            image_urls.append(image_url)
        else:
            with open(image_path, "rb") as image_file:
                image_data = base64.b64encode(image_file.read()).decode("utf-8")
                image_urls.append(f"data:image/png;base64,{image_data}")

    rp_cleanup.clean([f"/{job_id}"])
    return image_urls

def generate_image(job):
    """
    Generate an image using Z-Image-Turbo via stable-diffusion-cpp-python
    """
    job_input = job["input"]

    # Input validation
    validated_input = validate(job_input, INPUT_SCHEMA)
    if "errors" in validated_input:
        return {"error": validated_input["errors"]}
    
    job_input = validated_input["validated_input"]

    # Set default seed if not provided
    if job_input["seed"] is None:
        job_input["seed"] = -1 # -1 means random seed in stable-diffusion.cpp

    print(f"Generating image with prompt: {job_input['prompt']}")
    
    try:
        # Generate image using the high-level API
        output_images = MODELS.sd.generate_image(
            prompt=job_input["prompt"],
            height=job_input["height"],
            width=job_input["width"],
            step=job_input.get("num_inference_steps", 10), # Default to fewer steps for Turbo models if not specified
            cfg_scale=job_input.get("guidance_scale", 1.0), # Recommended 1.0 for Z-Image
            seed=job_input["seed"],
        )
        
        # stable-diffusion-cpp-python returns a list of PIL Images
        
    except Exception as err:
        print(f"[ERROR] Error in generation pipeline: {err}")
        return {
            "error": f"Unexpected error: {err}",
            # "refresh_worker": True, # Uncomment if we need to restart worker on error
        }

    image_urls = _save_and_upload_images(output_images, job["id"])

    # Extract the used seed from the first image info if available, or pass back input seed
    # stable-diffusion-cpp-python objects might have an .info attribute, but for safety we return input seed if -1 was not unresolved
    # Ideally we should get the effective seed. For now returning input seed.
    
    results = {
        "images": image_urls,
        "image_url": image_urls[0],
        "seed": job_input["seed"],
    }

    return results

runpod.serverless.start({"handler": generate_image})
