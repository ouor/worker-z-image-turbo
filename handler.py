import os
import base64
from io import BytesIO

from stable_diffusion_cpp import StableDiffusion
import runpod
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

def generate_image(job):
    """
    Generate a single image using Z-Image-Turbo and return base64 data
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
        # Generate image (always returns a list, we take the first one)
        output_images = MODELS.sd.generate_image(
            prompt=job_input["prompt"],
            height=job_input["height"],
            width=job_input["width"],
            step=job_input["num_inference_steps"],
            cfg_scale=job_input["guidance_scale"],
            seed=job_input["seed"],
        )
        
        if not output_images:
            raise Exception("No image generated")
            
        image = output_images[0]
        
        # Convert PIL Image to Base64
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        
        return {
            "mime_type": "image/png",
            "data": img_str
        }

    except Exception as err:
        print(f"[ERROR] Error in generation pipeline: {err}")
        return {
            "error": f"Unexpected error: {err}"
        }

runpod.serverless.start({"handler": generate_image})
