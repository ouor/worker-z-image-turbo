import os
import base64
from io import BytesIO
from PIL import ImageStat

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
        try:
            self.load_model()
        except Exception as e:
            print(f"[FATAL] Failed to load models: {e}")
            raise RuntimeError(f"Model load failed: {e}")

    def load_model(self):
        if not os.path.exists(DIFFUSION_MODEL_PATH):
            raise FileNotFoundError(f"Diffusion model not found at {DIFFUSION_MODEL_PATH}")
        if not os.path.exists(LLM_PATH):
             raise FileNotFoundError(f"LLM model not found at {LLM_PATH}")
        if not os.path.exists(VAE_PATH):
             raise FileNotFoundError(f"VAE model not found at {VAE_PATH}")

        print("Loading Z-Image-Turbo models...")
        try:
            # Initialize StableDiffusion with Z-Image-Turbo configuration
            self.sd = StableDiffusion(
                diffusion_model_path=DIFFUSION_MODEL_PATH,
                llm_path=LLM_PATH,
                vae_path=VAE_PATH,
                offload_params_to_cpu=False, # Recommended for lower VRAM usage
                diffusion_flash_attn=True,  # Enable Flash Attention for speed/memory
                wtype="default",
            )
            print("Models loaded successfully.")
        except Exception as e:
             raise RuntimeError(f"Error initializing StableDiffusion pipeline: {e}")

try:
    MODELS = ModelHandler()
except RuntimeError as e:
    # If model fails to load, we cannot recover. Print and exit.
    print(str(e))
    exit(1)

def is_valid_image(image, threshold=1.0):
    """
    Checks if the image has enough variance (is not a solid color).
    Threshold is based on standard deviation of pixel values.
    """
    stat = ImageStat.Stat(image)
    # Check standard deviation of RGB channels
    for stddev in stat.stddev:
        if stddev > threshold:
            return True
    return False

def generate_image(job):
    """
    Generate a single image using Z-Image-Turbo and return base64 data
    """
    try:
        job_input = job.get("input")
        if not job_input:
             return {"error": "Missing 'input' field in job payload"}

        # Input validation
        validated_input = validate(job_input, INPUT_SCHEMA)
        if "errors" in validated_input:
            return {"error": validated_input["errors"]}
        
        job_input = validated_input["validated_input"]

        # Logical validation for dimensions
        if job_input['height'] % 8 != 0 or job_input['width'] % 8 != 0:
             return {"error": "Height and Width must be multiples of 8"}
        
        # Set default seed if not provided
        if job_input["seed"] is None:
            job_input["seed"] = -1 # -1 means random seed in stable-diffusion.cpp

        print(f"Generating image with prompt: {job_input['prompt']}")
        
        # Generate image (always returns a list, we take the first one)
        output_images = MODELS.sd.generate_image(
            prompt=job_input["prompt"],
            height=job_input["height"],
            width=job_input["width"],
            cfg_scale=job_input["cfg_scale"],
        )
        
        if not output_images:
            return {"error": "Model failed to generate any images"}
            
        image = output_images[0]

        # Check for solid color (broken) output
        if not is_valid_image(image):
            print("[ERROR] Generated image is a solid color (broken output)")
            return {"error": "Generated image is invalid (solid color). This can happen due to numerical instability."}
            
        # Convert PIL Image to Base64
        try:
            buffered = BytesIO()
            image.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        except Exception as e:
             print(f"[ERROR] Failed to encode image: {e}")
             return {"error": "Failed to encode output image"}
        
        return {
            "mime_type": "image/png",
            "data": img_str
        }

    except Exception as err:
        print(f"[ERROR] Detailed error in generation pipeline: {err}")
        return {
            "error": f"Unexpected error: {err}"
        }

runpod.serverless.start({"handler": generate_image})
