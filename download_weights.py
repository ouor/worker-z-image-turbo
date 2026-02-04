import os
from huggingface_hub import hf_hub_download

MODELS_DIR = "/models"

def download_z_image_models():
    """
    Downloads the Z-Image-Turbo models (Diffusion, LLM, VAE) to the local directory.
    """
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    print("Downloading Z-Image-Turbo models...")

    # 1. Diffusion Model (GGUF)
    print("Downloading Diffusion Model (z_image_turbo-Q3_K.gguf)...")
    hf_hub_download(
        repo_id="leejet/Z-Image-Turbo-GGUF",
        filename="z_image_turbo-Q3_K.gguf",
        local_dir=MODELS_DIR,
        local_dir_use_symlinks=False
    )

    # 2. LLM / Text Encoder (GGUF)
    print("Downloading LLM (Qwen3-4B-Instruct-2507-Q4_K_M.gguf)...")
    hf_hub_download(
        repo_id="unsloth/Qwen3-4B-Instruct-2507-GGUF",
        filename="Qwen3-4B-Instruct-2507-Q4_K_M.gguf",
        local_dir=MODELS_DIR,
        local_dir_use_symlinks=False
    )

    # 3. VAE (Safetensors)
    print("Downloading VAE (ae.safetensors)...")
    # Note: Using the official FLUX.1-schnell repo for VAE as suggested in TIP.md
    hf_hub_download(
        repo_id="receptektas/black-forest-labs-ae_safetensors",
        filename="ae.safetensors",
        local_dir=MODELS_DIR,
        local_dir_use_symlinks=False
    )
    
    print(f"All models downloaded to {MODELS_DIR}")

if __name__ == "__main__":
    download_z_image_models()
