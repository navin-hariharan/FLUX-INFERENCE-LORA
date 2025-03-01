import os, json, requests, runpod
import random, time
import torch, base64
import numpy as np
from PIL import Image
import gc
from functools import lru_cache
import nodes
from nodes import NODE_CLASS_MAPPINGS
from comfy_extras import nodes_custom_sampler
from comfy_extras import nodes_flux
from comfy import model_management

# Cache for downloaded files to avoid re-downloading
downloaded_files_cache = {}

def download_file(url, save_dir='/content/ComfyUI/models/loras'):
    # Check if file already exists in cache
    if url in downloaded_files_cache:
        return downloaded_files_cache[url]

    os.makedirs(save_dir, exist_ok=True)
    file_name = url.split('/')[-1]
    file_path = os.path.join(save_dir, file_name)

    # Check if file already exists on disk
    if os.path.exists(file_path):
        downloaded_files_cache[url] = file_path
        return file_path

    response = requests.get(url)
    response.raise_for_status()
    with open(file_path, 'wb') as file:
        file.write(response.content)

    # Cache the downloaded file path
    downloaded_files_cache[url] = file_path
    return file_path

# Initialize node classes once
DualCLIPLoader = NODE_CLASS_MAPPINGS["DualCLIPLoader"]()
UNETLoader = NODE_CLASS_MAPPINGS["UNETLoader"]()
VAELoader = NODE_CLASS_MAPPINGS["VAELoader"]()
LoraLoader = NODE_CLASS_MAPPINGS["LoraLoader"]()
FluxGuidance = nodes_flux.NODE_CLASS_MAPPINGS["FluxGuidance"]()
RandomNoise = nodes_custom_sampler.NODE_CLASS_MAPPINGS["RandomNoise"]()
BasicGuider = nodes_custom_sampler.NODE_CLASS_MAPPINGS["BasicGuider"]()
KSamplerSelect = nodes_custom_sampler.NODE_CLASS_MAPPINGS["KSamplerSelect"]()
BasicScheduler = nodes_custom_sampler.NODE_CLASS_MAPPINGS["BasicScheduler"]()
SamplerCustomAdvanced = nodes_custom_sampler.NODE_CLASS_MAPPINGS["SamplerCustomAdvanced"]()
EmptyLatentImage = NODE_CLASS_MAPPINGS["EmptyLatentImage"]()
VAEDecode = NODE_CLASS_MAPPINGS["VAEDecode"]()

# Model caching
model_cache = {}

# Load models with caching
def load_models():
    global model_cache

    with torch.inference_mode():
        # Use cache if models are already loaded
        if "clip" not in model_cache:
            model_cache["clip"] = DualCLIPLoader.load_clip("t5xxl_fp16.safetensors", "clip_l.safetensors", "flux")[0]

        if "unet" not in model_cache:
            model_cache["unet"] = UNETLoader.load_unet("flux1-dev.sft", "default")[0]

        if "vae" not in model_cache:
            model_cache["vae"] = VAELoader.load_vae("ae.sft")[0]

    return model_cache["clip"], model_cache["unet"], model_cache["vae"]

# Cache for LoRA models
lora_cache = {}

# Helper function to get or load LoRA
def get_or_load_lora(unet, clip, lora_file, lora_strength_model, lora_strength_clip):
    cache_key = f"{lora_file}_{lora_strength_model}_{lora_strength_clip}"

    if cache_key not in lora_cache:
        unet_lora, clip_lora = LoraLoader.load_lora(unet, clip, lora_file, lora_strength_model, lora_strength_clip)
        lora_cache[cache_key] = (unet_lora, clip_lora)

    return lora_cache[cache_key]

def closestNumber(n, m):
    q = int(n / m)
    n1 = m * q
    if (n * m) > 0:
        n2 = m * (q + 1)
    else:
        n2 = m * (q - 1)
    if abs(n - n1) < abs(n - n2):
        return n1
    return n2

@torch.inference_mode()
def generate(input):
    try:
        values = input["input"]
        positive_prompt = values['positive_prompt']
        width = values['width']
        height = values['height']
        seed = values['seed']
        steps = values['steps']
        guidance = values['guidance']
        lora_strength_model = values['lora_strength_model']
        lora_strength_clip = values['lora_strength_clip']
        sampler_name = values['sampler_name']
        scheduler = values['scheduler']
        lora_url = values['lora_url']

        # Load base models if not already in cache
        clip, unet, vae = load_models()

        # Download LoRA file
        lora_file_path = download_file(lora_url)
        lora_file = os.path.basename(lora_file_path)

        # Generate random seed if not provided
        if seed == 0:
            random.seed(int(time.time()))
            seed = random.randint(0, 18446744073709551615)
        print(seed)

        # Use cached LoRA if possible
        unet_lora, clip_lora = get_or_load_lora(unet, clip, lora_file, lora_strength_model, lora_strength_clip)

        # Move operations to appropriate device and use memory-efficient encoding
        device = model_management.get_torch_device()

        # Encode prompt
        cond, pooled = clip_lora.encode_from_tokens(clip_lora.tokenize(positive_prompt), return_pooled=True)
        cond = [[cond, {"pooled_output": pooled}]]
        cond = FluxGuidance.append(cond, guidance)[0]

        # Generate image
        noise = RandomNoise.get_noise(seed)[0]
        guider = BasicGuider.get_guider(unet_lora, cond)[0]
        sampler = KSamplerSelect.get_sampler(sampler_name)[0]
        sigmas = BasicScheduler.get_sigmas(unet_lora, scheduler, steps, 1.0)[0]

        # Adjust dimensions to be divisible by 16
        width_adjusted = closestNumber(width, 16)
        height_adjusted = closestNumber(height, 16)
        latent_image = EmptyLatentImage.generate(width_adjusted, height_adjusted)[0]

        # Sample and decode
        sample, sample_denoised = SamplerCustomAdvanced.sample(noise, guider, sampler, sigmas, latent_image)
        decoded = VAEDecode.decode(vae, sample)[0].detach()

        # Save and encode image
        image_path = "/content/flux.png"
        Image.fromarray(np.array(decoded*255, dtype=np.uint8)[0]).save(image_path)

        with open(image_path, 'rb') as image_file:
            base64_string = base64.b64encode(image_file.read()).decode('utf-8')

        # Clear CUDA cache to free up VRAM
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Run garbage collection
        gc.collect()

        return {"image": base64_string}

    except Exception as e:
        # Clean up on error
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        return {"result": f"FAILED: {str(e)}", "status": "FAILED"}

    finally:
        # Ensure cleanup happens
        if 'image_path' in locals() and os.path.exists(image_path):
            os.remove(image_path)

# Function to clear all caches and free memory (call this if memory gets too high)
def clear_all_caches():
    global model_cache, lora_cache

    # Clear model cache
    model_cache = {}

    # Clear LoRA cache
    lora_cache = {}

    # Clear CUDA cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Run aggressive garbage collection
    gc.collect()

# Modify the handler to manage memory between jobs
def handler(event):
    # Check if memory usage is high before processing
    if torch.cuda.is_available() and torch.cuda.memory_allocated() > 0.9 * torch.cuda.get_device_properties(0).total_memory:
        clear_all_caches()

    result = generate(event)

    # Clean up after job
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    return result

# Start the serverless handler
runpod.serverless.start({"handler": handler})
