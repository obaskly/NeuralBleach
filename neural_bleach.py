import torch
import numpy as np
import cv2
import argparse
import os
import sys
from diffusers import StableDiffusionControlNetImg2ImgPipeline, ControlNetModel
from PIL import Image

def stego_washer_ultimate(image_path, output_filename=None, strength=0.25):
    # CHECK FILE EXISTENCE
    if not os.path.exists(image_path):
        print(f"[-] Error: File not found: {image_path}")
        return

    print(f"\n--- PROCESSING (ULTIMATE MODE): {image_path} ---")
    
    # Generate output filename if not provided
    if output_filename is None:
        base, ext = os.path.splitext(image_path)
        output_filename = f"{base}_clean.jpg"

    # Setup Device (Force CUDA if available, warn if CPU)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("[!] WARNING: Running on CPU. This will be extremely slow.")

    try:
        # LOAD CONTROLNET (Structure Lock)
        print("[*] Loading ControlNet (Canny Edge Detector)...")
        controlnet = ControlNetModel.from_pretrained(
            "lllyasviel/sd-controlnet-canny", 
            torch_dtype=torch.float16 if device == "cuda" else torch.float32
        )
        
        # LOAD REALISTIC VISION V5.1 (The "Face Saver")
        model_id = "SG161222/Realistic_Vision_V5.1_noVAE"
        print(f"[*] Loading High-Fidelity Model: {model_id}...")
        
        pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
            model_id,
            controlnet=controlnet,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            safety_checker=None
        ).to(device)
        
        # Optimization for local VRAM
        pipe.enable_attention_slicing()
        
        # PREPARE IMAGE & UPSCALE
        print("[*] Loading and Upscaling Image...")
        original_img = Image.open(image_path).convert("RGB")
        w, h = original_img.size
        
        # Upscale 1.2x for better face details
        target_w = int(w * 1.2)
        target_h = int(h * 1.2)
        
        # Snap to multiples of 8
        target_w = (target_w // 8) * 8
        target_h = (target_h // 8) * 8
        
        img_input = original_img.resize((target_w, target_h), Image.Resampling.LANCZOS)
        
        # EDGE DETECTION (OpenCV)
        print("[*] Extracting structural wireframe...")
        image_cv = np.array(img_input)
        # Thresholds: 50/200 are good for general detail
        canny_image = cv2.Canny(image_cv, 50, 200)
        canny_image = canny_image[:, :, None]
        canny_image = np.concatenate([canny_image, canny_image, canny_image], axis=2)
        control_image = Image.fromarray(canny_image)
        
        # LAUNDERING
        print(f"[*] AI Laundering (Strength: {strength})...")
        
        prompt = "RAW photo, subject, 8k uhd, dslr, soft lighting, high quality, film grain, Fujifilm XT3"
        # The optimized, short negative prompt (Under 77 tokens)
        negative_prompt = "cgi, 3d, cartoon, anime, text, bad anatomy, deformed iris, deformed pupils, extra fingers, mutated hands, poorly drawn face, mutation, disfigured, blurry, missing limbs, fused fingers, ugly, long neck"
        
        # Suppress progress bar slightly for cleaner CLI
        laundered_img = pipe(
            prompt=prompt,
            image=img_input,
            control_image=control_image,
            strength=strength, 
            controlnet_conditioning_scale=1.0,
            guidance_scale=5.0, 
            negative_prompt=negative_prompt
        ).images[0]
        
        # ANALOG HUMANIZATION
        print("[*] Applying Analog Simulation...")
        # Resize back to original dimensions
        laundered_img = laundered_img.resize((w, h), Image.Resampling.LANCZOS)
        
        r, g, b = laundered_img.split()
        
        # Chromatic Aberration
        r = Image.fromarray(np.roll(np.array(r), -1, axis=1)) 
        b = Image.fromarray(np.roll(np.array(b), 1, axis=1))
        humanized_img = Image.merge("RGB", (r, g, b))
        
        # Film Grain
        img_array = np.array(humanized_img).astype(np.float32)
        noise = np.random.normal(0, 4, img_array.shape).astype(np.float32)
        final_img = Image.fromarray(np.clip(img_array + noise, 0, 255).astype(np.uint8))

        # SAVE
        print(f"[*] Stripping Metadata & Saving to {output_filename}...")
        final_img.save(output_filename, "JPEG", quality=95, optimize=True)
        print(f"[+] SUCCESS! Process complete.")

    except Exception as e:
        print(f"[-] Critical Error: {e}")
        print("    Note: First run requires internet to download models (~4GB).")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Neural-Bleach Ultimate: AI Watermark Remover")
    parser.add_argument("image", help="Path to the image file")
    parser.add_argument("--out", help="Output filename", default=None)
    parser.add_argument("--strength", type=float, default=0.20, help="Laundering strength (default: 0.20)")
    
    args = parser.parse_args()
    
    stego_washer_ultimate(args.image, args.out, args.strength)
