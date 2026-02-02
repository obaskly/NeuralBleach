import torch
import numpy as np
import argparse
import sys
import os
from diffusers import StableDiffusionImg2ImgPipeline
from PIL import Image

def stego_washer(image_path, output_filename=None, strength=0.15):
    if not os.path.exists(image_path):
        print(f"[-] Error: File not found: {image_path}")
        return

    print(f"\n--- PROCESSING: {image_path} ---")
    
    # Generate output filename if not provided
    if output_filename is None:
        base, ext = os.path.splitext(image_path)
        output_filename = f"{base}_bleached.jpg"

    # SETUP AI MODEL
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("WARNING: Running on CPU. This will be very slow. Consider using Google Colab.")

    print(f"Loading Neural Network (Stable Diffusion v1.5) on {device}...")
    
    # Use float16 for GPU to save VRAM, float32 for CPU
    dtype = torch.float16 if device == "cuda" else torch.float32
    
    try:
        pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5", 
            torch_dtype=dtype,
            safety_checker=None 
        ).to(device)
        # Enable memory optimizations for lower VRAM usage
        pipe.enable_attention_slicing()
    except Exception as e:
        print(f"- Model loading failed: {e}")
        print("    (Did you run 'pip install -r requirements.txt'?)")
        return

    # LOAD & PREPARE IMAGE
    try:
        original_img = Image.open(image_path).convert("RGB")
        w, h = original_img.size
        
        # Resize to nearest multiple of 8
        new_w = (w // 8) * 8
        new_h = (h // 8) * 8
        img_input = original_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        
        # LAUNDERING
        print(f"AI Laundering (Strength: {strength})...")
        prompt = "high quality, realistic, sharp focus, 8k photograph"
        negative_prompt = "blur, watermark, text, low quality, cartoon, distortion, ugly"
        
        # Suppress progress bar output for cleaner CLI
        laundered_img = pipe(
            prompt=prompt, 
            image=img_input, 
            strength=strength, 
            guidance_scale=7.5,
            negative_prompt=negative_prompt
        ).images[0]
        
        # ANALOG HUMANIZATION
        print("Applying Analog Simulation...")
        r, g, b = laundered_img.split()
        
        # Chromatic Aberration
        r = Image.fromarray(np.roll(np.array(r), -1, axis=1))
        b = Image.fromarray(np.roll(np.array(b), 1, axis=1))
        humanized_img = Image.merge("RGB", (r, g, b))
        
        # Film Grain
        img_array = np.array(humanized_img).astype(np.float32)
        noise = np.random.normal(0, 6, img_array.shape).astype(np.float32)
        final_img = Image.fromarray(np.clip(img_array + noise, 0, 255).astype(np.uint8))

        # SAVE
        final_img = final_img.resize((w, h), Image.Resampling.LANCZOS)
        print(f"Stripping Metadata & Saving to {output_filename}...")
        final_img.save(output_filename, "JPEG", quality=95, optimize=True)
        
        print(f"SUCCESS. Clean image: {output_filename}")

    except Exception as e:
        print(f"- Error during processing: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NeuralBleach: AI Watermark Removal Tool")
    parser.add_argument("image", help="Path to the image file")
    parser.add_argument("--out", help="Output filename", default=None)
    parser.add_argument("--strength", type=float, default=0.15, help="Laundering strength (0.1 - 0.3)")
    
    args = parser.parse_args()
    
    stego_washer(args.image, args.out, args.strength)
