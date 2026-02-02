# 🧪 NeuralBleach

> **"If the pixels can dream, they can be made to forget."**

**NeuralBleach** is an advanced steganography removal suite designed to defeat robust, frequency-domain AI watermarks (like Google SynthID, Nano Banana, and invisible latent signatures).

Unlike traditional LSB tools that just "scrub bits," NeuralBleach uses **Generative Adversarial Laundering**—it forces the image through a secondary neural network (Stable Diffusion) to hallucinate a mathematically new pixel structure while preserving the visual semantic content.

---

## 🚀 Features

### 1. The "Launder" Protocol (Signal Disruption)

Uses **Stable Diffusion v1.5** (img2img) to "re-dream" the image.

* **Why:** Robust watermarks rely on specific pixel-grid coherence. By regenerating the image with a `strength` of 0.15, we force a new noise distribution that overwrites the original hidden signal.

### 2. The "Analog" Humanizer (Visual Spoofing)

Detectors don't just look for code; they look for "digital perfection." This module injects physical camera imperfections:

* **Chromatic Aberration:** Micro-shifts Red/Blue channels at the edges to simulate lens light refraction.
* **Film Grain Simulation:** Adds ISO-400 equivalent luminance noise to break the "AI smoothness."

### 3. Metadata Nuke

Automatically strips all **EXIF**, **XMP**, and **C2PA** (Content Credentials) headers, ensuring no digital signature remains in the file container.

---

## ⚡ Quick Start (Google Colab)

**Don't have a GPU?** Run Neural-Bleach in the cloud for free.

1. Open [Google Colab](https://colab.research.google.com/).
2. Change Runtime to **T4 GPU**.
3. Copy/Paste the script from `neural_bleach.ipynb`.
4. Run and upload your image.

---

## 🛠️ Local Installation

If you have a GPU (NVIDIA RTX 3060 or better recommended), you can run this locally.

### Prerequisites

```bash
pip install -r requirements.txt

```

### Usage

```python
from neural_bleach import stego_washer

# Basic Usage (Default Strength 0.15)
stego_washer("my_flagged_image.png")

# Aggressive Mode (If watermark persists)
stego_washer("my_flagged_image.png", strength=0.25, output_filename="clean_aggressive.jpg")

```

---

## 🧠 Technical Pipeline

| Stage | Process | Technical Goal |
| --- | --- | --- |
| **0. Pre-Processing** | `LANCZOS` Resizing | Break rigid grid alignment. |
| **1. Diffusion** | `SD v1.5 img2img` | Overwrite frequency-domain watermarks (DCT/Wavelet). |
| **2. Humanization** | `np.roll` + `Gaussian Noise` | Defeat visual "smoothness" classifiers. |
| **3. Sterilization** | `Image.save(quality=95)` | Strip C2PA/Metadata & Header signatures. |

---

## ⚠️ Disclaimer

**Educational Purposes Only.**
This tool is intended for cybersecurity research, CTF (Capture The Flag) competitions, and studying the robustness of digital watermarking technologies. The author is not responsible for misuse of this tool to bypass copyright protection or violate Terms of Service of generative AI platforms.

---

## 📜 License

Distributed under the MIT License. See `LICENSE` for more information.
