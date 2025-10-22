# app_day1.py - Streamlit Denoising App for mybinder.org
import streamlit as st
from diffusers import StableDiffusionInpaintPipeline
import torch
from PIL import Image
import numpy as np
import os

# Page config
st.set_page_config(page_title="AAI-521 Old Photo Denoising", layout="wide")

st.title("🔥 AAI-521 Extra Credit: Old Photo Denoising")
st.write("Upload a noisy old photo to restore it! (Optimized for mybinder.org - CPU Only)")

# Lazy model load with error handling
pipe = None

@st.cache_resource
def load_model():
    global pipe
    if pipe is None:
        st.info("Loading model... (Takes ~1-2 mins, may fail—try local for full power)")
        try:
            pipe = StableDiffusionInpaintPipeline.from_pretrained(
                "runwayml/stable-diffusion-v1-5",
                torch_dtype=torch.float32,  # CPU-friendly
                local_files_only=False  # Allow online fetch
            )
            pipe = pipe.to("cpu")  # Force CPU
            return pipe
        except Exception as e:
            st.error(f"Model load failed: {e}. Using placeholder mode.")
            return None
    return pipe

# Load model on demand
if "model_loaded" not in st.session_state:
    load_model()
    st.session_state.model_loaded = True

# Denoise function
def denoise_image(noisy_image):
    if pipe is None:
        st.error("Model not loaded due to resource limits. Try local run.")
        return None
    if isinstance(noisy_image, np.ndarray):
        noisy_image = Image.fromarray(noisy_image)
    noisy_image = noisy_image.convert('RGB').resize((256, 256))  # Smaller size for Binder
    mask = Image.new('L', noisy_image.size, 255)
    try:
        result = pipe(
            prompt="restored old photo, basic quality",
            image=noisy_image,
            mask_image=mask,
            strength=0.7,
            num_inference_steps=15  # Minimal steps for speed
        ).images[0]
        return result
    except Exception as e:
        st.error(f"Processing error: {e} (Memory or Binder limit)")
        return None

# UI
col1, col2 = st.columns(2)
with col1:
    uploaded_file = st.file_uploader("Choose a noisy image...", type=["jpg", "png", "jpeg"])
    if uploaded_file:
        noisy_image = Image.open(uploaded_file)
        st.image(noisy_image, caption="Noisy Photo", use_column_width=True)
with col2:
    if st.button("Restore Photo") and uploaded_file:
        with st.spinner("Restoring... (May take 2-3 mins on Binder)"):
            result = denoise_image(noisy_image)
            if result:
                st.image(result, caption="Restored Result", use_column_width=True)
                st.download_button("Download", result.to_bytes("png"), "restored.png")

# Sample test (if dataset exists)
if os.path.exists("dataset/damaged_noisy/photo_00.jpg"):
    if st.button("Test Sample"):
        sample = Image.open("dataset/damaged_noisy/photo_00.jpg").resize((256, 256))
        with st.spinner("Testing..."):
            result = denoise_image(sample)
            if result:
                st.image(result, caption="Sample Restored")

st.warning("⚠️ Binder sessions last ~2 hours. Refresh URL to restart. For best results, run locally with GPU.")
