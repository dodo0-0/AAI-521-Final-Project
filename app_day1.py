# app_day1.py 
import streamlit as st
from diffusers import StableDiffusionInpaintPipeline
import torch
from PIL import Image
import os

# Load model (once at startup)
@st.cache_resource
def load_model():
    pipe = StableDiffusionInpaintPipeline.from_pretrained("runwayml/stable-diffusion-inpainting")
    return pipe.to("cuda" if torch.cuda.is_available() else "cpu")

pipe = load_model()

# Denoise function
def denoise_image(noisy_image):
    if isinstance(noisy_image, np.ndarray):
        noisy_image = Image.fromarray(noisy_image)
    noisy_image = noisy_image.convert('RGB')
    mask = Image.new('L', noisy_image.size, 255)
    result = pipe(
        prompt="high quality restored old photo, clear, sharp, detailed",
        image=noisy_image,
        mask_image=mask,
        strength=0.8
    ).images[0]
    return result

# Streamlit UI
st.title("AAI-521 Extra Credit: Old Photo Denoising")
st.write("Upload a noisy old photo to restore it!")

uploaded_file = st.file_uploader("Choose a noisy image...", type=["jpg", "png", "jpeg"])
if uploaded_file is not None:
    noisy_image = Image.open(uploaded_file).convert('RGB')
    st.image(noisy_image, caption="Uploaded Noisy Photo", use_column_width=True)
    
    if st.button("Restore Photo"):
        with st.spinner("Restoring..."):
            restored_image = denoise_image(noisy_image)
            st.image(restored_image, caption="Restored Result", use_column_width=True)
            st.download_button("Download Restored Image", restored_image.to_bytes(), file_name="restored_day1.png")

# Optional: Load sample from dataset
if st.button("Test with Sample Photo"):
    sample_noisy = Image.open("dataset/damaged_noisy/photo_00.jpg")
    st.image(sample_noisy, caption="Sample Noisy Photo", use_column_width=True)
    with st.spinner("Restoring Sample..."):
        restored_sample = denoise_image(sample_noisy)
        st.image(restored_sample, caption="Restored Sample", use_column_width=True)
