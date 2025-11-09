import streamlit as st
from utils.stream_function import stream_data
from utils.load_images import load_random_images, display_images
from utils.predict import load_model, DiabetiCNN, prediction
import pandas as pd
import torch
from pathlib import Path
import os

st.set_page_config(
    page_title="Predict_eyes" 
)

device = "cuda:0" if torch.cuda.is_available() else "cpu"

if "pages" in str(Path(__file__).parent):
    PROJECT_ROOT = Path(__file__).parent.parent
else:
    PROJECT_ROOT = Path(__file__).parent

try:
    model_path_str = st.secrets["MODEL_PATH"]

    MODEL_PATH = Path(model_path_str) if Path(model_path_str).is_absolute() else PROJECT_ROOT / model_path_str
except KeyError:

    MODEL_PATH = PROJECT_ROOT / "model" / "drimdb_model.pth"

if not MODEL_PATH.exists():
    st.error(f"Model file not found at: {MODEL_PATH}")
    st.error(f"Project root: {PROJECT_ROOT}")
    st.error(f"Current directory: {os.getcwd()}")
    st.stop()
    
model = load_model(DiabetiCNN(), device=device, path=MODEL_PATH)

st.title("Predict if eyes is diabetic")


st.header("Sample eyes images")

images = load_random_images()

display_images(images)

# File uploader with size limit
uploaded_file = st.file_uploader(
    "Upload images eyes", 
    type=['png', 'jpg', 'jpeg', 'gif', 'bmp'],
    help="Maximum file size: 10MB. For best results, use clear retinal images."
)

if uploaded_file:
    file_size_mb = uploaded_file.size / (1024 * 1024)
    if file_size_mb > 10:
        st.error(f"File too large: {file_size_mb:.1f}MB. Please upload an image smaller than 10MB.")
    else:
        st.image(uploaded_file, caption=f"Uploaded: {uploaded_file.name} ({file_size_mb:.2f}MB)")
    
button_predict = st.button(label="Predict")
if button_predict:
    if uploaded_file is None:
        st.error("Please upload an image first!")
    else:
        try:
            with st.spinner("Analyzing image..."):
                result = prediction(model=model, image=uploaded_file, device=device)
            
            st.success(f"**Prediction:** {result['prediction']}")
            st.metric("Confidence", f"{result['confidence']:.2%}")
            
            # Display probabilities
            st.subheader("Probabilities:")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Good (Non-Diabetic)", f"{result['probabilities']['Good']:.2%}")
            with col2:
                st.metric("Bad (Diabetic)", f"{result['probabilities']['Bad']:.2%}")
        except Exception as e:
            st.error(f"Error during prediction: {str(e)}")
            st.error("Please try with a different image or ensure it's a valid eye/retinal image.")

