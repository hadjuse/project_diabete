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

# Use Streamlit secrets - access after set_page_config
device = "cuda:0" if torch.cuda.is_available() else "cpu"

# Get the project root directory (works locally and in cloud)
# When running from pages/, go up one level
if "pages" in str(Path(__file__).parent):
    PROJECT_ROOT = Path(__file__).parent.parent
else:
    PROJECT_ROOT = Path(__file__).parent

# Access secrets from .streamlit/secrets.toml or use relative paths
try:
    model_path_str = st.secrets["MODEL_PATH"]
    # If it's an absolute path, use it; otherwise make it relative to project root
    MODEL_PATH = Path(model_path_str) if Path(model_path_str).is_absolute() else PROJECT_ROOT / model_path_str
except KeyError:
    # Fallback to default relative path
    MODEL_PATH = PROJECT_ROOT / "model" / "drimdb_model.pth"
    st.warning(f"MODEL_PATH not in secrets, using default: {MODEL_PATH}")

if not MODEL_PATH.exists():
    st.error(f"Model file not found at: {MODEL_PATH}")
    st.error(f"Project root: {PROJECT_ROOT}")
    st.error(f"Current directory: {os.getcwd()}")
    st.stop()
    
model = load_model(DiabetiCNN(), device=device, path=MODEL_PATH)

st.title("Predict if eyes is diabetic")

# button_stream = st.button(label="Stream_test")

# text = """
# Je choisi de créer un monde rempli de vice
# """
# if button_stream:
#     st.write_stream(stream_data(text))
st.header("Sample eyes images")

images = load_random_images()

display_images(images)

uploaded_file = st.file_uploader(
    "Upload images eyes", type=['.png', '.jpg', '.jpeg', '.gif', '.bmp']
)

if uploaded_file:
    st.image(uploaded_file)
    
button_predict = st.button(label="Predict")
if button_predict:
    if uploaded_file is None:
        st.error("Please upload an image first!")
    else:
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

