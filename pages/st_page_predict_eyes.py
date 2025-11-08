import streamlit as st
from utils.stream_function import stream_data
from utils.load_images import load_random_images, display_images
from utils.predict import load_model, DiabetiCNN, prediction
import pandas as pd
import torch
from pathlib import Path

# Use Streamlit secrets instead of dotenv
device = "cuda:0" if torch.cuda.is_available() else "cpu"

# Access secrets from .streamlit/secrets.toml
try:
    MODEL_PATH = Path(st.secrets["MODEL_PATH"])
except KeyError:
    st.error("MODEL_PATH not found in secrets.toml. Please add it to .streamlit/secrets.toml")
    st.stop()

if not MODEL_PATH.exists():
    st.error(f"Model file not found at: {MODEL_PATH}")
    st.stop()
    
model = load_model(DiabetiCNN(), device=device, path=MODEL_PATH)

st.set_page_config(
    page_title="Predict_eyes" 
)

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

st.write("test:", st.secrets["MODEL_PATH"])