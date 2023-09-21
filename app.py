from pathlib import Path
import streamlit as st
import config
from utils import load_model, infer_uploaded_video, infer_uploaded_image, infer_uploaded_webcam

# main page heading
st.markdown("<h1 style='text-align: center;'>VAMS - MobiNext</h1>", unsafe_allow_html=True)

# Author details
st.sidebar.markdown("<p style='text-align: center;'><strong>**MobiNext Technologies**</strong></p>", unsafe_allow_html=True)
st.sidebar.markdown("<div style='text-align: center;'>Task: Real-time object detection</div>", unsafe_allow_html=True)

# sidebar
st.sidebar.title("Model configuration")

# model options
task_type = st.sidebar.selectbox(
    "Select Task",
    ["Detection"]
    )

model_type = None
if task_type == "Detection":
    model_type = st.sidebar.selectbox(
        "Select Model",
        config.DETECTION_MODEL_LIST
    )
else:
    st.error("Currently only 'Detection' function is implemented")

confidence = float(st.sidebar.slider("Select Model Confidence", 30, 100, 50)) / 100


model_path = ""
if model_type:
    model_path = Path(config.DETECTION_MODEL_DIR, str(model_type))
else:
    st.error("Please Select Model in Sidebar")

# load pretrained DL model
try:
    with st.spinner("Loading model..."):
        model = load_model(model_path)
except Exception as e:
    st.error(f"Unable to load model. Please check the specified path: {model_path}")

# image/video options
st.sidebar.title("Input configuration")
source_selectbox = st.sidebar.selectbox(
    "Select Source",
    config.SOURCES_LIST
)

source_img = None
try:
    if source_selectbox == config.SOURCES_LIST[0]: # Image
        infer_uploaded_image(confidence, model)
    elif source_selectbox == config.SOURCES_LIST[1]: # Video
        infer_uploaded_video(confidence, model)
    elif source_selectbox == config.SOURCES_LIST[2]: # Webcam
        infer_uploaded_webcam(confidence, model)
    else:
        st.error("Currently only 'Image' and 'Video' source are implemented")
except Exception as e:
    st.error(f"Unable to perform inference: {e}")

# display copyright information and other relevant details
import streamlit as st

st.sidebar.markdown("""
<div style="text-align: center;">
    <div>Copyright &copy; 2023 Mobinext Technologies.</div>
    <div>All rights reserved.</div>
</div>
""", unsafe_allow_html=True)

# Add author details at the bottom
st.markdown("<br><br>", unsafe_allow_html=True)  # Create some space
st.markdown("<p style='text-align: center;'>Created by MobiNext Technologies</p>", unsafe_allow_html=True)
