import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load TFLite Model
@st.cache_resource
def load_model():
    interpreter = tf.lite.Interpreter(model_path="model.tflite")
    interpreter.allocate_tensors()
    return interpreter

interpreter = load_model()

# Get Model Input & Output Details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Debugging: Print expected input shape
input_shape = input_details[0]['shape']  # Should be [1, 800, 800, 3]
st.write(f"Model Expected Input Shape: {input_shape}")  # Debugging step

# Define Classes
CLASSES = ["Cotton Blossom", "Cotton Bud", "Early Boll", "Matured Cotton Boll", "Split Cotton Boll"]

# Image Preprocessing (Resize to 800x800)
def preprocess_image(image, target_size):
    image = image.resize(target_size)  # Resize to match model input size
    image = np.array(image, dtype=np.float32) / 255.0  # Normalize to [0,1] range
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Streamlit UI
st.title("Cotton Growth Stage Classifier 🌱")
st.write("Upload an image of a cotton plant to classify its stage.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Resize to expected input size (800x800)
    target_size = (input_shape[1], input_shape[2])  # Extract model's expected size
    input_tensor = preprocess_image(image, target_size)
    
    interpreter.set_tensor(input_details[0]['index'], input_tensor)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])

    # Get Prediction
    prediction_idx = np.argmax(output_data)
    st.subheader(f"Prediction: **{CLASSES[prediction_idx]}** 🎯")
