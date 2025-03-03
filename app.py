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

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Debugging: Print expected input shape
input_shape = input_details[0]['shape']
st.write(f"Model Expected Input Shape: {input_shape}")  # Debugging step

# Preprocess the image
input_tensor = preprocess_image(image)

# Define Classes
CLASSES = ["Cotton Blossom", "Cotton Bud", "Early Boll", "Matured Cotton Boll", "Split Cotton Boll"]

# Image Preprocessing
def preprocess_image(image):
    image = image.resize((224, 224))  # Ensure the input size matches your model
    image = np.array(image, dtype=np.float32) / 255.0  # Normalize if required
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Streamlit UI
st.title("Cotton Growth Stage Classifier 🌱")
st.write("Upload an image of a cotton plant to classify its stage.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Preprocess & Predict
    input_tensor = preprocess_image(image)
    interpreter.set_tensor(input_details[0]['index'], input_tensor)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    
    # Get Prediction
    prediction_idx = np.argmax(output_data)
    st.subheader(f"Prediction: **{CLASSES[prediction_idx]}** 🎯")
