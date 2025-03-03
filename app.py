import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image, ImageDraw

# Load TFLite Model
@st.cache_resource
def load_model():
    interpreter = tf.lite.Interpreter(model_path="last_pooja_float16.tflite")  # Change model if needed
    interpreter.allocate_tensors()
    return interpreter

interpreter = load_model()

# Get model input details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Debugging: Print model expected input shape
input_shape = input_details[0]['shape']
st.write(f"Model Expected Input Shape: {input_shape}")  # Debugging step

# Define Labels (Change this based on your metadata.yaml file)
CLASSES = ["Cotton Blossom", "Cotton Bud", "Early Boll", "Matured Cotton Boll", "Split Cotton Boll"]

# Image Preprocessing
def preprocess_image(image):
    image = image.resize((input_shape[1], input_shape[2]))  # Resize to model expected size
    image = np.array(image, dtype=np.float32) / 255.0  # Normalize
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Function to Decode YOLOv8 Output
def decode_yolo_output(output_data, image_shape, conf_threshold=0.5):
    """
    Decodes YOLOv8 output into bounding boxes, class IDs, and confidence scores.
    """
    boxes, scores, class_ids = [], [], []

    for i in range(output_data.shape[1]):  # Iterate over detections
        confidence = output_data[0, 4, i]  # Confidence score
        if confidence > conf_threshold:
            x, y, w, h = output_data[0, 0:4, i]  # Bounding box
            class_id = np.argmax(output_data[0, 5:, i])  # Class ID

            # Convert YOLO box format (center_x, center_y, w, h) to (x1, y1, x2, y2)
            x1 = int((x - w / 2) * image_shape[1])
            y1 = int((y - h / 2) * image_shape[0])
            x2 = int((x + w / 2) * image_shape[1])
            y2 = int((y + h / 2) * image_shape[0])

            boxes.append([x1, y1, x2, y2])
            scores.append(float(confidence))
            class_ids.append(class_id)

    return boxes, scores, class_ids

# Draw Bounding Boxes on Image
def draw_boxes(image, boxes, scores, class_ids):
    draw = ImageDraw.Draw(image)
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box
        label = f"{CLASSES[class_ids[i]]}: {scores[i]:.2f}"
        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
        draw.text((x1, y1), label, fill="red")
    return image

# Streamlit UI
st.title("YOLOv8 Cotton Growth Stage Detector")
st.write("Upload an image of a cotton plant, and the model will detect its stage.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Preprocess & Predict
    input_tensor = preprocess_image(image)
    interpreter.set_tensor(input_details[0]['index'], input_tensor)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])

    # Debugging Output
    st.write(f"Model Output Shape: {output_data.shape}")
    st.write(f"First 10 Output Values: {output_data.flatten()[:10]}")  # Debugging

    # Decode YOLO Output
    image_shape = image.size  # (width, height)
    boxes, scores, class_ids = decode_yolo_output(output_data, image_shape)

    # Draw Bounding Boxes
    detected_image = draw_boxes(image, boxes, scores, class_ids)
    st.image(detected_image, caption="Detected Objects", use_column_width=True)

    # Display Predictions
    for i, class_id in enumerate(class_ids):
        st.subheader(f"Detected: **{CLASSES[class_id]}** with confidence {scores[i]:.2f} 🎯")
