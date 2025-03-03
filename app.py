import streamlit as st
import onnxruntime as ort
import numpy as np
import cv2
from PIL import Image, ImageDraw

# Load ONNX Model
@st.cache_resource
def load_model():
    return ort.InferenceSession("best.onnx")  # Load your ONNX model

session = load_model()

# Define Class Labels (Modify based on your dataset)
CLASSES = ["Cotton Blossom", "Cotton Bud", "Early Boll", "Matured Cotton Boll", "Split Cotton Boll"]

# Image Preprocessing
def preprocess_image(image):
    image = image.resize((640, 640))  # Resize to model expected input
    image = np.array(image, dtype=np.float32) / 255.0  # Normalize
    image = np.transpose(image, (2, 0, 1))  # HWC to CHW format
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image.astype(np.float32)

# Decode YOLO Output
def decode_yolo_output(output, image_shape, conf_threshold=0.5):
    boxes, scores, class_ids = [], [], []
    output = np.squeeze(output)  # Remove batch dimension if necessary

    for det in output:
        confidence = det[4]  # Confidence score
        if confidence > conf_threshold:
            x, y, w, h = det[:4]  # Bounding box
            class_id = int(det[5])  # Class ID

            # Convert YOLO format to (x1, y1, x2, y2)
            x1 = int((x - w / 2) * image_shape[0])
            y1 = int((y - h / 2) * image_shape[1])
            x2 = int((x + w / 2) * image_shape[0])
            y2 = int((y + h / 2) * image_shape[1])

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
st.title("🚜 YOLOv8 Cotton Growth Stage Detector")
st.write("Upload an image of a cotton plant, and the model will detect its stage.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)  # Fixed deprecation warning
    
    # Preprocess & Predict
    input_tensor = preprocess_image(image)
    input_tensor = np.array(input_tensor, dtype=np.float32)
    
    # Run Model
    outputs = session.run(None, {"images": input_tensor})

    # Decode YOLO Output
    image_shape = image.size  # (width, height)
    boxes, scores, class_ids = decode_yolo_output(outputs[0], image_shape)

    # Debugging Output (Now after decoding)
    st.write(f"Model Output Shape: {outputs[0].shape}")
    st.write(f"Boxes: {boxes}, Scores: {scores}, Class IDs: {class_ids}")
    st.write(f"Number of Classes: {len(CLASSES)}")

    # Draw Bounding Boxes
    detected_image = draw_boxes(image, boxes, scores, class_ids)
    st.image(detected_image, caption="Detected Objects", use_container_width=True)

    # Display Predictions
    for i, class_id in enumerate(class_ids):
        st.subheader(f"Detected: **{CLASSES[class_id]}** with confidence {scores[i]:.2f} 🎯")
