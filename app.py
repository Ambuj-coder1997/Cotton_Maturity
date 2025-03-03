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

# Decode YOLO Output (Fixed)
def decode_yolo_output(output, conf_threshold=0.3):  # Lowered threshold to 0.3
    boxes, scores, class_ids = [], [], []
    output = np.squeeze(output)  # Remove batch dimension if necessary

    st.write("Raw Model Output (Debugging):")
    for i, det in enumerate(output):
        st.write(f"Detection {i}: {det}")

    for det in output:
        confidence = det[4]  # Confidence score
        if confidence < conf_threshold:  # Skip low-confidence detections
            continue

        x, y, w, h = det[:4]  # Bounding box
        class_probs = det[5:]  # Class scores (YOLOv8 format)
        class_id = np.argmax(class_probs)  # Get class with highest probability

        if class_id < 0 or class_id >= len(CLASSES):  # Ensure valid class ID
            continue

        # Convert YOLO format to (x1, y1, x2, y2) using 640 instead of image_shape
        x1 = max(0, int((x - w / 2) * 640))
        y1 = max(0, int((y - h / 2) * 640))
        x2 = min(640, int((x + w / 2) * 640))
        y2 = min(640, int((y + h / 2) * 640))

        boxes.append([x1, y1, x2, y2])
        scores.append(float(confidence))
        class_ids.append(class_id)

    return boxes, scores, class_ids

# Draw Bounding Boxes on Image (Fixed)
def draw_boxes(image, boxes, scores, class_ids):
    draw = ImageDraw.Draw(image)
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box

        if class_ids[i] < 0 or class_ids[i] >= len(CLASSES):  # Safety check
            continue

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
    st.image(image, caption="Uploaded Image", use_container_width=True)
    
    # Preprocess & Predict
    input_tensor = preprocess_image(image)
    input_tensor = np.array(input_tensor, dtype=np.float32)
    
    # Run Model
    outputs = session.run(None, {"images": input_tensor})

    # Decode YOLO Output
    boxes, scores, class_ids = decode_yolo_output(outputs[0])

    # Debugging Output
    st.write(f"Model Output Shape: {outputs[0].shape}")
    st.write(f"Boxes: {boxes}, Scores: {scores}, Class IDs: {class_ids}")
    st.write(f"Number of Classes: {len(CLASSES)}")

    if not boxes:
        st.error("No valid objects detected. Try uploading a clearer image.")

    # Draw Bounding Boxes
    detected_image = draw_boxes(image, boxes, scores, class_ids)
    st.image(detected_image, caption="Detected Objects", use_container_width=True)

    # Display Predictions
    for i, class_id in enumerate(class_ids):
        if class_id < len(CLASSES):  # Ensure valid class ID
            st.subheader(f"Detected: **{CLASSES[class_id]}** with confidence {scores[i]:.2f} 🎯")
