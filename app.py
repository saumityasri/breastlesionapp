import streamlit as st
import numpy as np
import cv2
import tensorflow as tf

# Load the pre-trained models
@st.cache_resource
def load_models():
    cnn_model = tf.keras.models.load_model(r"C:\Users\SAUMI\Downloads\cnn_model.h5")
    dnn_model = tf.keras.models.load_model(r"C:\Users\SAUMI\Downloads\csv_model.h5")
    multimodal_model = tf.keras.models.load_model(r"C:\Users\SAUMI\Downloads\final_multimodal_model.h5")
    return cnn_model, dnn_model, multimodal_model

cnn_model, dnn_model, multimodal_model = load_models()

# Function to preprocess the input image
def preprocess_image(image_file):
    image = cv2.imdecode(np.fromstring(image_file.read(), np.uint8), cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (128, 128))
    image = image.reshape(1, 128, 128, 1) / 255.0
    return image

# Function to preprocess the input clinical data
def preprocess_metadata(metadata):
    return np.array(metadata).reshape(1, -1)

# Streamlit App
st.title("Breast Lesion Diagnosis - Multimodal AI")
st.sidebar.title("Upload Inputs")

# Image Upload
st.sidebar.header("Upload Ultrasound Image")
image_file = st.sidebar.file_uploader("Choose an ultrasound image file", type=["png", "jpg", "jpeg"])

# Clinical Data Input
st.sidebar.header("Enter Clinical Data")
age = st.sidebar.number_input("Age", min_value=1, max_value=120, step=1)
tissue_type = st.sidebar.selectbox("Tissue Type", options=["Dense", "Fatty", "Mixed"])
symptom = st.sidebar.selectbox("Symptom", options=["None", "Nipple Discharge", "Breast Scar", "Other"])
birads = st.sidebar.slider("BI-RADS Category", min_value=0, max_value=6, step=1)

# Encode Clinical Data (dummy encoding)
metadata = [age, 1 if tissue_type == "Dense" else 0, 1 if symptom == "Nipple Discharge" else 0, birads]

# Prediction
if st.sidebar.button("Predict"):
    if image_file is not None:
        # Preprocess Image
        image = preprocess_image(image_file)

        # Preprocess Metadata
        metadata = preprocess_metadata(metadata)

        # Generate Predictions
        cnn_features = cnn_model.predict(image)
        dnn_features = dnn_model.predict(metadata)
        combined_features = np.concatenate([cnn_features, dnn_features], axis=1)
        prediction = multimodal_model.predict(combined_features)
        prediction_class = np.argmax(prediction)

        # Display Results
        st.subheader("Prediction Results")
        st.write(f"Prediction: {'Malignant' if prediction_class == 1 else 'Benign'}")
        st.write(f"Confidence: {np.max(prediction) * 100:.2f}%")
    else:
        st.warning("Please upload an ultrasound image for prediction.")

# Footer
st.sidebar.markdown("---")
st.sidebar.write("**Developed by [Saumitya Srivastava]**")
