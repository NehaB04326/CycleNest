import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import warnings

# Suppress TensorFlow warnings
warnings.filterwarnings("ignore")

# Custom function to safely load model with version compatibility
@st.cache_resource(show_spinner=False)
def load_model(model_path):
    try:
        # Attempt standard loading first
        model = tf.keras.models.load_model(model_path)
        return model
    except Exception as e:
        st.error(f"Initial loading failed: {str(e)}")
        try:
            # Attempt with experimental feature
            model = tf.keras.models.load_model(
                model_path,
                custom_objects=None,
                compile=True,
                options=tf.saved_model.LoadOptions(
                    experimental_io_device='/job:localhost'
                )
            )
            return model
        except Exception as e:
            st.error(f"Model loading failed completely: {str(e)}")
            return None

# Configuration
IMG_HEIGHT, IMG_WIDTH = 224, 224
MODEL_PATH = "pcos_ultrasound_model.h5"

# Load model
model = load_model(MODEL_PATH)

if model is None:
    st.error("Failed to load model. Please check the model file and try again.")
    st.stop()

def preprocess_image(image_file):
    """Preprocess uploaded image for model prediction"""
    try:
        img = Image.open(image_file).convert("RGB")
        img = img.resize((IMG_WIDTH, IMG_HEIGHT))
        img_array = np.array(img).astype("float32") / 255.0
        return np.expand_dims(img_array, axis=0)
    except Exception as e:
        st.error(f"Image processing error: {str(e)}")
        return None

# UI Styling
st.markdown("""
<style>
:root {
    --primary: #d81e77;
    --secondary: #ff9ad8;
    --light: #fff0f5;
    --dark: #5a1457;
    --success: #22863a;
    --warning: #b0003a;
}

.main {
    background: linear-gradient(135deg, #ffcee9 0%, #ff9ad8 100%) !important;
}

.stApp {
    max-width: 750px;
    padding: 2rem;
    background: rgba(255, 240, 245, 0.8);
    border-radius: 20px;
    box-shadow: 0 10px 30px rgba(255, 105, 180, 0.3);
}

h1 {
    color: var(--primary);
    text-align: center;
    font-size: 2.5rem;
    margin-bottom: 0.5rem;
    text-shadow: 1px 1px 3px #ffb6c1;
}

.lead {
    text-align: center;
    color: var(--dark);
    font-size: 1.1rem;
    line-height: 1.6;
    max-width: 90%;
    margin: 0 auto 2rem;
}

.stFileUploader > label > div {
    background: var(--primary) !important;
    border-radius: 10px;
    padding: 0.7rem 1.3rem;
    transition: all 0.3s ease;
}

.result {
    font-size: 1.8rem;
    font-weight: 700;
    text-align: center;
    margin: 1.5rem 0;
}

.detected {
    color: var(--warning);
    text-shadow: 1px 1px 4px #ff6f91;
}

.normal {
    color: var(--success);
    text-shadow: 1px 1px 4px #7ee787;
}

.tips {
    background: #ffd6e8;
    border-left: 6px solid var(--primary);
    padding: 1rem;
    border-radius: 0 15px 15px 0;
    margin: 1rem 0;
}
</style>
""", unsafe_allow_html=True)

# App Interface
st.markdown('<h1>PCOS Ultrasound Analysis</h1>', unsafe_allow_html=True)
st.markdown("""
<p class="lead">
Polycystic Ovary Syndrome affects 1 in 5 women. Early detection through ultrasound 
analysis can help manage symptoms. Upload an ultrasound image below for analysis.
</p>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Choose an ultrasound image (JPG, PNG)", 
    type=["jpg", "jpeg", "png"],
    label_visibility="collapsed"
)

if uploaded_file is not None:
    try:
        # Display uploaded image
        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
        
        with st.spinner("Analyzing image..."):
            # Preprocess and predict
            img_array = preprocess_image(uploaded_file)
            if img_array is not None:
                prediction = model.predict(img_array)
                confidence = float(prediction[0][0])
                
                if confidence < 0.5:
                    result = "PCOS Characteristics Detected"
                    result_class = "detected"
                    tips = """
                    • Schedule a visit with your gynecologist\n
                    • Maintain a balanced diet rich in fiber\n
                    • Engage in regular physical activity\n
                    • Monitor your symptoms regularly
                    """
                else:
                    result = "Normal Scan Detected"
                    result_class = "normal"
                    tips = """
                    • Maintain a healthy lifestyle\n
                    • Schedule annual check-ups\n
                    • Monitor for any changes in symptoms
                    """
                
                st.markdown(
                    f'<p class="result {result_class}">{result}</p>', 
                    unsafe_allow_html=True
                )
                st.metric("Prediction Confidence", f"{max(confidence, 1-confidence):.1%}")
                
                with st.expander("Recommended Actions", expanded=True):
                    st.markdown(f'<p class="tips">{tips}</p>', unsafe_allow_html=True)
                
    except Exception as e:
        st.error(f"An error occurred during analysis: {str(e)}")
else:
    st.info("Please upload an ultrasound image to begin analysis.")


