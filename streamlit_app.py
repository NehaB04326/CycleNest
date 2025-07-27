import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load model once and cache it
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("pcos_ultrasound_model.h5")

model = load_model()

IMG_HEIGHT, IMG_WIDTH = 224, 224

def preprocess_image(image_file):
    img = Image.open(image_file).convert("RGB")
    img = img.resize((IMG_WIDTH, IMG_HEIGHT))
    img_array = np.array(img).astype("float32") / 255.0
    return np.expand_dims(img_array, axis=0)

# Stylish pink CSS
st.markdown(
    """
    <style>
    /* Background and fonts */
    body, .main {
        background: linear-gradient(135deg, #ffcee9 0%, #ff9ad8 100%);
        font-family: 'Poppins', sans-serif;
        color: #5a1457;
    }
    /* Container */
    .stApp {
        max-width: 700px;
        margin: 2rem auto;
        padding: 2rem;
        background: #fff0f5cc;
        border-radius: 20px;
        box-shadow: 0 10px 30px rgba(255, 105, 180, 0.3);
    }
    /* Title */
    h1 {
        font-weight: 700;
        font-size: 3rem;
        text-align: center;
        margin-bottom: 0.3rem;
        color: #d81e77;
        text-shadow: 1px 1px 3px #ffb6c1;
    }
    /* Lead paragraph */
    .lead {
        text-align: center;
        font-size: 1.25rem;
        margin-bottom: 2rem;
        font-weight: 500;
        color: #7a1f54;
        line-height: 1.6;
        max-width: 650px;
        margin-left: auto;
        margin-right: auto;
    }
    /* Upload button styling */
    .stFileUploader > label > div {
        background: #ff9ad8;
        color: white;
        border-radius: 10px;
        padding: 0.7rem 1.3rem;
        font-weight: 600;
        cursor: pointer;
        transition: background 0.3s ease;
    }
    .stFileUploader > label > div:hover {
        background: #d81e77;
    }
    /* Subheader */
    h2 {
        text-align: center;
        color: #d81e77;
        margin-bottom: 1.5rem;
        font-weight: 700;
    }
    /* Result text */
    .result {
        text-align: center;
        font-size: 1.8rem;
        margin-top: 1.5rem;
        font-weight: 700;
    }
    .result.detected {
        color: #b0003a;
        text-shadow: 1px 1px 4px #ff6f91;
    }
    .result.normal {
        color: #22863a;
        text-shadow: 1px 1px 4px #7ee787;
    }
    /* Confidence text */
    .confidence {
        text-align: center;
        font-size: 1.1rem;
        margin-top: 0.5rem;
        color: #7a1f54;
        font-weight: 600;
    }
    /* Tips box */
    .tips {
        max-width: 650px;
        background: #ffd6e8;
        border-left: 6px solid #d81e77;
        padding: 1rem 1.5rem;
        margin: 2rem auto;
        border-radius: 15px;
        color: #5a1457;
        font-weight: 600;
        font-size: 1.1rem;
        box-shadow: 1px 1px 7px #ffa3c4;
    }
    /* Uploaded image styling */
    .uploaded-image {
        border-radius: 15px;
        box-shadow: 0 8px 20px rgba(255, 105, 180, 0.4);
        margin-top: 1rem;
        margin-bottom: 1.5rem;
        display: block;
        margin-left: auto;
        margin-right: auto;
        max-width: 100%;
        height: auto;
    }
    /* Info message style */
    .stInfo {
        text-align: center;
        font-weight: 600;
        color: #9f5f81;
        margin-top: 2rem;
    }
    /* Error message style */
    .stError {
        text-align: center;
        font-weight: 600;
        color: #b00020;
        margin-top: 2rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Title and info
st.markdown('<h1>PCOS Awareness</h1>', unsafe_allow_html=True)
st.markdown(
    """
    <p class="lead">
    Polycystic Ovary Syndrome (PCOS) affects 1 in 5 women in India. Early detection can help manage symptoms effectively.
    Symptoms include irregular periods, acne, weight gain, and difficulty conceiving.
    But with the right support, PCOS can be managed!
    </p>
    """,
    unsafe_allow_html=True,
)

st.markdown('<h2>Check with Ultrasound</h2>', unsafe_allow_html=True)

uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image", output_format="PNG", use_container_width=True, clamp=True)

    with st.spinner("Analyzing image..."):
        try:
            img_array = preprocess_image(uploaded_file)
            prediction = model.predict(img_array)
            confidence = float(prediction[0][0])

            if confidence < 0.5:
                result = "⚠️ PCOS Detected"
                conf_display = 1.0 - confidence
                tips = "Eat a balanced diet, exercise regularly, reduce stress, and consult a gynecologist."
                st.markdown(f'<p class="result detected">{result}</p>', unsafe_allow_html=True)
            else:
                result = "✅ Normal"
                conf_display = confidence
                tips = "Keep up a healthy lifestyle and monitor your wellness."
                st.markdown(f'<p class="result normal">{result}</p>', unsafe_allow_html=True)

            st.markdown(f'<p class="confidence">Confidence: {conf_display:.2%}</p>', unsafe_allow_html=True)
            st.markdown(f'<div class="tips"><strong>Health Tip:</strong> {tips}</div>', unsafe_allow_html=True)
        except Exception as e:
            st.error(f"❌ Error during prediction: {e}")
else:
    st.info("Please upload an ultrasound image to get started.")
