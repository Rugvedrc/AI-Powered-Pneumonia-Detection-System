import streamlit as st
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import pickle
import pandas as pd
from PIL import Image
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import warnings

# Suppress TensorFlow warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')
warnings.filterwarnings('ignore')

# --- Page Configuration ---
st.set_page_config(
    page_title="AI Pneumonia Detection",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        margin-bottom: 2rem;
    }
    
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #667eea;
        color: #333;
    }
    
    .prediction-positive {
        background-color: #ffebee;
        border-left: 4px solid #f44336;
        padding: 1rem;
        border-radius: 8px;
        color: #333;
    }
    
    .prediction-negative {
        background-color: #e8f5e8;
        border-left: 4px solid #4caf50;
        padding: 1rem;
        border-radius: 8px;
        color: #333;
    }
    
    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 20px;
        border: none;
        padding: 0.5rem 2rem;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# --- Constants ---
MODEL_PATH = 'models/attention_model.keras'
THRESHOLD_PATH = 'models/attention_model_threshold.pkl'
SAMPLES_DIR = 'samples'
IMG_SIZE = (224, 224)
LAST_CONV_LAYER_NAME = "conv5_block16_concat"

# --- Cache expensive operations ---
@st.cache_resource
def load_model_and_threshold():
    try:
        # Load threshold
        with open(THRESHOLD_PATH, 'rb') as f:
            best_thresh = pickle.load(f)
        
        # Load model
        base_model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        
        # Create Grad-CAM model - Fixed input structure
        gradcam_model = tf.keras.models.Model(
            inputs=base_model.input,  # Fixed: Remove nested list structure
            outputs=[
                base_model.get_layer(LAST_CONV_LAYER_NAME).output,
                base_model.output
            ]
        )
        
        return base_model, gradcam_model, best_thresh
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None, None

# --- Fixed Grad-CAM functions ---
def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    try:
        with tf.GradientTape() as tape:
            conv_outputs, predictions = model(img_array)
            
            # Fixed: Handle predictions properly
            if isinstance(predictions, (list, tuple)):
                predictions = predictions[0]
            
            if pred_index is None:
                pred_index = tf.argmax(predictions[0])
            
            # Fixed: Extract scalar value properly
            if len(predictions.shape) > 1:
                class_channel = predictions[:, 0]  # Get first output column
            else:
                class_channel = predictions
                
            tape.watch(conv_outputs)

        grads = tape.gradient(class_channel, conv_outputs)
        
        # Fixed: Handle gradient shape
        if grads is None:
            return np.zeros((7, 7))
            
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        # Fixed: Handle conv_outputs indexing
        if isinstance(conv_outputs, (list, tuple)):
            conv_outputs = conv_outputs[0]
        else:
            conv_outputs = conv_outputs[0]  # Remove batch dimension
            
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-8)
        return heatmap.numpy()
        
    except Exception as e:
        print(f"Grad-CAM error details: {e}")
        return np.zeros((7, 7))

def create_gradcam_overlay(img, heatmap, alpha=0.4):
    try:
        heatmap_resized = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
        heatmap_uint8 = np.uint8(255 * heatmap_resized)
        heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        
        # Convert image to BGR if it's RGB
        if len(img.shape) == 3 and img.shape[2] == 3:
            img_bgr = cv2.cvtColor(np.uint8(img * 255), cv2.COLOR_RGB2BGR)
        else:
            img_bgr = np.uint8(img * 255)
            if len(img_bgr.shape) == 2:
                img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_GRAY2BGR)
        
        overlay = cv2.addWeighted(img_bgr, 1-alpha, heatmap_color, alpha, 0)
        return cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
    except Exception as e:
        st.error(f"Error creating overlay: {e}")
        return img

def load_and_preprocess_image(image_input):
    try:
        if isinstance(image_input, str):  # File path
            img = cv2.imread(image_input)
            if img is None:
                raise ValueError("Could not load image from path")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:  # PIL Image or numpy array
            img = np.array(image_input)
            if len(img.shape) == 3 and img.shape[2] == 3:
                pass  # Already RGB
            elif len(img.shape) == 2:  # Grayscale
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            else:
                raise ValueError("Unsupported image format")
        
        img_resized = cv2.resize(img, IMG_SIZE)
        img_array = np.expand_dims(img_resized / 255.0, axis=0)
        return img_resized, img_array
    except Exception as e:
        st.error(f"Error preprocessing image: {e}")
        return None, None

def get_sample_images():
    if os.path.exists(SAMPLES_DIR):
        return [f for f in os.listdir(SAMPLES_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    return []

# --- Main App ---
def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🫁 AI-Powered Pneumonia Detection System</h1>
        <h3>Uncertainty-Aware Attention-Based Deep Learning Model</h3>
    </div>
    """, unsafe_allow_html=True)

    # Project Description
    st.markdown("""
    ## 📋 About This Project
    
    This advanced AI system uses **uncertainty-aware attention mechanisms** combined with **Monte Carlo Dropout** 
    to detect pneumonia from chest X-ray images. Our model not only provides predictions but also quantifies 
    the uncertainty in its decisions, making it more reliable for medical applications.
    
    **Key Features:**
    - 🎯 **Attention Mechanism**: Focuses on the most relevant parts of X-ray images
    - 🔄 **Monte Carlo Dropout**: Provides uncertainty estimation for more reliable predictions  
    - 📊 **Grad-CAM Visualization**: Shows exactly where the model is looking
    - ⚡ **High Performance**: Achieves 96.5% accuracy with excellent sensitivity and specificity
    """)

    # Performance Metrics Table
    st.markdown("## 📊 Model Performance Metrics")
    
    metrics_data = {
        'Metric': [
            'Optimal Threshold',
            'Accuracy', 
            'Sensitivity (Recall)',
            'Specificity',
            'ROC-AUC'
        ],
        'Value': [
            '0.6446',
            '96.50%',
            '96.26%', 
            '97.16%',
            '99.42%'
        ],
        'What This Means': [
            'The number we use to decide if someone has pneumonia or not',
            'How often the AI gets the right answer out of 100 tries',
            'How good the AI is at finding pneumonia when it really exists', 
            'How good the AI is at saying someone is healthy when they really are healthy',
            'How well the AI can tell the difference between sick and healthy people'
        ]
    }
    
    df_metrics = pd.DataFrame(metrics_data)
    
    # Display metrics in a nice table
    st.dataframe(
        df_metrics,
        width='stretch',
        hide_index=True
    )

    # Load model
    base_model, gradcam_model, best_thresh = load_model_and_threshold()
    
    if base_model is None:
        st.error("❌ Failed to load model. Please check the model files.")
        return

    st.success(f"✅ Model loaded successfully! Optimal threshold: {best_thresh:.4f}")

    # Image Input Section
    st.markdown("## 🖼️ Pneumonia Detection")
    
    input_method = st.radio(
        "Choose input method:",
        ["Upload X-ray Image", "Select Sample Image"],
        horizontal=True
    )

    uploaded_image = None
    
    if input_method == "Upload X-ray Image":
        uploaded_file = st.file_uploader(
            "Upload a chest X-ray image", 
            type=['png', 'jpg', 'jpeg'],
            help="Supported formats: PNG, JPG, JPEG"
        )
        if uploaded_file:
            uploaded_image = Image.open(uploaded_file)
            
    else:  # Sample images
        sample_images = get_sample_images()
        if sample_images:
            selected_sample = st.selectbox(
                "Select a sample X-ray image:",
                sample_images,
                help="Choose from pre-loaded sample images"
            )
            if selected_sample:
                sample_path = os.path.join(SAMPLES_DIR, selected_sample)
                uploaded_image = sample_path
        else:
            st.warning("⚠️ No sample images found in the 'samples' folder.")

    # Prediction Section
    if uploaded_image is not None:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("### 📸 Original X-ray Image")
            if isinstance(uploaded_image, str):
                st.image(uploaded_image, width='stretch')
            else:
                st.image(uploaded_image, width='stretch')
        
        # Predict button
        if st.button("🔍 Analyze X-ray", type="primary"):
            with st.spinner("🧠 Analyzing X-ray image..."):
                try:
                    # Preprocess image
                    img, img_array = load_and_preprocess_image(uploaded_image)
                    
                    if img is None or img_array is None:
                        st.error("Failed to preprocess image. Please try a different image.")
                        return
                    
                    # Make prediction - Fixed model call
                    try:
                        conv_output, preds = gradcam_model(img_array)
                        
                        # Extract confidence score - Fixed indexing
                        if isinstance(preds, list):
                            confidence = float(preds[0][0, 0])
                        else:
                            confidence = float(preds[0, 0])
                            
                    except Exception as pred_error:
                        st.error(f"Prediction error: {pred_error}")
                        # Fallback to base model
                        preds = base_model.predict(img_array, verbose=0)
                        confidence = float(preds[0, 0])
                        conv_output = None

                    prediction = "PNEUMONIA" if confidence > best_thresh else "NORMAL"
                    
                    # Generate Grad-CAM if possible
                    if conv_output is not None:
                        heatmap = make_gradcam_heatmap(img_array, gradcam_model, LAST_CONV_LAYER_NAME)
                        overlay_img = create_gradcam_overlay(img, heatmap)
                        
                        with col2:
                            st.markdown("### 🔥 Grad-CAM Visualization")
                            st.image(overlay_img, width='stretch')
                            st.caption("Red regions show where the AI focused its attention")
                    else:
                        with col2:
                            st.markdown("### 🔥 Grad-CAM Visualization")
                            st.info("Grad-CAM visualization not available for this prediction")

                    # Results Section
                    st.markdown("## 📋 Analysis Results")
                    
                    # Create result cards
                    col3, col4, col5 = st.columns(3)
                    
                    with col3:
                        if prediction == "PNEUMONIA":
                            st.markdown(f"""
                            <div class="prediction-positive">
                                <h3>⚠️ PNEUMONIA DETECTED</h3>
                                <p><strong>Confidence:</strong> {confidence:.1%}</p>
                                <p>The AI model has detected signs consistent with pneumonia.</p>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown(f"""
                            <div class="prediction-negative">
                                <h3>✅ NORMAL</h3>
                                <p><strong>Confidence:</strong> {(1-confidence):.1%}</p>
                                <p>The AI model indicates the X-ray appears normal.</p>
                            </div>
                            """, unsafe_allow_html=True)
                    
                    with col4:
                        st.markdown("""
                        <div class="metric-card">
                            <h4>🎯 Model Decision</h4>
                            <p><strong>Threshold:</strong> {:.3f}</p>
                            <p><strong>Raw Score:</strong> {:.3f}</p>
                            <p><strong>Status:</strong> {} threshold</p>
                        </div>
                        """.format(
                            best_thresh, 
                            confidence, 
                            "Above" if confidence > best_thresh else "Below"
                        ), unsafe_allow_html=True)
                    
                        

                    # Confidence visualization
                    st.markdown("### 📊 Confidence Visualization")

                    fig = go.Figure(go.Indicator(
                        mode = "gauge+number+delta",
                        value = confidence,
                        domain = {'x': [0, 1], 'y': [0, 1]},
                        title = {'text': "Pneumonia Confidence Score"},
                        delta = {'reference': best_thresh},
                        gauge = {
                            'axis': {'range': [None, 1]},
                            'bar': {'color': "darkblue"},
                            'steps': [
                                {'range': [0, best_thresh], 'color': "lightgreen"},
                                {'range': [best_thresh, 1], 'color': "lightcoral"}
                            ],
                            'threshold': {
                                'line': {'color': "red", 'width': 4},
                                'thickness': 0.75,
                                'value': best_thresh
                            }
                        }
                    ))

                    fig.update_layout(height=300)
                    st.plotly_chart(fig, width='stretch')

                    # Add explanation below the meter
                    if prediction == "PNEUMONIA":
                        st.markdown(f"""
                        **🔍 What this means:** The AI model is {confidence:.0%} confident that this X-ray shows signs of pneumonia. 
                        Since this is above our threshold of {best_thresh:.1%}, the model predicts **PNEUMONIA**.
                        """)
                    else:
                        st.markdown(f"""
                        **🔍 What this means:** The AI model is {(1-confidence):.0%} confident that this X-ray appears normal. 
                        Since the pneumonia score of {confidence:.0%} is below our threshold of {best_thresh:.1%}, the model predicts **NORMAL**.
                        """)

                    # Important disclaimer
                    st.markdown("""
                    ---
                    ### ⚠️ Important Medical Disclaimer
                    
                    **This AI tool is for educational and research purposes only.** 
                    - This is **NOT** a substitute for professional medical diagnosis
                    - Always consult with qualified healthcare professionals
                    - Medical decisions should never be based solely on AI predictions
                    - This tool has not been approved by medical regulatory authorities
                    """)

                except Exception as e:
                    st.error(f"❌ Error during analysis: {str(e)}")
                    st.info("Please try with a different image or check the model files.")
                    # Debug information
                    st.write(f"Debug info: {type(e).__name__}: {e}")

    # Sidebar information
    with st.sidebar:
        st.markdown("## ℹ️ How It Works")
        st.markdown("""
        1. **Upload** a chest X-ray image
        2. **AI Analysis** using attention mechanisms  
        3. **Grad-CAM** highlights important regions
        4. **Results** with confidence scores
        """)
        
        st.markdown("## 🔧 Technical Details")
        st.markdown(f"""
        - **Model**: DenseNet121 with Attention
        - **Image Size**: {IMG_SIZE[0]}×{IMG_SIZE[1]} pixels
        - **Threshold**: {best_thresh if 'best_thresh' in locals() else 'Loading...'}
        - **Accuracy**: 96.5%
        - **Visualization**: Grad-CAM
        """)
        
        st.markdown("## 📚 About the Metrics")
        st.markdown("""
        - **Sensitivity**: How well we catch pneumonia cases
        - **Specificity**: How well we identify healthy patients  
        - **ROC-AUC**: Overall discrimination ability
        - **Grad-CAM**: Visual explanation of AI decisions
        """)

if __name__ == "__main__":
    main()