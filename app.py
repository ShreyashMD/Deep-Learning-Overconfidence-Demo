import streamlit as st
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision
import numpy as np
import pandas as pd
import altair as alt
from PIL import Image
from model import SimpleCNN
import time

# --- Page Configuration ---
st.set_page_config(
    page_title="DL Overconfidence Demo",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Styling ---
st.markdown("""
<style>
    .reportview-container {
        background: #f0f2f6
    }
    .big-font {
        font-size:20px !important;
    }
    .stProgress .st-bo {
        background-color: #f63366;
    }
</style>
""", unsafe_allow_html=True)

# --- Load Model ---
@st.cache_resource
def load_model(model_path='cifar10_model.pth'):
    device = torch.device('cpu') 
    model = SimpleCNN()
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        return model
    except FileNotFoundError:
        return None

# --- Helper Functions ---
classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

def preprocess_image(image):
    # Resize to 32x32 as expected by the model
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    return transform(image).unsqueeze(0)

def get_prediction(model, image_tensor, temperature=1.0):
    with torch.no_grad():
        logits = model(image_tensor)
        # Apply temperature scaling
        scaled_logits = logits / temperature
        probs = F.softmax(scaled_logits, dim=1)
        conf, pred_class = torch.max(probs, 1)
        return pred_class.item(), conf.item(), probs.squeeze().tolist()

def load_random_test_image():
    # Load dataset just to get images (not efficient but simple for demo)
    try:
        testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False)
    except:
         # attempt download if not present
         testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True)
         
    idx = np.random.randint(0, len(testset))
    img, label = testset[idx]
    return img, label

# --- Main App ---

st.title("🤖 Deep Learning Overconfidence Demo")
st.markdown("""
This application demonstrates how **uncalibrated Deep Learning models** can be **extremely confident even when they are wrong**.
Key takeaway: **High Confidence $\\neq$ Correctness**.
""")

# Sidebar
st.sidebar.header("Configuration")
temperature = st.sidebar.slider(
    "Softmax Temperature (Visualization Only)", 
    min_value=0.5, 
    max_value=5.0, 
    value=1.0, 
    step=0.1,
    help="Adjusting temperature smoothes or sharpens the probability distribution without changing the model's learned weights. Note how confidence changes while the prediction rank stays same."
)

if temperature != 1.0:
    st.sidebar.info(f"Applying Temperature T={temperature}")

# Load Model
model = load_model()

if model is None:
    st.error("Model file `cifar10_model.pth` not found. Please wait for training to complete.")
    st.stop()

# Layout
col1, col2 = st.columns([1, 1.5])

with col1:
    st.header("1. Input")
    
    upload_option = st.radio("Choose Input Method", ["Upload Image", "Random Test Image (CIFAR-10)"])
    
    image = None
    true_label_idx = None
    
    if upload_option == "Upload Image":
        uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "png", "jpeg"])
        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert('RGB')
    else:
        if st.button("Load New Random Image"):
            image, true_label_idx = load_random_test_image()
            st.session_state['current_image'] = image
            st.session_state['current_label'] = true_label_idx
            
        if 'current_image' in st.session_state:
            image = st.session_state['current_image']
            true_label_idx = st.session_state.get('current_label')

    if image is not None:
        # Display Image (upscaled for visibility)
        st.image(image, caption='Input Image', width=300)
        
        # Ground Truth display if available
        if true_label_idx is not None:
            st.info(f"**Ground Truth Label:** {classes[true_label_idx].capitalize()}")

with col2:
    if image is not None:
        st.header("2. Analysis")
        
        # Inference
        img_tensor = preprocess_image(image)
        pred_idx, confidence, probs = get_prediction(model, img_tensor, temperature)
        pred_label = classes[pred_idx]
        
        # --- Metrics Display ---
        m_col1, m_col2 = st.columns(2)
        with m_col1:
            st.metric("Predicted Class", pred_label.capitalize())
        with m_col2:
            st.metric("Model Confidence", f"{confidence:.2%}")
        
        # --- Correctness Check ---
        is_correct = None
        if true_label_idx is not None:
            is_correct = (pred_idx == true_label_idx)
            if is_correct:
                st.success(f"✅ Correct Prediction! (True: {classes[true_label_idx]})")
            else:
                st.error(f"❌ Incorrect Prediction! (True: {classes[true_label_idx]})")
                
                # --- Overconfidence Warning ---
                if confidence >= 0.90:
                    st.warning(
                        "⚠️ **DANGER: OVERCONFIDENCE DETECTED**\n\n"
                        "The model is **>90% confident** but **WRONG**.\n"
                        "This is a classic example of an uncalibrated neural network."
                    )
        
        # --- Visualization ---
        st.subheader("Probability Distribution")
        
        # Prepare data for chart
        probs_df = pd.DataFrame({
            'Class': classes,
            'Probability': probs
        })
        
        # Top-K
        top_k = 5
        top_probs = probs_df.nlargest(top_k, 'Probability')
        
        # Color coding bar chart
        # If ground truth is known: Green for correct class, Red for predicted if wrong.
        # If ground truth unknown: Blue default.
        
        def get_color(row):
            if true_label_idx is not None:
                if row['Class'] == classes[true_label_idx]:
                    return 'green' # Truth
                if row['Class'] == pred_label and not is_correct:
                    return 'red' # Wrong High Confidence
            return 'steelblue'

        top_probs['Color'] = top_probs.apply(get_color, axis=1)
        
        chart = alt.Chart(top_probs).mark_bar().encode(
            x=alt.X('Probability', scale=alt.Scale(domain=[0, 1])),
            y=alt.Y('Class', sort='-x'),
            color=alt.Color('Color', scale=None),
            tooltip=['Class', alt.Tooltip('Probability', format='.2%')]
        ).properties(height=300)
        
        st.altair_chart(chart, use_container_width=True)
        
        # --- Educational Context ---
        st.markdown("---")
        st.markdown("#### Why is this happening?")
        with st.expander("Learn about Softmax Overconfidence"):
            st.write("""
            Standard Neural Networks trained with **Cross-Entropy Loss** are incentivized to push probabilities towards 1 (occupied by the correct class) and 0 (others).
            
            However, the model continues to minimize loss even after getting the classification correct, pushing the values (logits) to extremes.
            
            This results in **extremely high confidence scores** (often >99%) even for inputs that are slightly different from the training data or completely ambiguous, provided they fall into one of the decision boundaries.
            """)
    else:
        st.info("Upload an image or load a random test image to see the analysis.")

