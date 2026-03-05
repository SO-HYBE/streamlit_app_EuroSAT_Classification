import streamlit as st
import pandas as pd
import os
import pickle

st.set_page_config(page_title="Model Analysis", page_icon="📊", layout="wide")
st.title("📊 Model Deep Dive")

model_name = st.sidebar.selectbox(
    "Select a model:",
    ["DNN-scratch", "DNN-tf", "CNN-scratch", "CNN-tf", "SOTA-tf"]
)

st.subheader(f"{model_name} Performance")

# --- 1. INTERACTIVE METRICS SECTION ---
st.markdown("### Training Metrics")

if model_name == 'SOTA-tf':
    st.info("💡 **Look at the curves:** The sudden shift in the graphs visualizes our two-phase training architecture, marking the transition from a frozen base model to full fine-tuning.")

col_loss, col_acc = st.columns(2)

# Determine the correct pickle file path
if "scratch" in model_name:
    history_path = f"models/{model_name.replace('-', '_')}_cpu.pkl"
else:
    history_path = f"models/{model_name.replace('-', '_')}_history.pkl"

train_loss, val_loss, train_acc, val_acc = [], [], [], []

try:
    with open(history_path, 'rb') as f:
        history_data = pickle.load(f)
        
    if "scratch" in model_name:
        train_loss = history_data.get('train_costs', [])
        val_loss = history_data.get('val_costs', [])
        train_acc = history_data.get('train_accs', [])
        val_acc = history_data.get('val_accs', [])
    else:
        train_loss = history_data.get('loss', [])
        val_loss = history_data.get('val_loss', [])

        train_acc = history_data.get('accuracy', history_data.get('acc', []))
        val_acc = history_data.get('val_accuracy', history_data.get('val_acc', []))
        
except FileNotFoundError:
    st.warning(f"⚠️ Could not load interactive data from {history_path}.")

with col_loss:
    st.markdown("**Loss Curve**")
    if train_loss:
        loss_df = pd.DataFrame({
            'Train Loss': train_loss,
            'Val Loss': val_loss if val_loss else [None]*len(train_loss)
        })
        st.line_chart(loss_df)
    else:
        curve_path = f"assets/training_curves/{model_name}_curves.png"
        if os.path.exists(curve_path):
            st.image(curve_path, width='stretch')
        else:
            st.info("No loss data or static image found.")

with col_acc:
    st.markdown("**Accuracy Curve**")
    if train_acc:
        acc_df = pd.DataFrame({
            'Train Accuracy': train_acc,
            'Val Accuracy': val_acc if val_acc else [None]*len(train_acc)
        })
        st.line_chart(acc_df)
    else:
        st.info("No interactive accuracy data found for this model yet.")

# --- 2. CONFUSION MATRIX SECTION ---
st.markdown("### Confusion Matrix")
cm_path = f"assets/confusion_matrices/{model_name}_cm.png"

col_cm1, col_cm2, col_cm3 = st.columns([1, 2, 1])
with col_cm2:
    if os.path.exists(cm_path):
        st.image(cm_path, width='stretch')
    else:
        st.warning(f"⚠️ Waiting for static image at: {cm_path}")

if model_name == 'SOTA-tf':
    st.markdown("### Grad-CAM Visualization")
    st.image('assets/sota-Grad-CAM.png', width='stretch')

# --- 3. MODEL STATISTICS ---
st.subheader("Model Statistics")

stats = {
    "DNN-scratch": {
        "Architecture": "Custom Dense → BatchNorm → ReLU",
        "Optimizer": "Adam (Custom Implementation)",
        "Train Time": "approx. 25 min",
        "Test Accuracy": "44.59%" 
    },
    "DNN-tf": {
        "Architecture": "Dense → BatchNorm → ReLU",
        "Optimizer": "Adam (TensorFlow)",
        "Train Time": "approx. 10 min",
        "Test Accuracy": "44.00%"
    },
    "CNN-scratch": {
        "Architecture": "Custom Conv2D → BatchNorm → ReLU → MaxPool (3-layers)",
        "Optimizer": "Adam (Custom Implementation)",
        "Train Time": "approx. 45 min",
        "Test Accuracy": "82.3%" 
    },
    "CNN-tf": {
        "Architecture": "Conv2D → BatchNorm → ReLU → MaxPool (6-layers)",
        "Optimizer": "Adam (TensorFlow)",
        "Train Time": "approx. 15 min",
        "Test Accuracy": "94.44%"
    },
    "SOTA-tf": {
        "Architecture": "Pre-trained EfficientNetB0",
        "Optimizer": "Adam (TensorFlow)",
        "Train Time": "approx. 20 min (Fine-tuning)",
        "Test Accuracy": "97.63%" 
    }
}

if model_name in stats:
    stats_df = pd.DataFrame([stats[model_name]])
    st.table(stats_df)
else:
    st.info("Statistics for this model have not been added yet.")

# --- 4. KEY INSIGHTS ---
insights = {
    "DNN-scratch": "✨ **Key Insight:** Flattening the 2D images into 1D arrays destroys local spatial relationships (like the natural shapes of buildings or rivers), which severely bottlenecks the model's ability to learn compared to a CNN.",
    "DNN-tf": "✨ **Key Insight:** While TensorFlow's graph execution makes training significantly faster than our scratch implementation, the fundamental architectural limitation of using Dense layers on image data remains.",
    "CNN-scratch": "✨ **Key Insight:** Building this convolutional network entirely from scratch using NumPy/CuPy revealed the massive mathematical overhead of spatial operations. Integrating custom Batch Normalization was absolutely essential.",
    "CNN-tf": "✨ **Key Insight:** TF's highly optimized CuDNN operations process the exact same mathematical convolutions as our scratch model but at a fraction of the time, highlighting the power of low-level hardware optimization.",
    "SOTA-tf": "✨ **Key Insight:** Utilizing transfer learning allowed the model to rapidly converge by leveraging general-purpose feature extractors trained on ImageNet, skipping the need to learn basic edges and textures from scratch."
}

if model_name in insights:
    st.info(insights[model_name])