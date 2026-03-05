import streamlit as st
import pickle
import tensorflow as tf
import pandas as pd
import os
from PIL import Image
import numpy as np
from models.inference_utils import predict_dnn, predict_cnn

st.set_page_config(
    page_title="Live Comparison"
)


@st.cache_resource
def load_all_models():
    models = {}
    
    with open('models/DNN_scratch_cpu.pkl', 'rb') as f:
        models['DNN-scratch'] = pickle.load(f)
    with open('models/CNN_scratch_cpu.pkl', 'rb') as f:
        models['CNN-scratch'] = pickle.load(f)
    
    models['SOTA-tf'] = tf.keras.models.load_model('models/SOTA_tf.h5')
    models['CNN-tf'] = tf.keras.models.load_model('models/CNN_tf.h5')
    models['DNN-tf'] = tf.keras.models.load_model('models/DNN_tf.h5')
    
    return models

models = load_all_models()

@st.cache_data
def load_test_data():
    df = pd.read_csv('data/test_labels.csv')
    return df

test_df = load_test_data()

st.title("🔬 Live Model Comparison")

col1, col2 = st.columns([1, 1])
with col1:
    class_filter = st.selectbox(
        "Filter by class:",
        ["All Classes", "AnnualCrop", "Forest", "HerbaceousVegetation", "Highway", "Industrial", "Pasture", "PermanentCrop", "Residential", "River", "SeaLake"]
    )
with col2:
    if 'difficulty' in test_df.columns:
        difficulty = st.selectbox("Difficulty:", ["All", "Easy", "Medium", "Hard"])
    else:
        difficulty = "All"

filtered_df = test_df.copy()

class_col = 'ClassName' if 'ClassName' in filtered_df.columns else 'Label'

filtered_df[class_col] = filtered_df[class_col].astype(str).str.strip()

if class_filter != "All Classes":
    filtered_df = filtered_df[filtered_df[class_col] == class_filter]

if 'last_class_filter' not in st.session_state:
    st.session_state.last_class_filter = class_filter
if 'last_difficulty' not in st.session_state:
    st.session_state.last_difficulty = difficulty

filters_changed = (st.session_state.last_class_filter != class_filter) or \
                  (st.session_state.last_difficulty != difficulty)

if filters_changed:
    st.session_state.last_class_filter = class_filter
    st.session_state.last_difficulty = difficulty

if class_filter != "All Classes":
    filtered_df = filtered_df[filtered_df[class_col] == class_filter]

if difficulty != "All" and 'difficulty' in filtered_df.columns:
    filtered_df = filtered_df[filtered_df['difficulty'] == difficulty]

sample_size = min(5, len(filtered_df))

if st.button("🔄 Get New Batch") or 'sample_indices' not in st.session_state or filters_changed:
    if sample_size > 0:
        st.session_state.sample_indices = filtered_df.sample(sample_size).index.tolist()
    else:
        st.session_state.sample_indices = []

if st.session_state.sample_indices:
    sample_images = filtered_df.loc[st.session_state.sample_indices]
else:
    sample_images = pd.DataFrame()
    st.warning("No images match the selected filters.")

st.subheader("Select models to compare:")
selected_models = st.multiselect(
    "Choose models:",
    ["DNN-scratch", "DNN-tf", "CNN-scratch", "CNN-tf", "SOTA-tf"],
    default=["CNN-scratch", "SOTA-tf"]
)

def predict(model, img, model_name):
    if "scratch" in model_name:
        img_resized = np.array(img.resize((64, 64))) / 255.0
        
        if "DNN" in model_name:
            X = img_resized.reshape(12288, 1)
            pred_class, probs = predict_dnn(model, X)
        else:
            X = img_resized.reshape(1, 64, 64, 3)
            pred_class, probs = predict_cnn(model, X)

        return int(np.squeeze(pred_class))
        
    else:
        if "SOTA" in model_name:
            img_resized = img.resize((224, 224))
        else:
            img_resized = img.resize((64, 64))
            
        img_array = tf.keras.preprocessing.image.img_to_array(img_resized)
        img_array = np.expand_dims(img_array, 0)
        preds = model.predict(img_array, verbose=0)
        return int(np.argmax(preds[0]))

if st.button("▶️ Classify Images") and not sample_images.empty:
    cols = st.columns(sample_size)
    class_names = ["AnnualCrop", "Forest", "HerbaceousVegetation", "Highway", "Industrial", "Pasture", "PermanentCrop", "Residential", "River", "SeaLake"]
    
    for idx, (col, row) in enumerate(zip(cols, sample_images.iterrows())):
        file_col = 'Filename' if 'Filename' in row[1] else 'filename'
        
        img_path = f"data/test_images/{os.path.basename(row[1][file_col])}"
        img = Image.open(img_path)
        
        with col:
            st.image(img, width=200)
            st.caption(f"True: {row[1][class_col]}")
            
            for model_name in selected_models:
                pred_idx = predict(models[model_name], img, model_name)
                pred_name = class_names[pred_idx]
                
                true_label = row[1][class_col]
                if isinstance(true_label, str):
                    if true_label in class_names:
                        true_label_idx = class_names.index(true_label)
                    else:
                        true_label_idx = -1
                else:
                    true_label_idx = true_label
                
                correct = pred_idx == true_label_idx
                
                if correct:
                    st.success(f"✅ {model_name}: {pred_name}")
                else:
                    st.error(f"❌ {model_name}: {pred_name}")