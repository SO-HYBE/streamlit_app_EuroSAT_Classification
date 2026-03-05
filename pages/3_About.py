import streamlit as st

st.set_page_config(page_title="About This Project", page_icon="ℹ️", layout="wide")

st.title("ℹ️ About This Project")

st.markdown("""
Welcome to the interactive companion dashboard for the research study: 
**"Comprehensive Study on EuroSAT Classification: From Scratch to State-of-the-Art"**
""")

st.divider()

# --- THE DATASET ---
st.header("🌍 The EuroSAT Dataset")
col_text, col_img = st.columns([2, 1])

with col_text:
    st.markdown("""
    This project utilizes the **EuroSAT** dataset, a benchmark land cover classification dataset based on Sentinel-2 satellite images. 
    
    It consists of 27,000 labeled and geo-referenced images divided into 10 distinct classes:
    * **Nature:** Forest, Herbaceous Vegetation, Pasture, River, Sea/Lake
    * **Agriculture:** Annual Crop, Permanent Crop
    * **Infrastructure:** Highway, Industrial, Residential
    
    The goal is to accurately classify the land use of a given geographic tile, which has major applications in urban planning, environmental monitoring, and agriculture.
    """)

# --- THE METHODOLOGY ---
st.header("🔬 Methodology & Architectural Progression")
st.markdown("""
The core of this research is a comparative analysis of different deep learning paradigms. Instead of just jumping to the best off-the-shelf model, this study builds from the ground up to demonstrate the mathematical and computational evolution of neural networks.
""")

expander1 = st.expander("Phase 1: Deep Neural Networks (DNN)", expanded=True)
with expander1:
    st.markdown("""
    * **DNN-Scratch:** A fully connected network built entirely from scratch using NumPy/CuPy. This phase highlights the foundational calculus of forward and backward propagation, and demonstrates how difficult it is to learn image features without spatial awareness.
    * **DNN-TF:** The same architecture implemented in TensorFlow to showcase the speedup gained from hardware-accelerated graph compilation.
    """)

expander2 = st.expander("Phase 2: Convolutional Neural Networks (CNN)", expanded=True)
with expander2:
    st.markdown("""
    * **CNN-Scratch:** A custom-built CNN demonstrating how spatial hierarchies and local receptive fields are extracted mathematically. This required writing custom inference utilities for spatial convolutions and pooling.
    * **CNN-TF:** The TensorFlow equivalent, utilizing highly optimized CuDNN routines to process the exact same mathematical operations at a fraction of the time with better accuracies.
    """)

expander3 = st.expander("Phase 3: State-of-the-Art (SOTA)", expanded=True)
with expander3:
    st.markdown("""
    * **Transfer Learning:** Utilizing a robust, pre-trained architecture (like EfficientNet) to achieve near-perfect accuracy by fine-tuning generalized feature extractors originally trained on ImageNet.
    """)

st.divider()

# --- ABOUT THE AUTHOR ---
st.header("👨‍💻 About the Author")
st.markdown("""
**Sohybe Amer** is an engineering student at Capital University with a strong focus on artificial intelligence, machine learning, and computer vision. 

This project was developed to bridge the gap between theoretical deep learning mathematics and practical, deployable AI engineering. By understanding the low-level calculus behind backpropagation and the high-level architecture of modern frameworks, this dashboard serves as a comprehensive exploration of neural network mechanics.
""")

st.divider()

# --- DOWNLOADS & LINKS ---
col_down, col_links = st.columns(2)

with col_down:
    st.subheader("📄 Research Paper")
    st.markdown("Download the full methodology, mathematical proofs, and results.")
    try:
        with open("assets/paper.pdf", "rb") as f:
            st.download_button(
                label="📥 Download Full Paper (PDF)", 
                data=f, 
                file_name="EuroSAT_Comparative_Study_paper.pdf",
                mime="application/pdf"
            )
    except FileNotFoundError:
        st.warning("⚠️ The paper PDF has not been uploaded to the `assets/` folder yet.")

with col_links:
    st.subheader("🔗 Connect & Explore")
    st.markdown("""
    * 💻 **GitHub:** [Repository & Source Code](https://github.com/SO-HYBE/Comprehensive-Study-on-EUROSAT-Classification)
    * 👔 **LinkedIn:** [Sohybe Amer](https://www.linkedin.com/in/sohybe-amer-4bba05b9/)
    """)