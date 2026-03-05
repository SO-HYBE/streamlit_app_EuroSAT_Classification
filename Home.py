import streamlit as st


st.set_page_config(
    page_title="EuroSAT Classification Project",
    page_icon="🌍",
    layout="wide"
)

st.title("🌍 EuroSAT Land Cover Classification")
st.subheader("A deep learning comparative study")

st.markdown("""
Welcome to the interactive dashboard for the EuroSAT classification project! 

This application allows you to evaluate and compare custom-built neural networks (built entirely from scratch) against industry-standard TensorFlow architectures.

### 👈 Navigation
Please use the sidebar to navigate through the app:
* **Live Comparison:** Run real-time inference using our trained models on unseen satellite imagery.
* **Model Analysis:** Dive deep into the training metrics, loss curves, and confusion matrices.
* **About:** Learn more about the dataset, the architectures, and the methodology behind this project.
""")

st.info("Select **Live Comparison** from the sidebar to get started!")

