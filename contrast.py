import streamlit as st
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt  
from streamlit_autorefresh import st_autorefresh
st_autorefresh(interval=600 * 1000, key="refresh")

# Streamlit App Title
st.title("Image Enhancement using Histogram Equalization & CLAHE")

# File uploader (accepts all image formats)
uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Convert to OpenCV format
    image = Image.open(uploaded_file)
    image = np.array(image)

    # Convert to Grayscale if it's not already
    if len(image.shape) == 3:
        gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray_image = image  # Already Grayscale

    # Histogram Equalization
    hist_equalized_image = cv2.equalizeHist(gray_image)

    # CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    clahe_image = clahe.apply(gray_image)

    # Display Images
    st.subheader("Processed Images")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.image(gray_image, caption="Original Image", use_container_width=True, channels="GRAY")
    
    with col2:
        st.image(hist_equalized_image, caption="Histogram Equalized Image", use_container_width=True, channels="GRAY")
    
    with col3:
        st.image(clahe_image, caption="CLAHE Enhanced Image", use_container_width=True, channels="GRAY")

    # Plot Histogram
    st.subheader("Histogram of Equalized Image")
    fig, ax = plt.subplots()
    ax.hist(hist_equalized_image.ravel(), bins=256, range=[0,256], color='red', alpha=0.7)
    ax.set_title("Histogram of Equalized Image")
    ax.set_xlabel("Pixel Value")
    ax.set_ylabel("Frequency")
    
    # Show the histogram plot
    st.pyplot(fig)
