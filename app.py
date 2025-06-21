import streamlit as st
import torch
import numpy as np
from torchvision.utils import make_grid
import matplotlib.pyplot as plt

# Import the Generator from model.py
from model import Generator

# --- App Configuration ---
st.set_page_config(
    page_title="Handwritten Digit Generator",
    layout="wide"
)

# --- Model Loading ---
# NOTE: Make sure the 'generator.pth' file is in the same directory as this 'app.py' file.
# You must first run 'train.py' in Google Colab and download the resulting 'generator.pth'.
MODEL_PATH = "generator.pth"
LATENT_DIM = 100
NUM_CLASSES = 10

# Use caching to load the model only once
@st.cache_resource
def load_model():
    # Check if the model file exists
    try:
        model = Generator(latent_dim=LATENT_DIM, num_classes=NUM_CLASSES)
        model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
        model.eval() # Set model to evaluation mode
        return model
    except FileNotFoundError:
        return None

generator = load_model()

# --- Helper Functions ---
def generate_images(digit, num_images=5):
    if generator is None:
        return None
    
    # Prepare latent vectors (noise) and labels
    noise = torch.randn(num_images, LATENT_DIM)
    labels = torch.LongTensor([digit] * num_images)
    
    # Generate images
    with torch.no_grad():
        generated_imgs = generator(noise, labels)
    
    # Post-process for display: normalize and convert to numpy
    generated_imgs = generated_imgs.detach().cpu().numpy()
    generated_imgs = 0.5 * generated_imgs + 0.5 # Denormalize from [-1, 1] to [0, 1]
    return generated_imgs

# --- UI Layout ---
st.title("Handwritten Digit Image Generator")
st.write("Generate synthetic MNIST-like images using your trained model. First, train the model using the `train.py` script in Google Colab, then place the `generator.pth` file here and run the app.")

if generator is None:
    st.error(f"**Model not found.** Please make sure the trained model file (`{MODEL_PATH}`) is in the same directory as the app.")
else:
    st.success("**Model loaded successfully.** Ready to generate digits.")

    st.header("Generate Images")
    
    col1, col2 = st.columns([1, 3])

    with col1:
        # User input: select a digit
        digit_to_generate = st.selectbox(
            label="Choose a digit to generate (0-9):",
            options=list(range(10))
        )

        # Generate button
        generate_button = st.button("Generate Images", type="primary")

    if generate_button:
        st.header(f"Generated images of digit: {digit_to_generate}")
        
        # Generate and display images
        images = generate_images(digit_to_generate, num_images=5)
        
        if images is not None:
            # Display the 5 images horizontally
            cols = st.columns(5)
            for i, image_np in enumerate(images):
                with cols[i]:
                    st.image(image_np.squeeze(), caption=f"Sample {i+1}", use_column_width=True)
        else:
            st.error("Could not generate images. Is the model loaded correctly?") 