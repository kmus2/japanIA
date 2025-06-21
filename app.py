import streamlit as st
import torch
import numpy as np
from torchvision.utils import make_grid
import matplotlib.pyplot as plt

# Import the Generator from model.py
from model import Generator

# --- Translations ---
TRANSLATIONS = {
    "en": {
        "page_title": "Handwritten Digit Generator",
        "sidebar_title": "Controls",
        "language_label": "Language",
        "header": "Handwritten Digit Generator",
        "subheader": "Generate synthetic MNIST-like images with a Conditional GAN model.",
        "usage_header": "How to use this app",
        "usage_body": "First, train the model by running the `train.py` script (preferably in Google Colab with a GPU). Then, place the resulting `generator.pth` file in the same directory as this app. Finally, select a digit and click the generate button!",
        "model_not_found": "Model not found.",
        "model_not_found_body": "Please make sure the trained model file (`generator.pth`) is in the same directory as the app.",
        "model_loaded": "Model loaded successfully.",
        "model_loaded_body": "Ready to generate digits.",
        "choose_digit_label": "Choose a digit to generate:",
        "generate_button": "Generate Images",
        "generating_info": "Generating images for digit",
        "generated_header": "Generated images of digit: {digit}",
        "sample_caption": "Sample {i}",
        "error_generating": "Could not generate images. Is the model loaded correctly?",
    },
    "ja": {
        "page_title": "手書き数字ジェネレーター",
        "sidebar_title": "コントロール",
        "language_label": "言語",
        "header": "手書き数字画像ジェネレーター",
        "subheader": "条件付きGANモデルで手書き数字の画像を生成します。",
        "usage_header": "このアプリの使い方",
        "usage_body": "まず、`train.py`スクリプトを実行してモデルをトレーニングします（GPUを備えたGoogle Colabを推奨）。次に、生成された`generator.pth`ファイルをこのアプリと同じディレクトリに配置します。最後に、数字を選択して生成ボタンをクリックしてください！",
        "model_not_found": "モデルが見つかりません。",
        "model_not_found_body": "訓練済みのモデルファイル (`generator.pth`) がアプリと同じディレクトリにあることを確認してください。",
        "model_loaded": "モデルは正常にロードされました。",
        "model_loaded_body": "数字を生成する準備ができました。",
        "choose_digit_label": "生成する数字を選択してください:",
        "generate_button": "画像を生成",
        "generating_info": "数字の画像を生成しています",
        "generated_header": "生成された数字の画像: {digit}",
        "sample_caption": "サンプル {i}",
        "error_generating": "画像を生成できませんでした。モデルは正しく読み込まれていますか？",
    },
}

# --- Session State Initialization ---
if 'lang' not in st.session_state:
    st.session_state.lang = 'en'

T = TRANSLATIONS[st.session_state.lang]

# --- App Configuration ---
st.set_page_config(
    page_title=T["page_title"],
    layout="wide",
    initial_sidebar_state="expanded"
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

# Sidebar
with st.sidebar:
    st.title("⚙️ " + T["sidebar_title"])

    lang_options = {"en": "English", "ja": "日本語"}
    
    # Determine the index of the current language for the radio button
    current_lang_index = list(lang_options.keys()).index(st.session_state.lang)

    selected_lang_key = st.radio(
        T["language_label"],
        options=list(lang_options.keys()),
        format_func=lambda key: lang_options[key],
        horizontal=True,
        key="language_toggle",  # Add a stable key to preserve state
        index=current_lang_index # Set index to sync with session state
    )

    if selected_lang_key != st.session_state.lang:
        st.session_state.lang = selected_lang_key
        st.rerun()
    
    st.markdown("---")

    digit_to_generate = st.selectbox(
        label=T["choose_digit_label"],
        options=list(range(10)),
        index=7 # Default to 7
    )

    generate_button = st.button(
        label=T["generate_button"],
        type="primary",
        use_container_width=True
    )

# Main Page
st.title("✨ " + T["header"])
st.markdown(f"*{T['subheader']}*")

with st.expander(T["usage_header"]):
    st.write(T["usage_body"])

if generator is None:
    st.error(f"**{T['model_not_found']}** {T['model_not_found_body']}", icon="❌")
else:
    st.success(f"**{T['model_loaded']}** {T['model_loaded_body']}", icon="✅")

    if generate_button:
        st.header(T["generated_header"].format(digit=digit_to_generate), anchor=False)
        
        with st.spinner(f"{T['generating_info']} **{digit_to_generate}**..."):
            images = generate_images(digit_to_generate, num_images=5)
        
        if images is not None:
            cols = st.columns(5)
            for i, image_np in enumerate(images):
                with cols[i]:
                    st.image(
                        image_np.squeeze(), 
                        caption=T["sample_caption"].format(i=i + 1),
                        use_container_width=True
                    )
            
            # Fun animation on success
            st.balloons()
        else:
            st.error(T["error_generating"], icon="🚨") 