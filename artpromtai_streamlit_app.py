
import streamlit as st
import torch
from diffusers import StableDiffusionPipeline
from transformers import CLIPTokenizer, CLIPTextModel
from PIL import Image
import numpy as np
import time
import io

# Set page configuration
st.set_page_config(
    page_title="ArtPromtAi - Text to Image Generator",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Cache the model loading to avoid reloading on each run
@st.cache_resource
def load_model():
    """Load Stable Diffusion model and tokenizer"""
    model_id = "runwayml/stable-diffusion-v1-5"

    # Check if CUDA is available
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load the pipeline
    if device == "cuda":
        pipe = StableDiffusionPipeline.from_pretrained(
            model_id, 
            torch_dtype=torch.float16,
            safety_checker=None,
            requires_safety_checker=False
        )
    else:
        pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            safety_checker=None,
            requires_safety_checker=False
        )

    pipe = pipe.to(device)

    # Load tokenizer separately for text analysis
    tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder")

    return pipe, tokenizer, text_encoder, device

@st.cache_data
def analyze_prompt(_tokenizer, prompt):
    """Analyze the input prompt and show tokenization"""
    tokens = _tokenizer(prompt, return_tensors="pt")
    token_ids = tokens["input_ids"][0].tolist()

    # Decode individual tokens
    individual_tokens = []
    for token_id in token_ids:
        token = _tokenizer.decode([token_id])
        individual_tokens.append(f"{token_id}: '{token}'")

    return {
        "total_tokens": len(token_ids),
        "token_breakdown": individual_tokens,
        "attention_mask": tokens["attention_mask"][0].tolist()
    }

def generate_image(pipe, prompt, **kwargs):
    """Generate image from text prompt"""
    with torch.autocast("cuda" if torch.cuda.is_available() else "cpu"):
        image = pipe(prompt, **kwargs).images[0]
    return image

def main():
    # Inject custom 3D gradient background CSS
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@700&family=Orbitron:wght@600&display=swap');
        body, .stApp {
            background: radial-gradient(ellipse 120% 100% at 50% 0%, #43cea2 0%, #6a5acd 60%, #e0e0e0 100%) !important;
            min-height: 100vh;
            font-family: 'Montserrat', 'Orbitron', Arial, sans-serif !important;
            color: #222;
        }
        .stApp {
            background-attachment: fixed;
            background-size: cover;
        }
        /* Remove black header bar when sidebar is closed */
        header[data-testid="stHeader"] {
            background: transparent !important;
        }
        /* Sidebar custom style */
        section[data-testid="stSidebar"] {
            background: linear-gradient(120deg, #43cea2 0%, #185a9d 100%);
            border-radius: 28px;
            box-shadow: 0 12px 48px 0 rgba(31, 38, 135, 0.37);
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
            border: 1.5px solid rgba(255, 255, 255, 0.22);
            margin: 20px 10px 20px 10px;
            padding: 32px 18px;
        }
        section[data-testid="stSidebar"] h1, section[data-testid="stSidebar"] h2, section[data-testid="stSidebar"] h3, section[data-testid="stSidebar"] h4 {
            color: #fff;
            text-shadow: 0 4px 16px rgba(0,0,0,0.18);
            font-family: 'Orbitron', 'Montserrat', Arial, sans-serif !important;
        }
        /* 3D effect for main headers */
        h1, h2, h3, h4 {
            font-family: 'Orbitron', 'Montserrat', Arial, sans-serif !important;
            text-shadow: 0 6px 32px rgba(106,90,205,0.30), 0 2px 0 #fff, 0 0 12px #43cea2;
            color: #18181b;
            letter-spacing: 1.2px;
        }
        /* 3D glassmorphism for cards and containers */
        .stImage, .stTextArea, .stButton, .stDownloadButton {
            border-radius: 22px !important;
            box-shadow: 0 12px 48px 0 rgba(31, 38, 135, 0.22);
            background: rgba(255,255,255,0.18) !important;
            backdrop-filter: blur(10px);
            border: 1.5px solid rgba(255,255,255,0.22) !important;
        }
        /* 3D effect for image captions */
        .stImage > img {
            box-shadow: 0 12px 48px 0 rgba(31, 38, 135, 0.22);
            border-radius: 20px;
            border: 2.5px solid #6a5acd;
        }
        /* Button style */
        .stButton > button, .stDownloadButton > button {
            font-family: 'Orbitron', 'Montserrat', Arial, sans-serif !important;
            font-size: 1.28em;
            font-weight: 800;
            background: linear-gradient(90deg, #43cea2 0%, #6a5acd 100%);
            color: #fff;
            border-radius: 10px;
            box-shadow: 0 4px 16px rgba(31,38,135,0.16);
            border: none;
            transition: transform 0.12s;
            letter-spacing: 1.2px;
            text-transform: uppercase;
            padding: 16px 32px !important;
        }
        .stButton > button:hover, .stDownloadButton > button:hover {
            transform: scale(1.06);
            box-shadow: 0 8px 32px rgba(31,38,135,0.22);
        }
        /* 3D effect for status and info */
        .stMarkdown, .stText {
            font-family: 'Montserrat', Arial, sans-serif !important;
            font-size: 1.15em;
            text-shadow: 0 4px 16px rgba(59,130,246,0.12);
            color: #222;
            letter-spacing: 1px;
        }
        /* Card effect for prompt and output containers */
        .stTextArea, .stImage {
            background: linear-gradient(120deg, #f8fafc 0%, #e0e7ef 100%);
            border-radius: 22px !important;
            box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.18);
            border: 1.5px solid #e0e0e0 !important;
        }
        /* High contrast for labels and captions */
        label, .stCaption, .stHeader, .stSubheader {
            color: #18181b !important;
            font-weight: 700 !important;
            letter-spacing: 1.2px !important;
            text-shadow: 0 2px 8px rgba(59,130,246,0.10);
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    # Centered Header
    st.markdown(
        """
        <div style='text-align: center; padding-top: 32px;'>
            <h1 style='font-size: 2.8em; font-weight: bold; color: #fff; margin-bottom: 0;'>ArtPromtAi - Text to Image Generator</h1>
            <p style='font-size: 1.3em; color: #f5f5f5; margin-top: 8px;'>Transform your text descriptions into stunning AI-generated images!</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Sidebar for model settings
    with st.sidebar:
        st.header("⚙️ Generation Settings")

        # Model loading status
        with st.spinner("Loading models..."):
            try:
                pipe, tokenizer, text_encoder, device = load_model()
            except Exception as e:
                st.error(f"Error loading models: {str(e)}")
                st.stop()

        # Generation parameters
        num_inference_steps = st.slider("Inference Steps", 10, 50, 20)
        guidance_scale = st.slider("Guidance Scale", 1.0, 20.0, 7.5)
        width = st.selectbox("Width", [512, 768, 1024], index=0)
        height = st.selectbox("Height", [512, 768, 1024], index=0)
        num_images = st.slider("Number of Images", 1, 4, 1)
        negative_prompt = "blurry, bad art, ugly, watermark, low quality"
        seed = None

    # Main content area
    # Centered layout: Prompt box and button at top, results below
    st.markdown("<div style='display: flex; flex-direction: column; align-items: center;'>", unsafe_allow_html=True)
    st.markdown("<div style='width: 420px;'>", unsafe_allow_html=True)
    st.markdown("""
    <h2 style='font-size: 2em; font-weight: bold; color: #18181b; letter-spacing: 1.2px; text-shadow: 0 2px 8px rgba(59,130,246,0.10); margin-bottom: 0.5em;'>📝 Enter The Story</h2>
    """, unsafe_allow_html=True)
    prompt = st.text_area(
        "Enter your image description:",
        value="A beautiful landscape with mountains and a lake at sunset",
        height=100,
        help="Describe the image you want to generate in detail"
    )
    generate_btn = st.button("Generate Images", type="primary", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # Results and status below input
    st.markdown("<div style='width: 420px; margin-top: 24px; text-align: left;'>", unsafe_allow_html=True)
    st.markdown(
        """
        <div style='text-align: left; width: 100%;'>
            <h2 style='font-size: 2em; font-weight: bold; color: #18181b; letter-spacing: 1.2px; text-shadow: 0 2px 8px rgba(59,130,246,0.10); margin-bottom: 0.5em;'>🖼️ Generated Images</h2>
        </div>
        """,
        unsafe_allow_html=True
    )
    status_text = st.empty()
    generation_time_text = st.empty()
    if generate_btn and prompt:
        generation_params = {
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "width": width,
            "height": height,
            "num_images_per_prompt": num_images,
            "negative_prompt": negative_prompt if negative_prompt else None
        }
        if seed is not None:
            torch.manual_seed(seed)
            generation_params["generator"] = torch.Generator(device=device).manual_seed(seed)
        progress_bar = st.progress(0)
        try:
            start_time = time.time()
            status_text.text("🔄 Generating images...")
            for i in range(num_images):
                progress_bar.progress((i + 1) / num_images)
                single_params = generation_params.copy()
                single_params["num_images_per_prompt"] = 1
                image = generate_image(pipe, prompt, **single_params)
                st.markdown("<div style='display: flex; justify-content: center; align-items: center;'>", unsafe_allow_html=True)
                st.image(image, caption=f"Generated Image {i+1}", width=width, clamp=True)
                st.markdown("</div>", unsafe_allow_html=True)
                img_buffer = io.BytesIO()
                image.save(img_buffer, format='PNG')
                st.download_button(
                    f"Download Image {i+1}",
                    data=img_buffer.getvalue(),
                    file_name=f"artpromtai_image_{i+1}.png",
                    mime="image/png"
                )
            generation_time = time.time() - start_time
            status_text.text("")
            generation_time_text.text(f"✅ Generation completed in {generation_time:.1f} seconds!")
        except Exception as e:
            st.error(f"Error during generation: {str(e)}")
            status_text.text("❌ Generation failed!")
            generation_time_text.text("")
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # ...existing code...

if __name__ == "__main__":
    main()
