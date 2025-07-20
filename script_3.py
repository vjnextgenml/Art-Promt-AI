# Create a complete Streamlit application for artpromtAi
streamlit_app_code = '''
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
    page_title="artpromtAi - Text to Image Generator",
    page_icon="🎨",
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
    # Header
    st.title("🎨 artpromtAi - Text to Image Generator")
    st.markdown("Transform your text descriptions into stunning AI-generated images!")
    
    # Sidebar for model settings
    with st.sidebar:
        st.header("⚙️ Generation Settings")
        
        # Model loading status
        with st.spinner("Loading models..."):
            try:
                pipe, tokenizer, text_encoder, device = load_model()
                st.success(f"Models loaded successfully! Using: {device.upper()}")
            except Exception as e:
                st.error(f"Error loading models: {str(e)}")
                st.stop()
        
        # Generation parameters
        num_inference_steps = st.slider("Inference Steps", 10, 50, 20)
        guidance_scale = st.slider("Guidance Scale", 1.0, 20.0, 7.5)
        width = st.selectbox("Width", [512, 768, 1024], index=0)
        height = st.selectbox("Height", [512, 768, 1024], index=0)
        num_images = st.slider("Number of Images", 1, 4, 1)
        
        # Seed for reproducibility
        use_seed = st.checkbox("Use Fixed Seed")
        seed = st.number_input("Seed", value=42) if use_seed else None
        
        # Advanced settings
        with st.expander("Advanced Settings"):
            negative_prompt = st.text_area("Negative Prompt", 
                value="blurry, bad art, ugly, watermark, low quality")
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📝 Text Input & Analysis")
        
        # Text input
        prompt = st.text_area(
            "Enter your image description:",
            value="A beautiful landscape with mountains and a lake at sunset",
            height=100,
            help="Describe the image you want to generate in detail"
        )
        
        # Prompt analysis
        if prompt:
            st.subheader("🔍 Prompt Analysis")
            analysis = analyze_prompt(tokenizer, prompt)
            
            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("Total Tokens", analysis["total_tokens"])
            with col_b:
                st.metric("Max Tokens", 77)  # CLIP max context length
            
            # Show tokenization breakdown
            with st.expander("View Token Breakdown"):
                for i, token in enumerate(analysis["token_breakdown"]):
                    st.text(f"{i:2d}. {token}")
        
        # Generate button
        generate_btn = st.button("🎨 Generate Images", type="primary", use_container_width=True)
    
    with col2:
        st.header("🖼️ Generated Images")
        
        if generate_btn and prompt:
            # Generation parameters
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
            
            # Progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                start_time = time.time()
                status_text.text("🔄 Generating images...")
                
                # Generate images
                for i in range(num_images):
                    progress_bar.progress((i + 1) / num_images)
                    
                    # Generate single image
                    single_params = generation_params.copy()
                    single_params["num_images_per_prompt"] = 1
                    
                    image = generate_image(pipe, prompt, **single_params)
                    
                    # Display image
                    st.image(image, caption=f"Generated Image {i+1}", use_column_width=True)
                    
                    # Download button for each image
                    img_buffer = io.BytesIO()
                    image.save(img_buffer, format='PNG')
                    st.download_button(
                        f"Download Image {i+1}",
                        data=img_buffer.getvalue(),
                        file_name=f"artpromtai_image_{i+1}.png",
                        mime="image/png"
                    )
                
                generation_time = time.time() - start_time
                status_text.text(f"✅ Generation completed in {generation_time:.1f} seconds!")
                
            except Exception as e:
                st.error(f"Error during generation: {str(e)}")
                status_text.text("❌ Generation failed!")
    
    # Footer with project information
    st.markdown("---")
    st.markdown("""
        ### About artpromtAi
        This project demonstrates text-to-image generation using:
        - **Text Tokenization**: CLIP tokenizer for processing text prompts
        - **Stable Diffusion Model**: Latent diffusion model for image generation  
        - **Streamlit Interface**: Interactive web application
        - **Dataset**: Trained on LAION-5B dataset with billions of image-text pairs
        
        **Hardware Requirements**: GPU with 8GB+ VRAM recommended for optimal performance.
    """)

if __name__ == "__main__":
    main()
'''

# Save the Streamlit app code
with open('artpromtai_streamlit_app.py', 'w') as f:
    f.write(streamlit_app_code)

print("✅ Streamlit Application Code Generated Successfully!")
print("📁 File saved as: artpromtai_streamlit_app.py")
print("\n🔧 Key Features Implemented:")
features = [
    "Model loading with caching",
    "Text tokenization analysis",  
    "Interactive parameter controls",
    "Progress tracking",
    "Multi-image generation",
    "Download functionality", 
    "Error handling",
    "Responsive UI layout"
]

for i, feature in enumerate(features, 1):
    print(f"  {i}. ✓ {feature}")

print(f"\n📊 Total Lines of Code: {len(streamlit_app_code.split(chr(10)))}")
print("🚀 Ready to deploy with: streamlit run artpromtai_streamlit_app.py")