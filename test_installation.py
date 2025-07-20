
"""
Quick Demo Script for artpromtAi
Run this to test your installation
"""

try:
    import torch
    import transformers
    import diffusers
    import streamlit

    print("✅ All required packages are installed!")
    print(f"PyTorch: {torch.__version__}")
    print(f"Transformers: {transformers.__version__}")
    print(f"Diffusers: {diffusers.__version__}")
    print(f"Streamlit: {streamlit.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    print("\n🚀 Ready to run artpromtAi!")
    print("Execute: streamlit run artpromtai_streamlit_app.py")

except ImportError as e:
    print(f"❌ Missing dependency: {e}")
    print("Please run: pip install -r requirements.txt")
