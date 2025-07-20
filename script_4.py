# Create requirements.txt and installation guide for artpromtAi

requirements_content = """# artpromtAi Requirements
# Core Dependencies for Text-to-Image Generation

# Deep Learning Framework
torch>=1.12.0
torchvision>=0.13.0

# Hugging Face Libraries
transformers>=4.21.0
diffusers>=0.20.0
accelerate>=0.20.0

# Streamlit Web Framework
streamlit>=1.25.0

# Image Processing
Pillow>=9.0.0
opencv-python>=4.5.0

# Tokenization
tokenizers>=0.13.0

# Scientific Computing
numpy>=1.21.0
scipy>=1.7.0

# Utilities
tqdm>=4.64.0
ftfy>=6.1.0
regex>=2022.7.9

# Optional: Fast CLIP tokenizer
# instant-clip-tokenizer>=0.1.0

# Optional: Model optimization
# safetensors>=0.3.0
# xformers>=0.0.20  # For memory optimization (requires CUDA)
"""

installation_guide = """# artpromtAi Installation Guide

## System Requirements

### Minimum Requirements:
- Python 3.8+
- 8GB RAM
- 4GB GPU VRAM (CUDA compatible)
- 50GB free disk space

### Recommended Requirements:
- Python 3.9+
- 16GB+ RAM  
- 12GB+ GPU VRAM (RTX 3080/4080, A100, etc.)
- 100GB+ free disk space
- CUDA 11.7+

## Installation Steps

### 1. Clone/Create Project Directory
```bash
mkdir artpromtai
cd artpromtai
```

### 2. Create Virtual Environment
```bash
# Using venv
python -m venv artpromtai_env
source artpromtai_env/bin/activate  # Linux/Mac
# artpromtai_env\\Scripts\\activate  # Windows

# OR using conda
conda create -n artpromtai python=3.9
conda activate artpromtai
```

### 3. Install PyTorch (GPU Version)
```bash
# For CUDA 11.7
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu117

# For CUDA 11.8  
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# For CPU only (not recommended)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### 4. Install Dependencies
```bash
pip install -r requirements.txt
```

### 5. Download Pre-trained Models (Optional)
Models will be automatically downloaded on first run, but you can pre-download:
```python
from diffusers import StableDiffusionPipeline
from transformers import CLIPTokenizer, CLIPTextModel

# This will download ~4-7GB of models
model_id = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(model_id)
tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
```

### 6. Run the Application
```bash
streamlit run artpromtai_streamlit_app.py
```

## Troubleshooting

### Common Issues:

1. **CUDA Out of Memory**
   - Reduce image dimensions (use 512x512 instead of 1024x1024)
   - Decrease batch size/number of images
   - Enable CPU offloading

2. **Model Download Errors**
   - Check internet connection
   - Clear Hugging Face cache: `rm -rf ~/.cache/huggingface/`
   - Try different model versions

3. **Slow Generation**
   - Ensure CUDA is properly installed
   - Use GPU instead of CPU
   - Consider using xformers for optimization

### Verification Commands:
```python
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
```
"""

project_structure = """# artpromtAi Project Structure

artpromtai/
├── artpromtai_streamlit_app.py    # Main Streamlit application
├── requirements.txt               # Python dependencies
├── README.md                     # Project documentation
├── models/                       # Model storage (auto-created)
│   ├── stable-diffusion-v1-5/    # Cached model files
│   └── clip/                     # CLIP model cache
├── outputs/                      # Generated images (auto-created)
│   ├── images/
│   └── logs/
└── utils/                        # Utility functions (optional)
    ├── __init__.py
    ├── tokenization.py           # Text processing utilities
    ├── image_processing.py       # Image manipulation
    └── model_utils.py            # Model loading helpers

## Key Files Description:

- **artpromtai_streamlit_app.py**: Main application with UI and generation logic
- **requirements.txt**: All necessary Python packages
- **models/**: Hugging Face models cached locally (~5-10GB)  
- **outputs/**: Generated images and logs
"""

# Save all files
with open('requirements.txt', 'w') as f:
    f.write(requirements_content)

with open('INSTALL.md', 'w') as f:
    f.write(installation_guide)

with open('PROJECT_STRUCTURE.md', 'w') as f:
    f.write(project_structure)

print("📦 Installation Package Created Successfully!")
print("\n📋 Files Generated:")
print("  ✓ requirements.txt - Python dependencies")
print("  ✓ INSTALL.md - Complete installation guide") 
print("  ✓ PROJECT_STRUCTURE.md - Project organization")
print("  ✓ artpromtai_streamlit_app.py - Main application (from previous step)")
print("  ✓ artpromtai_architecture.json - Project architecture")

print("\n🚀 Quick Start Commands:")
print("  1. pip install -r requirements.txt")
print("  2. streamlit run artpromtai_streamlit_app.py")
print("  3. Open http://localhost:8501 in your browser")

print(f"\n📊 Package Summary:")
print(f"  • Total Python packages: {len([line for line in requirements_content.split(chr(10)) if line and not line.startswith('#') and '>' in line])}")
print(f"  • Estimated download size: ~8-12GB (models + dependencies)")
print(f"  • Supported Python versions: 3.8+")
print(f"  • GPU memory requirement: 8GB+ VRAM")