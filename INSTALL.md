# artpromtAi Installation Guide

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
# artpromtai_env\Scripts\activate  # Windows

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
