# artpromtAi Project Structure

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
