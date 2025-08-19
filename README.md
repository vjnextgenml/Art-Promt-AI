🎨 ArtPromptAI — Painting Generator from Descriptive Stories
ArtPromptAI is an AI-powered creative tool that transforms descriptive stories into visually stunning paintings.
It combines Hugging Face models for story generation and text-to-image AI to create original artworks — bridging the gap between storytelling and visual art.

✨ Features
🖋 Story Generation — Creates rich, descriptive narratives from an image or text prompt.

🎨 Painting Generation — Converts stories into high-quality digital paintings.

🛠 Custom Model Training — Fine-tunes a tokenizer & uses a sample image-text dataset for improved coherence.

🌐 Streamlit-based UI — Simple, interactive, and user-friendly web app.

📂 Dataset Integration — Works with sample datasets or connects to external APIs for more variety.

🏗 Project Flow
1️⃣ Input → User uploads an image or enters a text description.
2️⃣ Story Generation → Hugging Face models create a detailed story.
3️⃣ Text-to-Image → AI generates a painting from the story.
4️⃣ Output → Displays the final artwork with download options.

🛠 Tech Stack
Languages: Python 🐍

Framework: Streamlit ⚡

AI Models: Hugging Face Transformers 🤖, Stable Diffusion 🎨

APIs: Hugging Face Hub, Image Dataset APIs 🌍

Datasets: Custom sample tokenizer dataset + image-text dataset 📊

📦 Installation
bash
Copy
Edit
# Clone the repository
git clone https://github.com/yourusername/ArtPromptAI.git
cd ArtPromptAI

# Create a virtual environment
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
🚀 Usage
bash
Copy
Edit
streamlit run app.py
📤 Upload an image or ✍️ enter a text prompt.

📝 Click Generate Story to create the narrative.

🎨 Click Generate Painting to see the AI artwork.

💾 Download or share your creation.

📂 Dataset Details
Tokenizer Training → Improves AI storytelling accuracy.

Image-Text Dataset → Trains/fine-tunes the text-to-image model.

API Support → Fetches additional data for richer outputs.

🔮 Future Enhancements
🖌 Add style selection (watercolor, oil, comic book, etc.)

🌍 Multilingual story & painting generation

🖼 Gallery to store & share creations

⚡ Real-time preview of painting generation


👨‍💻 Author
Viju Jaison
🎓 B.Tech Information Technology | 💡 Data Science & AI Enthusiast
