
"""
artpromtAi: Complete Text-to-Image Implementation Example
This example demonstrates all core components working together
"""

import torch
from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler
from transformers import CLIPTokenizer, CLIPTextModel
from PIL import Image
import numpy as np
import time
import os

class ArtPromptAI:
    """Complete text-to-image generation system"""

    def __init__(self, model_id="runwayml/stable-diffusion-v1-5", device=None):
        self.model_id = model_id
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.pipeline = None
        self.tokenizer = None
        self.text_encoder = None

        print(f"🚀 Initializing artpromtAi on {self.device}")
        self._load_models()

    def _load_models(self):
        """Load all required models and components"""
        print("📦 Loading Stable Diffusion pipeline...")

        # Load the main pipeline
        self.pipeline = StableDiffusionPipeline.from_pretrained(
            self.model_id,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            safety_checker=None,
            requires_safety_checker=False
        )

        # Move to device
        self.pipeline = self.pipeline.to(self.device)

        # Load tokenizer and text encoder separately for analysis
        print("🔤 Loading CLIP tokenizer and text encoder...")
        self.tokenizer = CLIPTokenizer.from_pretrained(
            self.model_id, 
            subfolder="tokenizer"
        )
        self.text_encoder = CLIPTextModel.from_pretrained(
            self.model_id, 
            subfolder="text_encoder"
        )

        print("✅ All models loaded successfully!")

    def analyze_prompt(self, prompt):
        """Analyze text prompt and show tokenization details"""
        print(f"🔍 Analyzing prompt: '{prompt}'")

        # Tokenize the prompt
        tokens = self.tokenizer(
            prompt, 
            return_tensors="pt", 
            max_length=77, 
            padding="max_length", 
            truncation=True
        )

        # Get token IDs and attention mask
        token_ids = tokens["input_ids"][0].tolist()
        attention_mask = tokens["attention_mask"][0].tolist()

        # Decode individual tokens
        token_breakdown = []
        for i, (token_id, mask) in enumerate(zip(token_ids, attention_mask)):
            if mask == 1:  # Only show non-padded tokens
                token_text = self.tokenizer.decode([token_id])
                token_breakdown.append({
                    'position': i,
                    'token_id': token_id,
                    'token_text': token_text,
                    'is_special': token_id in [49406, 49407]  # start/end tokens
                })

        analysis = {
            'original_prompt': prompt,
            'total_tokens': len(token_breakdown),
            'max_length': 77,
            'token_breakdown': token_breakdown,
            'truncated': len(prompt.split()) > 75  # Rough estimate
        }

        print(f"📊 Tokenization Results:")
        print(f"   • Total tokens: {analysis['total_tokens']}/77")
        print(f"   • Special tokens: {sum(1 for t in token_breakdown if t['is_special'])}")
        print(f"   • Content tokens: {analysis['total_tokens'] - sum(1 for t in token_breakdown if t['is_special'])}")

        return analysis

    def generate_image(self, prompt, **kwargs):
        """Generate image from text prompt with customizable parameters"""

        # Default parameters
        default_params = {
            'num_inference_steps': 20,
            'guidance_scale': 7.5,
            'width': 512,
            'height': 512,
            'num_images_per_prompt': 1,
            'negative_prompt': "blurry, bad art, low quality"
        }

        # Update with user parameters
        params = {**default_params, **kwargs}

        print(f"🎨 Generating image with parameters:")
        for key, value in params.items():
            print(f"   • {key}: {value}")

        # Generate image
        start_time = time.time()

        with torch.autocast(self.device):
            result = self.pipeline(prompt=prompt, **params)
            image = result.images[0]

        generation_time = time.time() - start_time
        print(f"⚡ Generation completed in {generation_time:.2f} seconds")

        return image, generation_time

    def batch_generate(self, prompts, save_dir="outputs", **kwargs):
        """Generate multiple images from a list of prompts"""
        os.makedirs(save_dir, exist_ok=True)
        results = []

        print(f"🔄 Processing {len(prompts)} prompts...")

        for i, prompt in enumerate(prompts, 1):
            print(f"[{i}/{len(prompts)}] Processing: '{prompt[:50]}...'")

            # Analyze prompt
            analysis = self.analyze_prompt(prompt)

            # Generate image
            image, gen_time = self.generate_image(prompt, **kwargs)

            # Save image
            filename = f"image_{i:03d}.png"
            filepath = os.path.join(save_dir, filename)
            image.save(filepath)

            results.append({
                'prompt': prompt,
                'analysis': analysis,
                'image_path': filepath,
                'generation_time': gen_time
            })

            print(f"💾 Saved: {filepath}")

        total_time = sum(r['generation_time'] for r in results)
        print(f"\n✅ Batch generation completed!")
        print(f"📊 Total time: {total_time:.2f}s | Average: {total_time/len(prompts):.2f}s per image")

        return results

def main():
    """Example usage of artpromtAi system"""

    # Initialize the system
    ai = ArtPromptAI()

    # Example prompts for testing
    test_prompts = [
        "A beautiful landscape with mountains and a lake at sunset, digital art",
        "A futuristic cityscape with flying cars and neon lights",
        "A cute cat wearing a wizard hat, photorealistic style"
    ]

    # Single image generation example
    print("\n" + "="*60)
    print("🖼️  SINGLE IMAGE GENERATION EXAMPLE")
    print("="*60)

    prompt = test_prompts[0]
    analysis = ai.analyze_prompt(prompt)
    image, gen_time = ai.generate_image(
        prompt,
        num_inference_steps=25,
        guidance_scale=8.0,
        width=768,
        height=768
    )

    # Save the image
    os.makedirs("examples", exist_ok=True)
    image.save("examples/single_example.png")
    print("💾 Saved example image: examples/single_example.png")

    # Batch generation example
    print("\n" + "="*60)
    print("🔄 BATCH GENERATION EXAMPLE")
    print("="*60)

    results = ai.batch_generate(
        test_prompts,
        save_dir="examples/batch",
        num_inference_steps=20,
        guidance_scale=7.5
    )

    # Print summary
    print("\n" + "="*60)
    print("📈 GENERATION SUMMARY")
    print("="*60)

    for i, result in enumerate(results, 1):
        print(f"Image {i}:")
        print(f"  Prompt: {result['prompt'][:50]}...")
        print(f"  Tokens: {result['analysis']['total_tokens']}")
        print(f"  Time: {result['generation_time']:.2f}s")
        print(f"  Saved: {result['image_path']}")

if __name__ == "__main__":
    main()
