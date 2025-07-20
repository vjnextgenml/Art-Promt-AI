# Create a comprehensive comparison table of text-to-image datasets
import pandas as pd

# Dataset comparison data based on research
datasets_data = {
    'Dataset': [
        'LAION-5B',
        'LAION-400M', 
        'COCO-Text',
        'Common Crawl',
        'MS COCO',
        'Visual Genome',
        'Flickr30k',
        'ConceptualCaptions'
    ],
    'Size': [
        '5.85 billion image-text pairs',
        '400 million image-text pairs',
        '173k text annotations in 63k images',
        'Petabytes of web data',
        '330k images with 2.5M captions',
        '108k images with 5.4M descriptions',
        '31k images with 158k captions',
        '3.3M image-caption pairs'
    ],
    'Language Support': [
        'English (2.32B) + 100+ languages',
        'Primarily English',
        'Multiple languages',
        'Multi-language',
        'English',
        'English', 
        'English',
        'English'
    ],
    'Source': [
        'Common Crawl (CLIP filtered)',
        'Common Crawl (CLIP filtered)',
        'MS COCO dataset extension',
        'Web crawl archives',
        'Flickr images with crowd annotations',
        'Crowd-sourced annotations',
        'Flickr images',
        'Alt-text from web images'
    ],
    'Quality Filter': [
        'CLIP similarity threshold',
        'CLIP similarity threshold',
        'Manual annotation quality',
        'Raw web data (unfiltered)',
        'Human verified',
        'Human verified',
        'Human verified',
        'Basic text cleaning'
    ],
    'Image Resolution': [
        'Variable (typically 512px+)',
        'Variable (typically 512px+)',
        'Variable (COCO images)',
        'Variable web sizes',
        'Variable',
        'Variable',
        'Variable',
        'Variable'
    ],
    'Training Use': [
        'Stable Diffusion, CLIP models',
        'Stable Diffusion v1.0-1.2',
        'Text detection research',
        'Foundation for LAION datasets',
        'Image captioning, VQA',
        'Scene understanding',
        'Image captioning',
        'Vision-language models'
    ],
    'Accessibility': [
        'Open source (requires agreement)',
        'Open source',
        'Open source',
        'Public archives',
        'Open source',
        'Open source',
        'Open source',
        'Open source (Google)'
    ]
}

# Create DataFrame
df = pd.DataFrame(datasets_data)

# Save as CSV for reference
df.to_csv('text_image_datasets_comparison.csv', index=False)

# Display summary
print("📊 Text-to-Image Datasets Comparison Table Created!")
print("=" * 60)
print(f"Total Datasets Analyzed: {len(df)}")
print(f"Largest Dataset: LAION-5B with 5.85 billion pairs")
print(f"Most Used for Training: LAION datasets for modern text-to-image models")
print("\n📁 Saved as: text_image_datasets_comparison.csv")

# Show basic statistics
print(f"\n📈 Dataset Statistics:")
print(f"  • Open Source: {len(df[df['Accessibility'].str.contains('Open source')])} out of {len(df)}")
print(f"  • Multi-language: {len(df[df['Language Support'].str.contains('Multi|Multiple|100\\+')])}")
print(f"  • CLIP filtered: {len(df[df['Quality Filter'].str.contains('CLIP')])}")
print(f"  • Used for Stable Diffusion: 2 (LAION-5B, LAION-400M)")

# Display first few rows
print(f"\n📋 Sample Data (first 3 rows):")
print(df[['Dataset', 'Size', 'Language Support', 'Training Use']].head(3).to_string(index=False))