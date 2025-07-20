import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np

# Updated components data with VAE Encoder and better positioning
components_data = [
    {"component": "User Input", "type": "Interface", "details": "Streamlit UI", "x": 1, "y": 5, "layer": "Input"},
    {"component": "CLIP Tokenizer", "type": "Processing", "details": "Text → Tokens", "x": 2.5, "y": 5, "layer": "Processing"},
    {"component": "Text Encoder", "type": "Model", "details": "CLIP ViT-L/14", "x": 4, "y": 5, "layer": "Processing"},
    {"component": "VAE Encoder", "type": "Model", "details": "Pixel → Latent", "x": 5.5, "y": 4, "layer": "Model"},
    {"component": "U-Net", "type": "Model", "details": "Denoising 860M", "x": 7, "y": 5, "layer": "Model"},
    {"component": "Scheduler", "type": "Processing", "details": "Noise Schedule", "x": 7, "y": 6.5, "layer": "Processing"},
    {"component": "VAE Decoder", "type": "Model", "details": "Latent → Pixel", "x": 8.5, "y": 5, "layer": "Model"},
    {"component": "Display", "type": "Output", "details": "512x512+ Image", "x": 10, "y": 5, "layer": "Output"},
    {"component": "LAION-5B", "type": "Dataset", "details": "5.85B pairs", "x": 4, "y": 2.5, "layer": "Data"}
]

df = pd.DataFrame(components_data)

# Create color mapping for different types
color_map = {
    'Interface': '#1FB8CD',  # Strong cyan
    'Processing': '#FFC185', # Light orange
    'Model': '#5D878F',      # Cyan
    'Output': '#D2BA4C',     # Moderate yellow
    'Dataset': '#ECEBD5'     # Light green
}

# Create the figure
fig = go.Figure()

# Add rectangular nodes for each component with larger sizes
for idx, row in df.iterrows():
    # Add rectangle shape for each component
    fig.add_shape(
        type="rect",
        x0=row['x']-0.5, y0=row['y']-0.4,
        x1=row['x']+0.5, y1=row['y']+0.4,
        fillcolor=color_map[row['type']],
        line=dict(color='#13343B', width=3),
        opacity=0.9
    )
    
    # Add component name text (larger font)
    fig.add_annotation(
        x=row['x'], y=row['y']+0.15,
        text=f"<b>{row['component']}</b>",
        showarrow=False,
        font=dict(size=11, color='black'),
        xanchor='center',
        yanchor='middle'
    )
    
    # Add details text (larger font)
    fig.add_annotation(
        x=row['x'], y=row['y']-0.15,
        text=row['details'],
        showarrow=False,
        font=dict(size=9, color='black'),
        xanchor='center',
        yanchor='middle'
    )

# Define main flow connections with larger arrows
main_flow = [
    (1.5, 5, 2.0, 5),       # User Input → Tokenizer
    (3.0, 5, 3.5, 5),       # Tokenizer → Text Encoder
    (4.5, 5, 6.5, 5),       # Text Encoder → U-Net
    (7.5, 5, 8.0, 5),       # U-Net → VAE Decoder
    (9.0, 5, 9.5, 5),       # VAE Decoder → Display
]

# Add main flow arrows (larger and more visible)
for x1, y1, x2, y2 in main_flow:
    fig.add_annotation(
        x=x2, y=y2,
        ax=x1, ay=y1,
        arrowhead=3,
        arrowsize=2,
        arrowwidth=4,
        arrowcolor='#B4413C',
        showarrow=True,
        text=""
    )

# Add secondary connections
secondary_connections = [
    (7, 6.1, 7, 5.4),       # Scheduler → U-Net
    (4, 3.0, 4, 4.6),       # LAION-5B → Text Encoder
    (5.5, 4.4, 6.5, 4.6)    # VAE Encoder → U-Net
]

# Add secondary arrows (different color)
for x1, y1, x2, y2 in secondary_connections:
    fig.add_annotation(
        x=x2, y=y2,
        ax=x1, ay=y1,
        arrowhead=2,
        arrowsize=1.5,
        arrowwidth=3,
        arrowcolor='#964325',
        showarrow=True,
        text=""
    )

# Add layer separation lines
layer_lines = [
    {"x": [1.75, 1.75], "y": [1.5, 7.5], "name": "Input|Processing"},
    {"x": [4.75, 4.75], "y": [1.5, 7.5], "name": "Processing|Model"},
    {"x": [9.25, 9.25], "y": [1.5, 7.5], "name": "Model|Output"}
]

for line in layer_lines:
    fig.add_trace(go.Scatter(
        x=line['x'], y=line['y'],
        mode='lines',
        line=dict(color='#13343B', width=1, dash='dot'),
        showlegend=False,
        hoverinfo='skip'
    ))

# Add layer labels with boxes
layer_labels = [
    {"text": "Input", "x": 1, "y": 7.2},
    {"text": "Processing", "x": 3.25, "y": 7.2},
    {"text": "Model", "x": 6.5, "y": 7.2},
    {"text": "Output", "x": 10, "y": 7.2},
    {"text": "Training Data", "x": 4, "y": 1.8}
]

for label in layer_labels:
    # Add background rectangle for layer labels
    fig.add_shape(
        type="rect",
        x0=label['x']-0.3, y0=label['y']-0.15,
        x1=label['x']+0.3, y1=label['y']+0.15,
        fillcolor='white',
        line=dict(color='#13343B', width=2),
        opacity=0.9
    )
    
    fig.add_annotation(
        x=label['x'], y=label['y'],
        text=f"<b>{label['text']}</b>",
        showarrow=False,
        font=dict(size=10, color='#13343B'),
        xanchor='center'
    )

# Add invisible scatter points for legend
for comp_type in df['type'].unique():
    fig.add_trace(go.Scatter(
        x=[None], y=[None],
        mode='markers',
        marker=dict(size=15, color=color_map[comp_type], 
                   line=dict(width=2, color='#13343B')),
        name=comp_type,
        showlegend=True
    ))

# Update layout
fig.update_layout(
    title='ArtPromptAI Text-to-Image Flow',
    xaxis=dict(
        showgrid=False,
        showticklabels=False,
        zeroline=False,
        range=[0, 11]
    ),
    yaxis=dict(
        showgrid=False,
        showticklabels=False,
        zeroline=False,
        range=[1, 8]
    ),
    legend=dict(orientation='h', yanchor='bottom', y=1.05, xanchor='center', x=0.5),
    plot_bgcolor='#f8f9fa',
    paper_bgcolor='white',
    showlegend=True
)

# Save the chart
fig.write_image('artprompt_architecture.png')