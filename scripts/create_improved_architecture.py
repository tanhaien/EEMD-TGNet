#!/usr/bin/env python3
"""
Create Professional EEMD-TGNet Architecture Diagram with Academic Standards
"""
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, ConnectionPatch, Rectangle
import numpy as np

# Set up professional academic style
plt.style.use('default')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif', 'serif']
plt.rcParams['font.size'] = 10
plt.rcParams['axes.linewidth'] = 1.2

# Academic-standard colors (grayscale and muted colors)
colors = {
    'input': '#F8F9FA',      # Light gray
    'process': '#E9ECEF',     # Medium light gray  
    'model': '#DEE2E6',       # Medium gray
    'output': '#CED4DA',      # Darker gray
    'accent': '#6C757D',      # Dark gray for accents
    'text': '#212529',        # Dark text
    'border': '#495057',      # Border gray
    'highlight': '#ADB5BD'    # Highlight gray
}

fig, ax = plt.subplots(1, 1, figsize=(14, 9))
ax.set_xlim(0, 14)
ax.set_ylim(0, 9)
ax.axis('off')

# Title
ax.text(7, 8.5, 'EEMD-TGNet Architecture Framework', fontsize=16, fontweight='bold', 
        ha='center', va='center', color=colors['text'])

# Step 1: Input Layer
input_rect = Rectangle((0.5, 6.5), 2, 1.2, facecolor=colors['input'], 
                      edgecolor=colors['border'], linewidth=1.5)
ax.add_patch(input_rect)
ax.text(1.5, 7.1, 'Input Time Series', fontsize=11, fontweight='bold',
        ha='center', va='center', color=colors['text'])
ax.text(1.5, 6.8, 'X(t) ∈ ℝᵀˣᵈ', fontsize=9,
        ha='center', va='center', color=colors['text'], style='italic')

# Step 2: EEMD Decomposition
eemd_rect = Rectangle((3.5, 6), 3, 2.2, facecolor=colors['process'], 
                     edgecolor=colors['border'], linewidth=1.5)
ax.add_patch(eemd_rect)
ax.text(5, 7.5, 'EEMD Decomposition', fontsize=12, fontweight='bold',
        ha='center', va='center', color=colors['text'])
ax.text(5, 7.1, 'X(t) = ∑ⁿᵢ₌₁ IMFᵢ(t) + R(t)', fontsize=10,
        ha='center', va='center', color=colors['text'])

# EEMD parameters box
param_rect = Rectangle((3.7, 6.1), 2.6, 0.8, facecolor=colors['model'], 
                      edgecolor=colors['accent'], linewidth=1)
ax.add_patch(param_rect)
ax.text(5, 6.5, 'Parameters:', fontsize=9, fontweight='bold',
        ha='center', va='center', color=colors['text'])
ax.text(5, 6.25, '• Ensemble: 100, Noise: 0.2σ', fontsize=8,
        ha='center', va='center', color=colors['text'])

# Step 3: IMF Selection
select_rect = Rectangle((7.5, 6), 2.5, 2.2, facecolor=colors['process'], 
                       edgecolor=colors['border'], linewidth=1.5)
ax.add_patch(select_rect)
ax.text(8.75, 7.5, 'IMF Selection', fontsize=12, fontweight='bold',
        ha='center', va='center', color=colors['text'])
ax.text(8.75, 7.1, 'Criteria:', fontsize=10, fontweight='bold',
        ha='center', va='center', color=colors['text'])
ax.text(8.75, 6.8, '• Autocorr > 0.1', fontsize=9,
        ha='center', va='center', color=colors['text'])
ax.text(8.75, 6.6, '• Variance > 5%', fontsize=9,
        ha='center', va='center', color=colors['text'])
ax.text(8.75, 6.4, '• Exclude HF noise', fontsize=9,
        ha='center', va='center', color=colors['text'])

# Step 4: Parallel TCN-GRU Models
model_y = 4
model_positions = [(1.5, model_y), (3.5, model_y), (5.5, model_y), (7.5, model_y), (9.5, model_y)]
imf_labels = ['IMF₁', 'IMF₂', 'IMF₃', 'IMF₄', 'IMF₅']

# Header for parallel processing
ax.text(5.5, 5, 'Parallel TCN-GRU Processing', fontsize=12, fontweight='bold',
        ha='center', va='center', color=colors['text'])

for i, (x, y) in enumerate(model_positions):
    # IMF input
    imf_rect = Rectangle((x-0.4, y+0.3), 0.8, 0.3, facecolor=colors['input'], 
                        edgecolor=colors['border'], linewidth=1)
    ax.add_patch(imf_rect)
    ax.text(x, y+0.45, imf_labels[i], fontsize=9, fontweight='bold',
            ha='center', va='center', color=colors['text'])
    
    # TCN layer
    tcn_rect = Rectangle((x-0.5, y-0.1), 1, 0.35, facecolor=colors['model'], 
                        edgecolor=colors['accent'], linewidth=1)
    ax.add_patch(tcn_rect)
    ax.text(x, y+0.075, 'TCN', fontsize=9, fontweight='bold',
            ha='center', va='center', color=colors['text'])
    
    # GRU layer
    gru_rect = Rectangle((x-0.5, y-0.5), 1, 0.35, facecolor=colors['model'], 
                        edgecolor=colors['accent'], linewidth=1)
    ax.add_patch(gru_rect)
    ax.text(x, y-0.325, 'GRU', fontsize=9, fontweight='bold',
            ha='center', va='center', color=colors['text'])
    
    # Prediction output
    pred_rect = Rectangle((x-0.3, y-0.9), 0.6, 0.25, facecolor=colors['output'], 
                         edgecolor=colors['border'], linewidth=1)
    ax.add_patch(pred_rect)
    ax.text(x, y-0.775, f'Ŷ{i+1}(t)', fontsize=8, fontweight='bold',
            ha='center', va='center', color=colors['text'])
    
    # Connection arrows
    ax.annotate('', xy=(x, y+0.3), xytext=(x, y+0.25),
                arrowprops=dict(arrowstyle='->', color=colors['accent'], lw=1.2))
    ax.annotate('', xy=(x, y-0.1), xytext=(x, y-0.15),
                arrowprops=dict(arrowstyle='->', color=colors['accent'], lw=1.2))
    ax.annotate('', xy=(x, y-0.5), xytext=(x, y-0.55),
                arrowprops=dict(arrowstyle='->', color=colors['accent'], lw=1.2))

# Step 5: Aggregation
agg_rect = Rectangle((4.5, 1.8), 2, 0.8, facecolor=colors['process'], 
                    edgecolor=colors['border'], linewidth=1.5)
ax.add_patch(agg_rect)
ax.text(5.5, 2.4, 'Ensemble Aggregation', fontsize=11, fontweight='bold',
        ha='center', va='center', color=colors['text'])
ax.text(5.5, 2.1, 'Ŷ(t) = ∑ⁿᵢ₌₁ Ŷᵢ(t)', fontsize=10,
        ha='center', va='center', color=colors['text'])

# Step 6: Final Output
output_rect = Rectangle((4.5, 0.5), 2, 0.8, facecolor=colors['output'], 
                       edgecolor=colors['border'], linewidth=1.5)
ax.add_patch(output_rect)
ax.text(5.5, 1.1, 'Solar Forecast', fontsize=11, fontweight='bold',
        ha='center', va='center', color=colors['text'])
ax.text(5.5, 0.8, 'Ŷ(t+h) ∈ ℝʰ', fontsize=9,
        ha='center', va='center', color=colors['text'], style='italic')

# Connection arrows between main stages
arrows = [
    ((2.5, 7.1), (3.5, 7.1)),  # Input to EEMD
    ((6.5, 7.1), (7.5, 7.1)),  # EEMD to Selection
]

for start, end in arrows:
    ax.annotate('', xy=end, xytext=start,
                arrowprops=dict(arrowstyle='->', color=colors['accent'], lw=2))

# Arrows from selection to parallel models
for i, (x, y) in enumerate(model_positions):
    start_x = 8.75
    start_y = 6
    ax.annotate('', xy=(x, y+0.6), xytext=(start_x, start_y),
                arrowprops=dict(arrowstyle='->', color=colors['accent'], 
                              lw=1, linestyle='--', alpha=0.7))

# Arrows from models to aggregation
for i, (x, y) in enumerate(model_positions):
    ax.annotate('', xy=(5.5, 2.6), xytext=(x, y-0.9),
                arrowprops=dict(arrowstyle='->', color=colors['accent'], 
                              lw=1.2, alpha=0.8))

# Final arrow
ax.annotate('', xy=(5.5, 1.3), xytext=(5.5, 1.8),
            arrowprops=dict(arrowstyle='->', color=colors['accent'], lw=2))

# Technical specifications sidebar
spec_rect = Rectangle((11.5, 1.5), 2.3, 5, facecolor=colors['input'], 
                     edgecolor=colors['border'], linewidth=1.5)
ax.add_patch(spec_rect)
ax.text(12.65, 6.2, 'Technical Specifications', fontsize=10, fontweight='bold',
        ha='center', va='center', color=colors['text'])

# TCN specs
tcn_spec_rect = Rectangle((11.7, 4.8), 1.9, 1.2, facecolor=colors['model'], 
                         edgecolor=colors['accent'], linewidth=1)
ax.add_patch(tcn_spec_rect)
ax.text(12.65, 5.7, 'TCN Architecture', fontsize=9, fontweight='bold',
        ha='center', va='center', color=colors['text'])
ax.text(12.65, 5.4, '• 3 dilated conv layers', fontsize=8,
        ha='center', va='center', color=colors['text'])
ax.text(12.65, 5.2, '• Dilation: [1,2,4]', fontsize=8,
        ha='center', va='center', color=colors['text'])
ax.text(12.65, 5.0, '• 64 filters, kernel=3', fontsize=8,
        ha='center', va='center', color=colors['text'])

# GRU specs
gru_spec_rect = Rectangle((11.7, 3.3), 1.9, 1.2, facecolor=colors['model'], 
                         edgecolor=colors['accent'], linewidth=1)
ax.add_patch(gru_spec_rect)
ax.text(12.65, 4.2, 'GRU Architecture', fontsize=9, fontweight='bold',
        ha='center', va='center', color=colors['text'])
ax.text(12.65, 3.9, '• 128 hidden units', fontsize=8,
        ha='center', va='center', color=colors['text'])
ax.text(12.65, 3.7, '• 2 layers', fontsize=8,
        ha='center', va='center', color=colors['text'])
ax.text(12.65, 3.5, '• Dropout: 0.3', fontsize=8,
        ha='center', va='center', color=colors['text'])

# Optimization specs
opt_spec_rect = Rectangle((11.7, 1.8), 1.9, 1.2, facecolor=colors['highlight'], 
                         edgecolor=colors['border'], linewidth=1)
ax.add_patch(opt_spec_rect)
ax.text(12.65, 2.7, 'Edge Optimization', fontsize=9, fontweight='bold',
        ha='center', va='center', color=colors['text'])
ax.text(12.65, 2.4, '• Pruning: 30%', fontsize=8,
        ha='center', va='center', color=colors['text'])
ax.text(12.65, 2.2, '• Quantization: INT8', fontsize=8,
        ha='center', va='center', color=colors['text'])
ax.text(12.65, 2.0, '• Memory: 75% ↓', fontsize=8,
        ha='center', va='center', color=colors['text'])

plt.tight_layout()
plt.savefig('/home/anhpt95te/Downloads/UAI paper/results/eemd_tgnet_architecture_professional.png', 
            dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
plt.savefig('/home/anhpt95te/Downloads/UAI paper/results/eemd_tgnet_architecture_professional.pdf', 
            dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
plt.close()

print("Professional architecture diagram created successfully!")