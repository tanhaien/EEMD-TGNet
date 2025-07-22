#!/usr/bin/env python3
"""
Create EEMD-TGNet Architecture Diagram
"""
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, ConnectionPatch
import numpy as np

# Set up the figure
fig, ax = plt.subplots(1, 1, figsize=(14, 10))
ax.set_xlim(0, 14)
ax.set_ylim(0, 10)
ax.axis('off')

# Colors
colors = {
    'input': '#E8F4FD',
    'eemd': '#FFE5B4', 
    'tcn': '#D4EDDA',
    'gru': '#F8D7DA',
    'output': '#E2E3E5',
    'arrow': '#6C757D',
    'text': '#212529'
}

# Title
ax.text(7, 9.5, 'EEMD-TGNet Architecture', fontsize=18, fontweight='bold', 
        ha='center', va='center', color=colors['text'])

# Input Data Box
input_box = FancyBboxPatch((0.5, 7.5), 2.5, 1, 
                          boxstyle="round,pad=0.1", 
                          facecolor=colors['input'], 
                          edgecolor='black', linewidth=2)
ax.add_patch(input_box)
ax.text(1.75, 8, 'Solar Energy\nTime Series\nX(t)', fontsize=11, fontweight='bold',
        ha='center', va='center', color=colors['text'])

# EEMD Decomposition
eemd_box = FancyBboxPatch((4, 7), 3, 2, 
                         boxstyle="round,pad=0.1",
                         facecolor=colors['eemd'], 
                         edgecolor='black', linewidth=2)
ax.add_patch(eemd_box)
ax.text(5.5, 8.5, 'EEMD Decomposition', fontsize=12, fontweight='bold',
        ha='center', va='center', color=colors['text'])
ax.text(5.5, 7.8, 'X(t) = Σ IMFᵢ(t) + R(t)', fontsize=10,
        ha='center', va='center', color=colors['text'])
ax.text(5.5, 7.3, '• Ensemble Size: 100\n• Noise Amplitude: 0.2σ\n• Max IMFs: 10', 
        fontsize=9, ha='center', va='center', color=colors['text'])

# Arrow from Input to EEMD
arrow1 = ConnectionPatch((3, 8), (4, 8), "data", "data",
                        arrowstyle="->", shrinkA=0, shrinkB=0, 
                        mutation_scale=20, fc=colors['arrow'], ec=colors['arrow'])
ax.add_artist(arrow1)

# IMF Selection Box
imf_box = FancyBboxPatch((8, 7), 2.5, 2,
                        boxstyle="round,pad=0.1",
                        facecolor=colors['eemd'],
                        edgecolor='black', linewidth=2)
ax.add_patch(imf_box)
ax.text(9.25, 8.5, 'IMF Selection', fontsize=12, fontweight='bold',
        ha='center', va='center', color=colors['text'])
ax.text(9.25, 7.8, '• Autocorr > 0.1\n• Variance > 5%\n• Exclude HF noise\n• Select 4-6 IMFs', 
        fontsize=9, ha='center', va='center', color=colors['text'])

# Arrow from EEMD to IMF Selection
arrow2 = ConnectionPatch((7, 8), (8, 8), "data", "data",
                        arrowstyle="->", shrinkA=0, shrinkB=0,
                        mutation_scale=20, fc=colors['arrow'], ec=colors['arrow'])
ax.add_artist(arrow2)

# Individual TCN-GRU Models for each IMF
imf_positions = [(2, 5), (4.5, 5), (7, 5), (9.5, 5), (12, 5)]
model_labels = ['IMF₁', 'IMF₂', 'IMF₃', 'IMF₄', 'IMF₅']

for i, (x, y) in enumerate(imf_positions):
    # IMF input
    imf_input = FancyBboxPatch((x-0.4, y+1), 0.8, 0.4,
                              boxstyle="round,pad=0.05",
                              facecolor=colors['input'],
                              edgecolor='gray', linewidth=1)
    ax.add_patch(imf_input)
    ax.text(x, y+1.2, model_labels[i], fontsize=9, fontweight='bold',
            ha='center', va='center', color=colors['text'])
    
    # TCN Block
    tcn_box = FancyBboxPatch((x-0.5, y+0.2), 1, 0.6,
                            boxstyle="round,pad=0.05",
                            facecolor=colors['tcn'],
                            edgecolor='green', linewidth=1.5)
    ax.add_patch(tcn_box)
    ax.text(x, y+0.5, 'TCN', fontsize=9, fontweight='bold',
            ha='center', va='center', color=colors['text'])
    
    # GRU Block
    gru_box = FancyBboxPatch((x-0.5, y-0.4), 1, 0.6,
                            boxstyle="round,pad=0.05",
                            facecolor=colors['gru'],
                            edgecolor='red', linewidth=1.5)
    ax.add_patch(gru_box)
    ax.text(x, y-0.1, 'GRU', fontsize=9, fontweight='bold',
            ha='center', va='center', color=colors['text'])
    
    # Prediction output
    pred_box = FancyBboxPatch((x-0.3, y-1), 0.6, 0.4,
                             boxstyle="round,pad=0.05",
                             facecolor=colors['output'],
                             edgecolor='gray', linewidth=1)
    ax.add_patch(pred_box)
    ax.text(x, y-0.8, f'Ŷ{i+1}', fontsize=9, fontweight='bold',
            ha='center', va='center', color=colors['text'])
    
    # Arrows
    ax.arrow(x, y+1, 0, -0.12, head_width=0.08, head_length=0.05, 
             fc=colors['arrow'], ec=colors['arrow'])
    ax.arrow(x, y+0.18, 0, -0.15, head_width=0.08, head_length=0.05,
             fc=colors['arrow'], ec=colors['arrow'])
    ax.arrow(x, y-0.42, 0, -0.15, head_width=0.08, head_length=0.05,
             fc=colors['arrow'], ec=colors['arrow'])

# Arrows from IMF Selection to individual models
for i, (x, y) in enumerate(imf_positions):
    if i < 3:  # Only draw for first 3 to avoid clutter
        start_x = 9.25 + (i-1) * 0.3
        ax.plot([start_x, x], [7, y+1.4], 'k--', alpha=0.6, linewidth=1)

# Aggregation Box
agg_box = FancyBboxPatch((6, 2.5), 2, 1,
                        boxstyle="round,pad=0.1",
                        facecolor=colors['output'],
                        edgecolor='black', linewidth=2)
ax.add_patch(agg_box)
ax.text(7, 3.2, 'Aggregation', fontsize=12, fontweight='bold',
        ha='center', va='center', color=colors['text'])
ax.text(7, 2.8, 'Ŷ(t) = Σ Ŷᵢ(t)', fontsize=10,
        ha='center', va='center', color=colors['text'])

# Arrows to aggregation
for i, (x, y) in enumerate(imf_positions):
    ax.plot([x, 7], [y-1, 3.5], 'k-', alpha=0.7, linewidth=1.5)
    # Add arrowhead
    dx = 7 - x
    dy = 3.5 - (y-1)
    length = np.sqrt(dx**2 + dy**2)
    dx_norm = dx / length
    dy_norm = dy / length
    ax.arrow(7 - 0.15*dx_norm, 3.5 - 0.15*dy_norm, 
             0.15*dx_norm, 0.15*dy_norm,
             head_width=0.08, head_length=0.1, 
             fc=colors['arrow'], ec=colors['arrow'])

# Final Output
output_box = FancyBboxPatch((6, 1), 2, 0.8,
                           boxstyle="round,pad=0.1",
                           facecolor=colors['input'],
                           edgecolor='black', linewidth=2)
ax.add_patch(output_box)
ax.text(7, 1.4, 'Final Forecast\nŶ(t)', fontsize=11, fontweight='bold',
        ha='center', va='center', color=colors['text'])

# Arrow from aggregation to output
arrow_final = ConnectionPatch((7, 2.5), (7, 1.8), "data", "data",
                             arrowstyle="->", shrinkA=0, shrinkB=0,
                             mutation_scale=20, fc=colors['arrow'], ec=colors['arrow'])
ax.add_artist(arrow_final)

# Model Optimization Box (side annotation)
opt_box = FancyBboxPatch((11.5, 2.5), 2.3, 2,
                        boxstyle="round,pad=0.1",
                        facecolor='#FFF3CD',
                        edgecolor='orange', linewidth=2)
ax.add_patch(opt_box)
ax.text(12.65, 3.8, 'Model Optimization', fontsize=10, fontweight='bold',
        ha='center', va='center', color=colors['text'])
ax.text(12.65, 3.4, '• Pruning (30% sparsity)', fontsize=9,
        ha='center', va='center', color=colors['text'])
ax.text(12.65, 3.1, '• Quantization (FP32→INT8)', fontsize=9,
        ha='center', va='center', color=colors['text'])
ax.text(12.65, 2.8, '• 75% memory reduction', fontsize=9,
        ha='center', va='center', color=colors['text'])

# TCN Architecture Details
ax.text(2.5, 0.5, 'TCN Details:\n• 3 dilated conv layers\n• Dilation rates: [1,2,4]\n• 64 filters, kernel=3\n• ReLU + Dropout(0.2)', 
        fontsize=8, ha='left', va='center', 
        bbox=dict(boxstyle="round,pad=0.3", facecolor=colors['tcn'], alpha=0.7))

# GRU Architecture Details  
ax.text(10.5, 0.5, 'GRU Details:\n• 128 hidden units\n• 2 layers\n• Dropout(0.3)\n• Recurrent dropout(0.2)', 
        fontsize=8, ha='left', va='center',
        bbox=dict(boxstyle="round,pad=0.3", facecolor=colors['gru'], alpha=0.7))

plt.tight_layout()
plt.savefig('/home/anhpt95te/Downloads/UAI paper/results/eemd_tgnet_architecture.png', 
            dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
plt.savefig('/home/anhpt95te/Downloads/UAI paper/results/eemd_tgnet_architecture.pdf', 
            dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
plt.close()

print("Architecture diagram created successfully!")
print("Files saved:")
print("- /home/anhpt95te/Downloads/UAI paper/results/eemd_tgnet_architecture.png")
print("- /home/anhpt95te/Downloads/UAI paper/results/eemd_tgnet_architecture.pdf")