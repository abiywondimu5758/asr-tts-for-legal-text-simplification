#!/usr/bin/env python3
"""
Update visualization cells in all MMS-TTS LoRA training notebooks with:
- EMA smoothing (alpha=0.3) for train_losses_vits
- Prioritize VITS loss in plots
- Remove WER/CER sections
- Add safety checks for list lengths
"""

import json
import re
from pathlib import Path

def read_notebook(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def write_notebook(path, nb):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)

def get_cell_source(cell):
    return ''.join(cell.get('source', []))

def update_cell_source(cell, new_source):
    if isinstance(new_source, str):
        lines = new_source.splitlines(True)
        if not lines or not lines[-1].endswith('\n'):
            lines.append('\n')
        cell['source'] = lines
    else:
        cell['source'] = new_source

# EMA smoothing function and improved main visualization
main_viz_source = '''import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# EMA smoothing function
def apply_ema(values, alpha=0.3):
    """Apply Exponential Moving Average smoothing to a list of values"""
    if not values or len(values) == 0:
        return []
    smoothed = [values[0]]
    for val in values[1:]:
        smoothed.append(alpha * val + (1 - alpha) * smoothed[-1])
    return smoothed

# Check if losses were tracked
if 'train_losses' not in globals() or not train_losses:
    print("Warning: Training losses not tracked. Please ensure the training loop tracks losses.")
    train_losses = []
    train_steps = []
    val_losses = []
    val_steps = []
    train_losses_vits = []

# Initialize train_losses_vits if it doesn't exist
if 'train_losses_vits' not in globals():
    train_losses_vits = []

# Apply EMA smoothing to VITS training loss
train_losses_vits_smoothed = []
if train_losses_vits and len(train_losses_vits) > 0:
    train_losses_vits_smoothed = apply_ema(train_losses_vits, alpha=0.3)

# Safety check: Ensure all lists have the same length for plotting
min_len = min(
    len(train_steps) if train_steps else float('inf'),
    len(train_losses) if train_losses else float('inf'),
    len(train_losses_vits) if train_losses_vits else float('inf')
)
if min_len != float('inf') and min_len > 0:
    train_steps = train_steps[:min_len]
    train_losses = train_losses[:min_len]
    train_losses_vits = train_losses_vits[:min_len]
    if len(train_losses_vits_smoothed) > min_len:
        train_losses_vits_smoothed = train_losses_vits_smoothed[:min_len]

# Create comprehensive figure
fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

# Plot 1: Training Loss Over Time (top, spans 2 columns)
ax1 = fig.add_subplot(gs[0, :])
if train_losses_vits and train_steps and len(train_losses_vits) == len(train_steps):
    # Plot smoothed VITS loss as primary (thick green solid line)
    if train_losses_vits_smoothed and len(train_losses_vits_smoothed) == len(train_steps):
        ax1.plot(train_steps, train_losses_vits_smoothed, 'g-', linewidth=3, label='Training Loss (VITS, Smoothed)', alpha=0.9)
    # Plot raw VITS loss (thinner dashed green line)
    ax1.plot(train_steps, train_losses_vits, 'g--', linewidth=1.5, label='Training Loss (VITS, Raw)', alpha=0.6)
    # Plot optimized loss (very thin dashed blue line)
    if train_losses and len(train_losses) == len(train_steps):
        ax1.plot(train_steps, train_losses, 'b--', linewidth=1, label='Training Loss (Optimized)', alpha=0.4)
    ax1.set_xlabel('Training Steps', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Loss', fontsize=13, fontweight='bold')
    ax1.set_title('Training Loss Over Time', fontsize=15, fontweight='bold', pad=15)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.legend(fontsize=11, loc='best')
    ax1.set_facecolor('#f8f9fa')
else:
    ax1.text(0.5, 0.5, 'No training data available\\nPlease track losses during training', ha='center', va='center', fontsize=12)
    ax1.set_title('Training Loss Over Time', fontsize=15, fontweight='bold', pad=15)

# Plot 2: Validation Loss
ax2 = fig.add_subplot(gs[1, 0])
if val_losses and val_steps and len(val_losses) == len(val_steps):
    ax2.plot(val_steps, val_losses, 'r-', linewidth=2.5, marker='o', markersize=5, label='Validation Loss', alpha=0.8)
    ax2.set_xlabel('Training Steps', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Loss', fontsize=12, fontweight='bold')
    ax2.set_title('Validation Loss', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.legend(fontsize=11)
    ax2.set_facecolor('#f8f9fa')
else:
    ax2.text(0.5, 0.5, 'No validation data', ha='center', va='center', fontsize=12)
    ax2.set_title('Validation Loss', fontsize=14, fontweight='bold')

# Plot 3: Combined Loss Plot (Training vs Validation)
ax3 = fig.add_subplot(gs[1, 1])
if train_losses_vits_smoothed and train_steps and len(train_losses_vits_smoothed) == len(train_steps):
    # Prioritize smoothed VITS loss for comparison with validation
    ax3.plot(train_steps, train_losses_vits_smoothed, 'g-', linewidth=2.5, label='Training Loss (VITS, Smoothed)', alpha=0.9)
    # Show raw VITS loss as secondary line
    if train_losses_vits and len(train_losses_vits) == len(train_steps):
        ax3.plot(train_steps, train_losses_vits, 'g--', linewidth=1, label='Training Loss (VITS, Raw)', alpha=0.5)
    # Show optimized loss as secondary line
    if train_losses and len(train_losses) == len(train_steps):
        ax3.plot(train_steps, train_losses, 'b--', linewidth=0.8, label='Training Loss (Optimized)', alpha=0.4)
    # Add validation loss
    if val_losses and val_steps and len(val_losses) == len(val_steps):
        ax3.plot(val_steps, val_losses, 'r-', linewidth=2, marker='o', markersize=4, label='Validation Loss', alpha=0.8)
    ax3.set_xlabel('Training Steps', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Loss', fontsize=12, fontweight='bold')
    ax3.set_title('Training vs Validation Loss', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3, linestyle='--')
    ax3.legend(fontsize=10, loc='best')
    ax3.set_facecolor('#f8f9fa')
else:
    ax3.text(0.5, 0.5, 'No training data', ha='center', va='center', fontsize=12)
    ax3.set_title('Training vs Validation Loss', fontsize=14, fontweight='bold')

# Plot 4: Training Statistics Summary
ax4 = fig.add_subplot(gs[2, :])
ax4.axis('off')
if train_losses_vits and len(train_losses_vits) > 0:
    summary_text = f"Training Summary: {OUTPUT_DIR}\\n"
    summary_text += f"Training Loss (VITS): Initial={train_losses_vits[0]:.6f}, Final={train_losses_vits[-1]:.6f}, Best={min(train_losses_vits):.6f}\\n"
    if train_losses and len(train_losses) > 0:
        summary_text += f"Training Loss (Optimized): Initial={train_losses[0]:.6f}, Final={train_losses[-1]:.6f}, Best={min(train_losses):.6f}\\n"
    if val_losses and len(val_losses) > 0:
        summary_text += f"Validation Loss: Initial={val_losses[0]:.6f}, Final={val_losses[-1]:.6f}, Best={min(val_losses):.6f}"
    ax4.text(0.5, 0.5, summary_text, ha='center', va='center', fontsize=12, family='monospace')
else:
    ax4.text(0.5, 0.5, 'No training statistics available', ha='center', va='center', fontsize=12)

# Add overall title
fig.suptitle(f'Training Progress: {OUTPUT_DIR}', fontsize=18, fontweight='bold', y=0.995)

plt.tight_layout(rect=[0, 0, 1, 0.99])
plt.show()

# Save high-resolution plot
plot_path = f"{OUTPUT_DIR}_training_plots.png"
fig.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
print(f"\\nTraining plots saved to: {plot_path}")

'''

# Detailed summary visualization (Cell 33 equivalent)
detailed_viz_source = '''import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# EMA smoothing function (if not already defined)
def apply_ema(values, alpha=0.3):
    """Apply Exponential Moving Average smoothing to a list of values"""
    if not values or len(values) == 0:
        return []
    smoothed = [values[0]]
    for val in values[1:]:
        smoothed.append(alpha * val + (1 - alpha) * smoothed[-1])
    return smoothed

# Check if losses were tracked
if 'train_losses' not in globals() or not train_losses:
    print("Warning: Training losses not tracked. Please ensure the training loop tracks losses.")
    train_losses = []
    train_steps = []
    val_losses = []
    val_steps = []
    train_losses_vits = []

# Initialize train_losses_vits if it doesn't exist
if 'train_losses_vits' not in globals():
    train_losses_vits = []

# Apply EMA smoothing to VITS training loss
train_losses_vits_smoothed = []
if train_losses_vits and len(train_losses_vits) > 0:
    train_losses_vits_smoothed = apply_ema(train_losses_vits, alpha=0.3)

# Safety check: Ensure all lists have the same length for plotting
min_len = min(
    len(train_steps) if train_steps else float('inf'),
    len(train_losses) if train_losses else float('inf'),
    len(train_losses_vits) if train_losses_vits else float('inf')
)
if min_len != float('inf') and min_len > 0:
    train_steps = train_steps[:min_len]
    train_losses = train_losses[:min_len]
    train_losses_vits = train_losses_vits[:min_len]
    if len(train_losses_vits_smoothed) > min_len:
        train_losses_vits_smoothed = train_losses_vits_smoothed[:min_len]

# Create comprehensive figure
fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

# Plot 1: Training Loss (top left, spans 2 columns)
ax1 = fig.add_subplot(gs[0, :])
if train_losses_vits and train_steps and len(train_losses_vits) == len(train_steps):
    # Plot smoothed VITS loss as primary (thick green solid line)
    if train_losses_vits_smoothed and len(train_losses_vits_smoothed) == len(train_steps):
        ax1.plot(train_steps, train_losses_vits_smoothed, 'g-', linewidth=3, label='Training Loss (VITS, Smoothed)', alpha=0.9)
    # Plot raw VITS loss (thinner dashed green line)
    ax1.plot(train_steps, train_losses_vits, 'g--', linewidth=1.5, label='Training Loss (VITS, Raw)', alpha=0.6)
    # Plot optimized loss (very thin dashed blue line)
    if train_losses and len(train_losses) == len(train_steps):
        ax1.plot(train_steps, train_losses, 'b--', linewidth=1, label='Training Loss (Optimized)', alpha=0.4)
    ax1.set_xlabel('Training Steps', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Loss', fontsize=13, fontweight='bold')
    ax1.set_title('Training Loss Over Time', fontsize=15, fontweight='bold', pad=15)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.legend(fontsize=11, loc='best')
    ax1.set_facecolor('#f8f9fa')
else:
    ax1.text(0.5, 0.5, 'No training data available\\nPlease track losses during training', ha='center', va='center', fontsize=12)
    ax1.set_title('Training Loss Over Time', fontsize=15, fontweight='bold', pad=15)

# Plot 2: Validation Loss
ax2 = fig.add_subplot(gs[1, 0])
if val_losses and val_steps and len(val_losses) == len(val_steps):
    ax2.plot(val_steps, val_losses, 'r-', linewidth=2.5, marker='o', markersize=5, label='Validation Loss', alpha=0.8)
    ax2.set_xlabel('Training Steps', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Loss', fontsize=12, fontweight='bold')
    ax2.set_title('Validation Loss', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.legend(fontsize=11)
    ax2.set_facecolor('#f8f9fa')
else:
    ax2.text(0.5, 0.5, 'No validation data', ha='center', va='center', fontsize=12)
    ax2.set_title('Validation Loss', fontsize=14, fontweight='bold')

# Plot 3: Combined Loss Plot (Training vs Validation)
ax3 = fig.add_subplot(gs[1, 1])
if train_losses_vits_smoothed and train_steps and len(train_losses_vits_smoothed) == len(train_steps):
    # Prioritize smoothed VITS loss for comparison with validation
    ax3.plot(train_steps, train_losses_vits_smoothed, 'g-', linewidth=2.5, label='Training Loss (VITS, Smoothed)', alpha=0.9)
    # Show raw VITS loss as secondary line
    if train_losses_vits and len(train_losses_vits) == len(train_steps):
        ax3.plot(train_steps, train_losses_vits, 'g--', linewidth=1, label='Training Loss (VITS, Raw)', alpha=0.5)
    # Show optimized loss as secondary line
    if train_losses and len(train_losses) == len(train_steps):
        ax3.plot(train_steps, train_losses, 'b--', linewidth=0.8, label='Training Loss (Optimized)', alpha=0.4)
    # Add validation loss
    if val_losses and val_steps and len(val_losses) == len(val_steps):
        ax3.plot(val_steps, val_losses, 'r-', linewidth=2, marker='o', markersize=4, label='Validation Loss', alpha=0.8)
    ax3.set_xlabel('Training Steps', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Loss', fontsize=12, fontweight='bold')
    ax3.set_title('Training vs Validation Loss', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3, linestyle='--')
    ax3.legend(fontsize=10, loc='best')
    ax3.set_facecolor('#f8f9fa')
else:
    ax3.text(0.5, 0.5, 'No training data', ha='center', va='center', fontsize=12)
    ax3.set_title('Training vs Validation Loss', fontsize=14, fontweight='bold')

# Plot 4: Training Statistics Summary
ax4 = fig.add_subplot(gs[2, :])
ax4.axis('off')
if train_losses_vits and len(train_losses_vits) > 0:
    summary_text = f"Training Summary: {OUTPUT_DIR}\\n"
    summary_text += f"Training Loss (VITS): Initial={train_losses_vits[0]:.6f}, Final={train_losses_vits[-1]:.6f}, Best={min(train_losses_vits):.6f}\\n"
    if train_losses and len(train_losses) > 0:
        summary_text += f"Training Loss (Optimized): Initial={train_losses[0]:.6f}, Final={train_losses[-1]:.6f}, Best={min(train_losses):.6f}\\n"
    if val_losses and len(val_losses) > 0:
        summary_text += f"Validation Loss: Initial={val_losses[0]:.6f}, Final={val_losses[-1]:.6f}, Best={min(val_losses):.6f}"
    ax4.text(0.5, 0.5, summary_text, ha='center', va='center', fontsize=12, family='monospace')
else:
    ax4.text(0.5, 0.5, 'No training statistics available', ha='center', va='center', fontsize=12)

# Add overall title
fig.suptitle(f'Training Progress: {OUTPUT_DIR}', fontsize=18, fontweight='bold', y=0.995)

plt.tight_layout(rect=[0, 0, 1, 0.99])
plt.show()

# Save high-resolution plot
plot_path = f"{OUTPUT_DIR}_training_plots.png"
fig.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
print(f"\\nTraining plots saved to: {plot_path}")

# Print detailed summary statistics
print("\\n" + "="*70)
print(f"TRAINING SUMMARY: {OUTPUT_DIR}")
print("="*70)
if train_losses_vits and len(train_losses_vits) > 0:
    print(f"\\nTraining Loss (VITS):")
    print(f" Initial: {train_losses_vits[0]:.6f}")
    print(f" Final: {train_losses_vits[-1]:.6f}")
    print(f" Best: {min(train_losses_vits):.6f} (at step {train_steps[train_losses_vits.index(min(train_losses_vits))]})")
    print(f" Improvement: {((train_losses_vits[0] - min(train_losses_vits)) / train_losses_vits[0] * 100):.2f}%")
if train_losses and len(train_losses) > 0:
    print(f"\\nTraining Loss (Optimized):")
    print(f" Initial: {train_losses[0]:.6f}")
    print(f" Final: {train_losses[-1]:.6f}")
    print(f" Best: {min(train_losses):.6f} (at step {train_steps[train_losses.index(min(train_losses))]})")
if val_losses and len(val_losses) > 0:
    print(f"\\nValidation Loss:")
    print(f" Initial: {val_losses[0]:.6f}")
    print(f" Final: {val_losses[-1]:.6f}")
    print(f" Best: {min(val_losses):.6f} (at step {val_steps[val_losses.index(min(val_losses))]})")
    print(f" Improvement: {((val_losses[0] - min(val_losses)) / val_losses[0] * 100):.2f}%")
print("="*70)

'''

# Epoch-based loss plot (Cell 36 equivalent)
epoch_viz_source = '''# Training and Validation Loss Plot (Epoch-based)
import matplotlib.pyplot as plt
import numpy as np

# EMA smoothing function (if not already defined)
def apply_ema(values, alpha=0.3):
    """Apply Exponential Moving Average smoothing to a list of values"""
    if not values or len(values) == 0:
        return []
    smoothed = [values[0]]
    for val in values[1:]:
        smoothed.append(alpha * val + (1 - alpha) * smoothed[-1])
    return smoothed

# Check if losses were tracked
if 'train_losses' not in globals() or not train_losses:
    print("Warning: Training losses not tracked. Please ensure the training loop tracks losses.")
    train_losses = []
    train_steps = []
    val_losses = []
    val_steps = []
    train_losses_vits = []

# Initialize train_losses_vits if it doesn't exist
if 'train_losses_vits' not in globals():
    train_losses_vits = []

# Apply EMA smoothing to VITS training loss
train_losses_vits_smoothed = []
if train_losses_vits and len(train_losses_vits) > 0:
    train_losses_vits_smoothed = apply_ema(train_losses_vits, alpha=0.3)

# Safety check: Ensure all lists have the same length for plotting
min_len = min(
    len(train_losses) if train_losses else float('inf'),
    len(train_losses_vits) if train_losses_vits else float('inf')
)
if min_len != float('inf') and min_len > 0:
    train_losses = train_losses[:min_len]
    train_losses_vits = train_losses_vits[:min_len]
    if len(train_losses_vits_smoothed) > min_len:
        train_losses_vits_smoothed = train_losses_vits_smoothed[:min_len]

# For epoch-based plot, we'll use steps as approximate epochs
if train_losses_vits and len(train_losses_vits) > 0:
    epochs = list(range(len(train_losses_vits)))
    
    plt.figure(figsize=(10, 6))
    # Plot smoothed VITS loss as primary
    if train_losses_vits_smoothed and len(train_losses_vits_smoothed) == len(epochs):
        plt.plot(epochs, train_losses_vits_smoothed, 'g-', linewidth=2.5, label='Training Loss (VITS, Smoothed)', alpha=0.9)
    # Plot raw VITS loss
    plt.plot(epochs, train_losses_vits, 'g--', linewidth=1.5, label='Training Loss (VITS, Raw)', alpha=0.7)
    # Plot optimized loss
    if train_losses and len(train_losses) == len(epochs):
        plt.plot(epochs, train_losses, 'b--', linewidth=1, label='Training Loss (Optimized)', alpha=0.5)
    # Add validation loss
    if val_losses and len(val_losses) > 0:
        val_epochs = list(range(len(val_losses)))
        plt.plot(val_epochs, val_losses, 'orange', linewidth=2, marker='o', markersize=4, label='Validation Loss', alpha=0.8)
    
    plt.xlabel('Steps (Epochs)', fontsize=12, fontweight='bold')
    plt.ylabel('Loss', fontsize=12, fontweight='bold')
    plt.title('Training and Validation Loss', fontsize=14, fontweight='bold', pad=15)
    plt.legend(fontsize=11, loc='best')
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.show()
    
    # Save the plot
    plot_path = f"{OUTPUT_DIR}_loss_epochs.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Epoch-based loss plot saved to: {plot_path}")
    
    # Print summary
    print("\\n" + "="*70)
    print(f"EPOCH-BASED SUMMARY: {OUTPUT_DIR}")
    print("="*70)
    if train_losses_vits and len(train_losses_vits) > 0:
        print(f"\\nTraining Loss (VITS):")
        print(f" Initial: {train_losses_vits[0]:.6f}")
        print(f" Final: {train_losses_vits[-1]:.6f}")
        print(f" Best: {min(train_losses_vits):.6f}")
    if train_losses and len(train_losses) > 0:
        print(f"\\nTraining Loss (Optimized):")
        print(f" Initial: {train_losses[0]:.6f}")
        print(f" Final: {train_losses[-1]:.6f}")
    if val_losses and len(val_losses) > 0:
        print(f"\\nValidation Loss:")
        print(f" Initial: {val_losses[0]:.6f}")
        print(f" Final: {val_losses[-1]:.6f}")
        print(f" Best: {min(val_losses):.6f}")
    print("="*70)
else:
    print("Warning: No training losses available. Please track losses during training.")

'''

notebooks = {
    'train_mms_tts_lora_v0_kaggle.ipynb': {'version': 'v0', 'is_kaggle': True},
    'train_mms_tts_lora_v1_kaggle.ipynb': {'version': 'v1', 'is_kaggle': True},
    'train_mms_tts_lora_v1.ipynb': {'version': 'v1', 'is_kaggle': False},
    'train_mms_tts_lora_v2_kaggle.ipynb': {'version': 'v2', 'is_kaggle': True},
    'train_mms_tts_lora_v2.ipynb': {'version': 'v2', 'is_kaggle': False},
    'train_mms_tts_lora_v3_kaggle.ipynb': {'version': 'v3', 'is_kaggle': True},
    'train_mms_tts_lora_v3.ipynb': {'version': 'v3', 'is_kaggle': False},
}

base_dir = Path('/Users/blank/Documents/Audio')

for nb_name, nb_info in notebooks.items():
    nb_path = base_dir / nb_name
    if not nb_path.exists():
        print(f"⚠ WARNING: {nb_name} not found, skipping...")
        continue
    
    print(f"\nUpdating {nb_name}...")
    nb = read_notebook(nb_path)
    
    # Find visualization cells
    viz_cells = {}
    for i, cell in enumerate(nb['cells']):
        if cell.get('cell_type') != 'code':
            continue
        source = get_cell_source(cell)
        
        # Main visualization (Cell 27 equivalent - first comprehensive visualization)
        if 'Training Loss Over Time' in source and 'Training vs Validation Loss' in source and not viz_cells.get('main'):
            viz_cells['main'] = i
        # Detailed visualization (Cell 33 equivalent - second comprehensive with WER/CER)
        elif 'Word Error Rate' in source or ('WER' in source and 'CER' in source and not viz_cells.get('detailed')):
            viz_cells['detailed'] = i
        # Epoch-based plot (Cell 36 equivalent)
        elif 'Training and Validation Loss' in source and 'Epoch' in source and 'epoch' in source.lower():
            viz_cells['epoch'] = i
    
    print(f"  Found visualization cells: {list(viz_cells.keys())}")
    
    updated = False
    
    # Update main visualization (Cell 27)
    if 'main' in viz_cells:
        update_cell_source(nb['cells'][viz_cells['main']], main_viz_source)
        print(f"    ✓ Updated main visualization cell {viz_cells['main']}")
        updated = True
    
    # Update detailed visualization (Cell 33) - replace with improved version without WER/CER
    if 'detailed' in viz_cells:
        update_cell_source(nb['cells'][viz_cells['detailed']], detailed_viz_source)
        print(f"    ✓ Updated detailed visualization cell {viz_cells['detailed']} (removed WER/CER)")
        updated = True
    
    # Update epoch-based plot (Cell 36)
    if 'epoch' in viz_cells:
        update_cell_source(nb['cells'][viz_cells['epoch']], epoch_viz_source)
        print(f"    ✓ Updated epoch-based plot cell {viz_cells['epoch']}")
        updated = True
    
    # Save notebook
    if updated:
        write_notebook(nb_path, nb)
        print(f"  ✓ Saved {nb_name}")
    else:
        print(f"  ⚠ No visualization cells found or updated")

print(f"\n{'='*70}")
print("Visualization cells update complete!")
print(f"{'='*70}")
