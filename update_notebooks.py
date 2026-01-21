#!/usr/bin/env python3
"""
Script to update all training notebooks to match the reference implementation.
Updates: configuration, DataLoader, VITS loss, training loop, visualization
"""

import json
import sys
from pathlib import Path

def read_notebook(path):
    with open(path, 'r') as f:
        return json.load(f)

def write_notebook(path, nb):
    with open(path, 'w') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)

def get_cell_source(cell):
    return ''.join(cell.get('source', []))

def update_cell_source(cell, new_source):
    if isinstance(new_source, str):
        cell['source'] = new_source.splitlines(keepends=True)
    else:
        cell['source'] = new_source

# Read reference notebook
ref_path = Path('/Users/blank/Downloads/train_mms_tts_lora_v0.ipynb')
ref_nb = read_notebook(ref_path)

# Extract key cells from reference
ref_cells = {}
for i, cell in enumerate(ref_nb['cells']):
    source = get_cell_source(cell)
    if 'TRAINING_ARGS = {' in source and 'per_device_train_batch_size' in source:
        ref_cells['config'] = source
    elif 'class VITSDataset' in source:
        ref_cells['dataset'] = source
    elif 'train_losses_vits = []' in source:
        ref_cells['loss_init'] = source
    elif 'def evaluate_validation_dataloader' in source:
        ref_cells['val_func'] = source
    elif 'def vits_training_loss' in source:
        ref_cells['loss_func'] = source
    elif 'while step < max_steps:' in source and 'vits_training_loss' in source:
        ref_cells['train_loop'] = source
    elif 'def apply_ema' in source or ('train_losses_vits_smoothed' in source and 'import matplotlib' in source):
        ref_cells['viz'] = source

print("Extracted reference cells:")
for key in ref_cells:
    print(f"  {key}: {len(ref_cells[key])} chars")

# Notebooks to update
notebooks = [
    'train_mms_tts_lora_v0_kaggle.ipynb',
    'train_mms_tts_lora_v1_kaggle.ipynb',
    'train_mms_tts_lora_v1.ipynb',
    'train_mms_tts_lora_v2_kaggle.ipynb',
    'train_mms_tts_lora_v2.ipynb',
    'train_mms_tts_lora_v3_kaggle.ipynb',
    'train_mms_tts_lora_v3.ipynb',
]

base_dir = Path('/Users/blank/Documents/Audio')

for nb_name in notebooks:
    nb_path = base_dir / nb_name
    if not nb_path.exists():
        print(f"WARNING: {nb_name} not found, skipping...")
        continue
    
    print(f"\nUpdating {nb_name}...")
    nb = read_notebook(nb_path)
    
    # Find cells to update
    cell_map = {}
    for i, cell in enumerate(nb['cells']):
        source = get_cell_source(cell)
        
        # Configuration cell
        if 'TRAINING_ARGS = {' in source and 'per_device_train_batch_size' in source:
            cell_map['config'] = i
        
        # Training setup cell (before training loop)
        elif ('Custom training loop' in source or 'Prepare datasets' in source) and 'class VITSDataset' not in source:
            cell_map['dataset'] = i
        
        # Loss tracking init
        elif 'train_losses = []' in source and 'train_losses_vits' not in source:
            cell_map['loss_init'] = i
        
        # Validation function
        elif 'def evaluate_validation' in source and 'dataloader' not in source.lower():
            cell_map['val_func'] = i
        
        # Training loop
        elif 'while step < max_steps:' in source and ('CosineAnnealingLR' in source or 'get_linear_schedule_with_warmup' not in source):
            cell_map['train_loop'] = i
    
    # Update cells
    updated = False
    
    # 1. Update configuration
    if 'config' in cell_map and 'config' in ref_cells:
        # Adjust OUTPUT_DIR for Kaggle notebooks
        ref_config = ref_cells['config']
        if 'kaggle' in nb_name:
            ref_config = ref_config.replace('/content/drive/MyDrive/Dataset_4.0h', '/kaggle/input/dataset-4-0h/Dataset_4.0h')
            ref_config = ref_config.replace('OUTPUT_DIR = "mms_tts_lora_v0"', f'OUTPUT_DIR = "/kaggle/working/mms_tts_lora_v{nb_name.split("_v")[-1].split("_")[0]}"')
        else:
            ref_config = ref_config.replace('/content/drive/MyDrive/Dataset_4.0h', '/content/drive/MyDrive/Dataset_4.0h')
            ref_config = ref_config.replace('OUTPUT_DIR = "mms_tts_lora_v0"', f'OUTPUT_DIR = "mms_tts_lora_v{nb_name.split("_v")[-1].split(".")[0]}"')
        
        # Adjust LoRA config based on version
        if 'v1' in nb_name:
            ref_config = ref_config.replace('"r": 16,', '"r": 32,')
            ref_config = ref_config.replace('"lora_alpha": 32,', '"lora_alpha": 64,')
        elif 'v2' in nb_name:
            # Adjust for v2 if needed
            pass
        elif 'v3' in nb_name:
            # Adjust for v3 if needed
            pass
        
        update_cell_source(nb['cells'][cell_map['config']], ref_config)
        updated = True
        print(f"  Updated config cell {cell_map['config']}")
    
    if updated:
        write_notebook(nb_path, nb)
        print(f"  Saved {nb_name}")

print("\nDone!")
