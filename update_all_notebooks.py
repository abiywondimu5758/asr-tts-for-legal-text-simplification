#!/usr/bin/env python3
"""
Comprehensive script to update all training notebooks to match reference implementation.
Updates configuration, DataLoader, VITS loss, training loop, and visualization cells.
"""

import json
import sys
from pathlib import Path
import re

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
        cell['source'] = [line + '\n' if not line.endswith('\n') else line for line in new_source.splitlines(True)]
    else:
        cell['source'] = new_source

# Read reference notebook
ref_path = Path('/Users/blank/Downloads/train_mms_tts_lora_v0.ipynb')
print(f"Reading reference notebook: {ref_path}")
ref_nb = read_notebook(ref_path)

# Extract key cells from reference
ref_cells = {}
for i, cell in enumerate(ref_nb['cells']):
    if cell.get('cell_type') != 'code':
        continue
    source = get_cell_source(cell)
    
    if 'TRAINING_ARGS = {' in source and 'per_device_train_batch_size' in source:
        ref_cells['config'] = source
        print(f"  Extracted config from cell {i}")
    elif 'class VITSDataset' in source and 'def collate_fn' in source:
        ref_cells['dataset'] = source
        print(f"  Extracted dataset from cell {i}")
    elif 'train_losses_vits = []' in source and '# Optimized loss' in source:
        ref_cells['loss_init'] = source
        print(f"  Extracted loss_init from cell {i}")
    elif 'def evaluate_validation_dataloader' in source:
        ref_cells['val_func'] = source
        print(f"  Extracted val_func from cell {i}")
    elif 'def vits_training_loss' in source:
        ref_cells['loss_func'] = source
        print(f"  Extracted loss_func from cell {i}")
    elif 'while step < max_steps:' in source and 'get_linear_schedule_with_warmup' in source:
        ref_cells['train_loop'] = source
        print(f"  Extracted train_loop from cell {i}")
    elif 'train_losses_vits_smoothed' in source and 'import matplotlib' in source:
        ref_cells['viz_main'] = source
        print(f"  Extracted viz_main from cell {i}")

print(f"\nExtracted {len(ref_cells)} key cells from reference\n")

# Notebooks to update with their versions
notebooks = {
    'train_mms_tts_lora_v0_kaggle.ipynb': {'version': 'v0', 'is_kaggle': True, 'r': 16, 'alpha': 32},
    'train_mms_tts_lora_v1_kaggle.ipynb': {'version': 'v1', 'is_kaggle': True, 'r': 32, 'alpha': 64},
    'train_mms_tts_lora_v1.ipynb': {'version': 'v1', 'is_kaggle': False, 'r': 32, 'alpha': 64},
    'train_mms_tts_lora_v2_kaggle.ipynb': {'version': 'v2', 'is_kaggle': True, 'r': 64, 'alpha': 128},
    'train_mms_tts_lora_v2.ipynb': {'version': 'v2', 'is_kaggle': False, 'r': 64, 'alpha': 128},
    'train_mms_tts_lora_v3_kaggle.ipynb': {'version': 'v3', 'is_kaggle': True, 'r': 128, 'alpha': 256},
    'train_mms_tts_lora_v3.ipynb': {'version': 'v3', 'is_kaggle': False, 'r': 128, 'alpha': 256},
}

base_dir = Path('/Users/blank/Documents/Audio')

for nb_name, nb_info in notebooks.items():
    nb_path = base_dir / nb_name
    if not nb_path.exists():
        print(f"WARNING: {nb_name} not found, skipping...")
        continue
    
    print(f"\n{'='*70}")
    print(f"Updating {nb_name} (version={nb_info['version']}, kaggle={nb_info['is_kaggle']})")
    print(f"{'='*70}")
    
    nb = read_notebook(nb_path)
    updated_count = 0
    
    # Find cells to update
    cell_map = {}
    for i, cell in enumerate(nb['cells']):
        if cell.get('cell_type') != 'code':
            continue
        source = get_cell_source(cell)
        
        if 'TRAINING_ARGS = {' in source and 'per_device_train_batch_size' in source:
            cell_map['config'] = i
        elif ('Custom training loop' in source or 'Prepare datasets' in source) and 'class VITSDataset' not in source:
            cell_map['dataset'] = i
        elif 'train_losses = []' in source and 'train_losses_vits' not in source:
            cell_map['loss_init'] = i
        elif 'def evaluate_validation' in source and 'dataloader' not in source.lower():
            cell_map['val_func'] = i
        elif 'while step < max_steps:' in source and ('CosineAnnealingLR' in source or 'get_linear_schedule_with_warmup' not in source):
            cell_map['train_loop'] = i
        elif 'def vits_training_loss' not in source and 'vits_training_loss' in source and 'def ' not in source[:200]:
            # Loss function is inside training loop
            pass
    
    print(f"  Found {len(cell_map)} cells to update: {list(cell_map.keys())}")
    
    # 1. Update configuration cell
    if 'config' in cell_map and 'config' in ref_cells:
        config_source = ref_cells['config']
        
        # Adjust paths for Kaggle vs Colab
        if nb_info['is_kaggle']:
            config_source = config_source.replace('/content/drive/MyDrive/Dataset_4.0h', '/kaggle/input/dataset-4-0h/Dataset_4.0h')
            config_source = config_source.replace('OUTPUT_DIR = "mms_tts_lora_v0"', f'OUTPUT_DIR = "/kaggle/working/mms_tts_lora_{nb_info["version"]}"')
        else:
            config_source = config_source.replace('/content/drive/MyDrive/Dataset_4.0h', '/content/drive/MyDrive/Dataset_4.0h')
            config_source = config_source.replace('OUTPUT_DIR = "mms_tts_lora_v0"', f'OUTPUT_DIR = "mms_tts_lora_{nb_info["version"]}"')
        
        # Adjust LoRA config based on version
        config_source = re.sub(r'"r": \d+,', f'"r": {nb_info["r"]},', config_source)
        config_source = re.sub(r'"lora_alpha": \d+,', f'"lora_alpha": {nb_info["alpha"]},', config_source)
        
        update_cell_source(nb['cells'][cell_map['config']], config_source)
        updated_count += 1
        print(f"    ✓ Updated config cell {cell_map['config']}")
    
    # 2. Update dataset preparation cell
    if 'dataset' in cell_map and 'dataset' in ref_cells:
        dataset_source = ref_cells['dataset']
        update_cell_source(nb['cells'][cell_map['dataset']], dataset_source)
        updated_count += 1
        print(f"    ✓ Updated dataset cell {cell_map['dataset']}")
    
    # 3. Update loss init cell
    if 'loss_init' in cell_map and 'loss_init' in ref_cells:
        update_cell_source(nb['cells'][cell_map['loss_init']], ref_cells['loss_init'])
        updated_count += 1
        print(f"    ✓ Updated loss_init cell {cell_map['loss_init']}")
    
    # 4. Update validation function cell
    if 'val_func' in cell_map and 'val_func' in ref_cells:
        # Need to check if there's a cell before training loop that needs the validation function
        # or if we need to add it before the training loop
        val_source = ref_cells['val_func']
        # Check if this cell already has both validation functions
        current_source = get_cell_source(nb['cells'][cell_map['val_func']])
        if 'evaluate_validation_dataloader' not in current_source:
            # Add the dataloader-based validation function
            val_source = val_source  # Use the full reference validation function
            update_cell_source(nb['cells'][cell_map['val_func']], val_source)
            updated_count += 1
            print(f"    ✓ Updated val_func cell {cell_map['val_func']}")
    
    # 5. Update training loop (this is the most complex)
    if 'train_loop' in cell_map and 'train_loop' in ref_cells and 'loss_func' in ref_cells:
        # Combine loss function and training loop
        train_source = ref_cells['loss_func'] + '\n\n' + ref_cells['train_loop']
        update_cell_source(nb['cells'][cell_map['train_loop']], train_source)
        updated_count += 1
        print(f"    ✓ Updated train_loop cell {cell_map['train_loop']} (includes loss function)")
    
    # Save notebook
    if updated_count > 0:
        write_notebook(nb_path, nb)
        print(f"\n  ✓ Saved {nb_name} ({updated_count} cells updated)")
    else:
        print(f"\n  ⚠ No cells updated in {nb_name}")

print(f"\n{'='*70}")
print("Update complete!")
print(f"{'='*70}")
