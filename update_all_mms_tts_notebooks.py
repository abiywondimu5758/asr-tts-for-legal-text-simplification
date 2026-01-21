#!/usr/bin/env python3
"""
Comprehensive script to update all MMS-TTS LoRA training notebooks systematically.
Applies all improvements from the reference notebook to all target notebooks.
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
        # Preserve line endings properly
        lines = new_source.splitlines(True)
        if not lines or not lines[-1].endswith('\n'):
            lines.append('\n')
        cell['source'] = lines
    else:
        cell['source'] = new_source

# Read reference notebook
ref_path = Path('/Users/blank/Downloads/train_mms_tts_lora_v0.ipynb')
print(f"Reading reference notebook: {ref_path}")
ref_nb = read_notebook(ref_path)

# Extract all key cells from reference
ref_cells = {}
for i, cell in enumerate(ref_nb['cells']):
    if cell.get('cell_type') != 'code':
        continue
    source = get_cell_source(cell)
    
    if 'TRAINING_ARGS = {' in source and 'per_device_train_batch_size' in source:
        ref_cells['config'] = (i, source)
    elif 'class VITSDataset' in source and 'def collate_fn' in source:
        ref_cells['dataset'] = (i, source)
    elif 'train_losses_vits = []' in source and '# Optimized loss' in source:
        ref_cells['loss_init'] = (i, source)
    elif 'def evaluate_validation_dataloader' in source:
        ref_cells['val_func'] = (i, source)
    elif 'def vits_training_loss' in source:
        ref_cells['loss_func'] = (i, source)
    elif 'while step < max_steps:' in source and 'DataLoader' in source:
        ref_cells['train_loop'] = (i, source)
    elif 'train_losses_vits_smoothed' in source and 'def apply_ema' in source:
        ref_cells['viz_main'] = (i, source)

print(f"Extracted {len(ref_cells)} key cells from reference:")
for key, (idx, _) in ref_cells.items():
    print(f"  {key}: cell {idx}")

# Notebooks to update with metadata
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
        print(f"\n⚠ WARNING: {nb_name} not found, skipping...")
        continue
    
    print(f"\n{'='*70}")
    print(f"Updating {nb_name}")
    print(f"  Version: {nb_info['version']}, Kaggle: {nb_info['is_kaggle']}, LoRA: r={nb_info['r']}, alpha={nb_info['alpha']}")
    print(f"{'='*70}")
    
    nb = read_notebook(nb_path)
    updated_cells = []
    
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
        elif 'def evaluate_validation' in source and 'dataloader' not in source.lower() and 'evaluate_validation_dataloader' not in source:
            cell_map['val_func'] = i
        elif 'while step < max_steps:' in source and 'CosineAnnealingLR' in source:
            cell_map['train_loop'] = i
    
    print(f"  Found {len(cell_map)} cells to update: {list(cell_map.keys())}")
    
    # 1. Update configuration
    if 'config' in cell_map and 'config' in ref_cells:
        _, config_source = ref_cells['config']
        
        # Adjust paths
        if nb_info['is_kaggle']:
            config_source = config_source.replace('/content/drive/MyDrive/Dataset_4.0h', '/kaggle/input/dataset-4-0h/Dataset_4.0h')
            config_source = re.sub(r'OUTPUT_DIR = "[^"]*"', f'OUTPUT_DIR = "/kaggle/working/mms_tts_lora_{nb_info["version"]}"', config_source)
        else:
            config_source = config_source.replace('/content/drive/MyDrive/Dataset_4.0h', '/content/drive/MyDrive/Dataset_4.0h')
            config_source = re.sub(r'OUTPUT_DIR = "[^"]*"', f'OUTPUT_DIR = "mms_tts_lora_{nb_info["version"]}"', config_source)
        
        # Adjust LoRA config
        config_source = re.sub(r'"r": \d+,', f'"r": {nb_info["r"]},', config_source)
        config_source = re.sub(r'"lora_alpha": \d+,', f'"lora_alpha": {nb_info["alpha"]},', config_source)
        
        update_cell_source(nb['cells'][cell_map['config']], config_source)
        updated_cells.append(('config', cell_map['config']))
        print(f"    ✓ Updated config cell {cell_map['config']}")
    
    # 2. Update dataset preparation
    if 'dataset' in cell_map and 'dataset' in ref_cells:
        _, dataset_source = ref_cells['dataset']
        update_cell_source(nb['cells'][cell_map['dataset']], dataset_source)
        updated_cells.append(('dataset', cell_map['dataset']))
        print(f"    ✓ Updated dataset cell {cell_map['dataset']}")
    
    # 3. Update loss init
    if 'loss_init' in cell_map and 'loss_init' in ref_cells:
        _, loss_init_source = ref_cells['loss_init']
        update_cell_source(nb['cells'][cell_map['loss_init']], loss_init_source)
        updated_cells.append(('loss_init', cell_map['loss_init']))
        print(f"    ✓ Updated loss_init cell {cell_map['loss_init']}")
    
    # 4. Update validation function
    if 'val_func' in cell_map and 'val_func' in ref_cells:
        _, val_func_source = ref_cells['val_func']
        update_cell_source(nb['cells'][cell_map['val_func']], val_func_source)
        updated_cells.append(('val_func', cell_map['val_func']))
        print(f"    ✓ Updated val_func cell {cell_map['val_func']}")
    
    # 5. Update training loop (combine loss function and training loop)
    if 'train_loop' in cell_map:
        if 'loss_func' in ref_cells and 'train_loop' in ref_cells:
            _, loss_func_source = ref_cells['loss_func']
            _, train_loop_source = ref_cells['train_loop']
            
            # Combine them - loss function first, then training loop
            combined_source = loss_func_source + '\n\n' + train_loop_source
            
            update_cell_source(nb['cells'][cell_map['train_loop']], combined_source)
            updated_cells.append(('train_loop', cell_map['train_loop']))
            print(f"    ✓ Updated train_loop cell {cell_map['train_loop']} (with loss function)")
    
    # Save notebook
    if updated_cells:
        write_notebook(nb_path, nb)
        print(f"\n  ✓ Saved {nb_name} ({len(updated_cells)} cells updated)")
    else:
        print(f"\n  ⚠ No cells updated")

print(f"\n{'='*70}")
print("Update complete!")
print(f"{'='*70}")
