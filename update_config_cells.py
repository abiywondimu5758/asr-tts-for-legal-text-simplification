#!/usr/bin/env python3
"""
Update configuration cells in all MMS-TTS LoRA notebooks with correct training arguments:
- Learning rate: 5e-5 → 5e-6
- Warmup steps: 100 → 500
- Eval steps: 500 → 250
- Save steps: 500 → 250
- Logging steps: 100 → 50
- Batch sizes: per_device_train_batch_size: 8, per_device_eval_batch_size: 4
- Gradient accumulation: 8 (keep if already 8, update if different)
- Save total limit: 3 → 5
- Max steps: 3000 (keep if already 3000)
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
        print(f"⚠ WARNING: {nb_name} not found, skipping...")
        continue
    
    print(f"\nUpdating {nb_name}...")
    nb = read_notebook(nb_path)
    
    # Find configuration cell
    config_cell_idx = None
    for i, cell in enumerate(nb['cells']):
        if cell.get('cell_type') != 'code':
            continue
        source = get_cell_source(cell)
        if 'TRAINING_ARGS = {' in source and 'per_device_train_batch_size' in source:
            config_cell_idx = i
            break
    
    if config_cell_idx is None:
        print(f"  ⚠ Could not find configuration cell in {nb_name}")
        continue
    
    # Get the configuration cell source
    config_source = get_cell_source(nb['cells'][config_cell_idx])
    
    # Apply updates
    print(f"  Updating configuration cell {config_cell_idx}...")
    
    # 1. Update learning rate: 5e-5 → 5e-6
    config_source = re.sub(
        r'"learning_rate":\s*5e-5',
        '"learning_rate": 5e-6',
        config_source
    )
    config_source = re.sub(
        r'"learning_rate":\s*1e-5',
        '"learning_rate": 5e-6',
        config_source
    )
    
    # 2. Update warmup_steps: 100 → 500
    config_source = re.sub(
        r'"warmup_steps":\s*100',
        '"warmup_steps": 500',
        config_source
    )
    config_source = re.sub(
        r'"warmup_steps":\s*\d+',
        '"warmup_steps": 500',
        config_source
    )
    
    # 3. Update eval_steps: 500 → 250
    config_source = re.sub(
        r'"eval_steps":\s*500',
        '"eval_steps": 250',
        config_source
    )
    config_source = re.sub(
        r'"eval_steps":\s*\d+',
        '"eval_steps": 250',
        config_source
    )
    
    # 4. Update save_steps: 500 → 250
    config_source = re.sub(
        r'"save_steps":\s*500',
        '"save_steps": 250',
        config_source
    )
    config_source = re.sub(
        r'"save_steps":\s*\d+',
        '"save_steps": 250',
        config_source
    )
    
    # 5. Update logging_steps: 100 → 50
    config_source = re.sub(
        r'"logging_steps":\s*100',
        '"logging_steps": 50',
        config_source
    )
    config_source = re.sub(
        r'"logging_steps":\s*\d+',
        '"logging_steps": 50',
        config_source
    )
    
    # 6. Update per_device_train_batch_size: 2 → 8 (or keep if already 8)
    config_source = re.sub(
        r'"per_device_train_batch_size":\s*2',
        '"per_device_train_batch_size": 8',
        config_source
    )
    config_source = re.sub(
        r'"per_device_train_batch_size":\s*\d+',
        lambda m: '"per_device_train_batch_size": 8' if int(re.search(r'\d+', m.group(0)).group(0)) < 8 else m.group(0),
        config_source
    )
    
    # 7. Update per_device_eval_batch_size: 2 → 4 (or keep if already 4)
    config_source = re.sub(
        r'"per_device_eval_batch_size":\s*2',
        '"per_device_eval_batch_size": 4',
        config_source
    )
    config_source = re.sub(
        r'"per_device_eval_batch_size":\s*\d+',
        lambda m: '"per_device_eval_batch_size": 4' if int(re.search(r'\d+', m.group(0)).group(0)) < 4 else m.group(0),
        config_source
    )
    
    # 8. Update gradient_accumulation_steps: ensure it's 8
    config_source = re.sub(
        r'"gradient_accumulation_steps":\s*\d+',
        '"gradient_accumulation_steps": 8',
        config_source
    )
    
    # 9. Update save_total_limit: 3 → 5
    config_source = re.sub(
        r'"save_total_limit":\s*3',
        '"save_total_limit": 5',
        config_source
    )
    config_source = re.sub(
        r'"save_total_limit":\s*\d+',
        lambda m: '"save_total_limit": 5' if int(re.search(r'\d+', m.group(0)).group(0)) < 5 else m.group(0),
        config_source
    )
    
    # 10. Ensure max_steps is 3000
    config_source = re.sub(
        r'"max_steps":\s*\d+',
        '"max_steps": 3000',
        config_source
    )
    
    # Update the cell
    update_cell_source(nb['cells'][config_cell_idx], config_source)
    
    # Save notebook
    write_notebook(nb_path, nb)
    print(f"  ✓ Updated configuration in {nb_name}")

print(f"\n{'='*70}")
print("Configuration cells update complete!")
print(f"{'='*70}")
