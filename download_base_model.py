#!/usr/bin/env python3
"""
Download and save the base Wav2Vec2 model to /models folder.
This script downloads agkphysics/wav2vec2-large-xlsr-53-amharic and saves it locally.
"""

import os
from pathlib import Path
from transformers import (
    Wav2Vec2Processor,
    Wav2Vec2ForCTC,
)

# Configuration
MODEL_NAME = "agkphysics/wav2vec2-large-xlsr-53-amharic"
MODELS_DIR = Path("models")
BASE_MODEL_DIR = MODELS_DIR / "base_model"

def download_and_save_model():
    """Download the base model and processor and save to local directory"""
    print("=" * 70)
    print("Downloading Base Model")
    print("=" * 70)
    print(f"Model: {MODEL_NAME}")
    print(f"Save location: {BASE_MODEL_DIR}")
    print()
    
    # Create models directory if it doesn't exist
    MODELS_DIR.mkdir(exist_ok=True)
    BASE_MODEL_DIR.mkdir(exist_ok=True)
    
    # Check if model already exists
    if (BASE_MODEL_DIR / "config.json").exists():
        print(f"⚠️  Model already exists at {BASE_MODEL_DIR}")
        response = input("Do you want to re-download? (y/n): ").strip().lower()
        if response != 'y':
            print("Skipping download.")
            return
    
    try:
        print("📥 Downloading processor...")
        processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME)
        print("✓ Processor downloaded successfully")
        
        print("📥 Downloading model...")
        model = Wav2Vec2ForCTC.from_pretrained(MODEL_NAME)
        print("✓ Model downloaded successfully")
        
        print(f"\n💾 Saving processor to {BASE_MODEL_DIR}...")
        processor.save_pretrained(BASE_MODEL_DIR)
        print("✓ Processor saved successfully")
        
        print(f"💾 Saving model to {BASE_MODEL_DIR}...")
        model.save_pretrained(BASE_MODEL_DIR)
        print("✓ Model saved successfully")
        
        print("\n" + "=" * 70)
        print("✅ Download and save completed successfully!")
        print("=" * 70)
        print(f"\nModel files saved to: {BASE_MODEL_DIR.absolute()}")
        print("\nFiles saved:")
        for file in sorted(BASE_MODEL_DIR.iterdir()):
            size = file.stat().st_size / (1024 * 1024)  # Size in MB
            print(f"  - {file.name} ({size:.2f} MB)")
        
    except Exception as e:
        print(f"\n❌ Error downloading or saving model: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = download_and_save_model()
    if success:
        print("\n✨ The base model is now ready to use in your Streamlit app!")
    else:
        print("\n⚠️  Failed to download the model. Please check your internet connection and try again.")




