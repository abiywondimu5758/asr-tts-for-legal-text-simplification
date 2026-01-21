#!/usr/bin/env python3
"""
Test script for AddisuSeteye/speecht5_tts_amharic2 model
Downloads the model from Hugging Face and generates speech from Amharic text
"""

import torch
import soundfile as sf
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

try:
    from transformers import (
        SpeechT5Processor, 
        SpeechT5ForTextToSpeech, 
        SpeechT5HifiGan,
        SpeechT5Tokenizer,
        SpeechT5FeatureExtractor
    )
    from datasets import load_dataset
except ImportError:
    print("Error: Required packages not installed.")
    print("Please install with: pip install transformers datasets torch soundfile")
    exit(1)

# Model configuration
MODEL_NAME = "AddisuSeteye/speecht5_tts_amharic2"
BASE_SPEECHT5_NAME = "microsoft/speecht5_tts"  # Base model for feature extractor
VOCODER_NAME = "microsoft/speecht5_hifigan"
SPEAKER_EMBEDDINGS_DATASET = "Matthijs/cmu-arctic-xvectors"
OUTPUT_DIR = Path("speecht5_test_outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

# Test cases with Amharic text (Ge'ez script)
TEST_CASES = [
    {
        "name": "Simple greeting",
        "text": "ሰላም፣ እንዴት ነህ?",
    },
    {
        "name": "Legal domain - contract definition",
        "text": "ውል ማለት በሁለት ወይም ከሁለት በላይ በሆኑ ሰዎች መካከል የሚደረግ ስምምነት ነው።",
    },
    {
        "name": "Legal domain - equality before law",
        "text": "ማንኛውም ሰው በሕግ ፊት እኩል ነው።",
    },
    {
        "name": "Legal domain - constitutional rights",
        "text": "አንቀፅ አምስት የኢትዮጵያ ዜጎች የፖለቲካ መብቶችን ያቀልማል።",
    },
]

def load_speaker_embeddings(speaker_index=7306):
    """
    Load speaker embeddings from the CMU Arctic xvectors dataset.
    You can change the speaker_index to get different voice characteristics.
    """
    try:
        print(f"Loading speaker embeddings from dataset (speaker index: {speaker_index})...")
        embeddings_dataset = load_dataset(
            SPEAKER_EMBEDDINGS_DATASET, 
            split="validation"
        )
        speaker_embeddings = torch.tensor(embeddings_dataset[speaker_index]["xvector"]).unsqueeze(0)
        print(f"✓ Speaker embeddings loaded (shape: {speaker_embeddings.shape})")
        return speaker_embeddings
    except Exception as e:
        print(f"⚠ Warning: Could not load speaker embeddings from dataset: {e}")
        print("Using default zero embeddings instead...")
        # Use default zero embeddings (512-dimensional for SpeechT5)
        return torch.zeros((1, 512))

def download_and_load_model():
    """Download and load the SpeechT5 model, processor, and vocoder"""
    print("=" * 70)
    print("Loading SpeechT5 TTS Model for Amharic")
    print("=" * 70)
    print()
    
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}")
    print()
    
    # Load processor - construct from components if preprocessor_config.json is missing
    print(f"1. Loading processor...")
    try:
        # Try loading processor directly first
        processor = SpeechT5Processor.from_pretrained(MODEL_NAME)
        print(f"✓ Processor loaded directly from {MODEL_NAME}")
    except Exception as e:
        print(f"  Processor not available as a single file, constructing from components...")
        print(f"  This is normal if the model repo doesn't include preprocessor_config.json")
        
        # Load tokenizer - try from fine-tuned model first, fallback to base
        print(f"  Loading tokenizer from {MODEL_NAME}...")
        try:
            tokenizer = SpeechT5Tokenizer.from_pretrained(MODEL_NAME)
            print(f"  ✓ Tokenizer loaded from {MODEL_NAME}")
        except Exception as e2:
            print(f"  ⚠ Tokenizer not found in {MODEL_NAME}, trying base model...")
            try:
                tokenizer = SpeechT5Tokenizer.from_pretrained(BASE_SPEECHT5_NAME)
                print(f"  ✓ Tokenizer loaded from {BASE_SPEECHT5_NAME}")
            except Exception as e3:
                raise Exception(f"Could not load tokenizer from either {MODEL_NAME} or {BASE_SPEECHT5_NAME}: {e2}, {e3}")
        
        # Load feature extractor from base SpeechT5 model
        print(f"  Loading feature extractor from {BASE_SPEECHT5_NAME}...")
        try:
            feature_extractor = SpeechT5FeatureExtractor.from_pretrained(BASE_SPEECHT5_NAME)
            print(f"  ✓ Feature extractor loaded from {BASE_SPEECHT5_NAME}")
        except Exception as e2:
            print(f"  ✗ Error loading feature extractor: {e2}")
            raise
        
        # Construct processor from tokenizer and feature extractor
        processor = SpeechT5Processor(tokenizer=tokenizer, feature_extractor=feature_extractor)
        print(f"✓ Processor constructed successfully from components")
    
    # Load model
    print(f"2. Loading model from {MODEL_NAME}...")
    try:
        model = SpeechT5ForTextToSpeech.from_pretrained(MODEL_NAME)
        model.to(device)
        model.eval()
        print(f"✓ Model loaded successfully and moved to {device}")
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        raise
    
    # Load vocoder
    print(f"3. Loading vocoder from {VOCODER_NAME}...")
    try:
        vocoder = SpeechT5HifiGan.from_pretrained(VOCODER_NAME)
        vocoder.to(device)
        vocoder.eval()
        print(f"✓ Vocoder loaded successfully and moved to {device}")
    except Exception as e:
        print(f"✗ Error loading vocoder: {e}")
        raise
    
    print()
    return processor, model, vocoder, device

def generate_speech(processor, model, vocoder, text, speaker_embeddings, device):
    """Generate speech from text using the SpeechT5 model"""
    # Process text
    inputs = processor(text=text, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    speaker_embeddings = speaker_embeddings.to(device)
    
    # Generate speech
    with torch.no_grad():
        speech = model.generate_speech(
            inputs["input_ids"], 
            speaker_embeddings, 
            vocoder=vocoder
        )
    
    # Move to CPU and convert to numpy
    if isinstance(speech, torch.Tensor):
        speech = speech.cpu().numpy()
    
    return speech

def main():
    print("=" * 70)
    print("Testing AddisuSeteye/speecht5_tts_amharic2 Model")
    print("=" * 70)
    print()
    
    # Check PyTorch
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    print()
    
    # Load model components
    try:
        processor, model, vocoder, device = download_and_load_model()
    except Exception as e:
        print(f"\n✗ Failed to load model: {e}")
        print("\nMake sure you have:")
        print("  1. Internet connection to download from Hugging Face")
        print("  2. Required packages: pip install transformers datasets torch soundfile")
        print("  3. Sufficient disk space for the models")
        return
    
    # Load speaker embeddings
    speaker_embeddings = load_speaker_embeddings()
    print()
    
    # Test with sample texts
    print("=" * 70)
    print("Generating Speech from Amharic Text")
    print("=" * 70)
    print()
    
    sampling_rate = 16000  # SpeechT5 standard sampling rate
    
    for i, test_case in enumerate(TEST_CASES, 1):
        print(f"Test {i}: {test_case['name']}")
        print(f"Text (Ge'ez): {test_case['text']}")
        print(f"Text length: {len(test_case['text'])} characters")
        
        try:
            # Generate speech
            speech = generate_speech(
                processor, 
                model, 
                vocoder, 
                test_case['text'], 
                speaker_embeddings, 
                device
            )
            
            # Save audio file
            output_file = OUTPUT_DIR / f"speecht5_test_{i}.wav"
            sf.write(
                str(output_file), 
                speech, 
                samplerate=sampling_rate
            )
            
            file_size = output_file.stat().st_size / 1024
            duration = len(speech) / sampling_rate
            print(f"✓ Audio generated successfully!")
            print(f"  Output: {output_file}")
            print(f"  File size: {file_size:.1f} KB")
            print(f"  Duration: {duration:.2f} seconds")
            print(f"  Sampling rate: {sampling_rate} Hz")
            
        except Exception as e:
            print(f"✗ Error generating speech: {e}")
            import traceback
            traceback.print_exc()
        
        print("-" * 70)
        print()
    
    print("=" * 70)
    print("Testing Complete!")
    print("=" * 70)
    print(f"\nAll output files saved in: {OUTPUT_DIR.absolute()}")
    print("\nYou can listen to the generated audio files to evaluate the quality.")
    print("\nTo use a different voice, modify the speaker_index in load_speaker_embeddings()")
    print("or load custom speaker embeddings from your own dataset.")

if __name__ == "__main__":
    main()
