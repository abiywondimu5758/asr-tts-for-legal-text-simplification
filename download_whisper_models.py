# download_whisper_models.py

from transformers import (
    WhisperProcessor, 
    WhisperForConditionalGeneration,
    AutoProcessor,
    AutoModelForCTC,
    Wav2Vec2FeatureExtractor
)

# Whisper models
whisper_models = {
    # "baseline": "openai/whisper-base",
    # "amharic_finetuned": "seyyaw/whisper-finetuned-amharic",
}

# Wav2Vec2 models
wav2vec2_models = {
    "speechbrain_wav2vec2": "speechbrain/asr-wav2vec2-dvoice-amharic",
    "agkphysics_wav2vec2": "agkphysics/wav2vec2-large-xlsr-53-amharic"
}

# Download Whisper models
for name, model_id in whisper_models.items():
    print(f"\nDownloading Whisper model {name}: {model_id}")
    try:
        processor = WhisperProcessor.from_pretrained(model_id)
        model = WhisperForConditionalGeneration.from_pretrained(model_id)
        print(f"Finished downloading {name}")
    except Exception as e:
        print(f"Error downloading {name}: {e}")

# Download Wav2Vec2 models
for name, model_id in wav2vec2_models.items():
    print(f"\nDownloading Wav2Vec2 model {name}: {model_id}")
    try:
        # Download feature extractor and tokenizer separately for Wav2Vec2
        try:
            feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_id)
            print(f"  Downloaded feature extractor for {name}")
        except Exception as e:
            print(f"  Warning: Could not download feature extractor: {e}")
        
        # Download the model
        model = AutoModelForCTC.from_pretrained(model_id)
        print(f"Finished downloading {name}")
    except Exception as e:
        print(f"Error downloading {name}: {e}")
        import traceback
        traceback.print_exc()

print("\nAll models downloaded and cached successfully.")
