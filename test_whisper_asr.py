# test_whisper_asr.py

import torch
import soundfile as sf
import librosa
from transformers import (
    WhisperProcessor, 
    WhisperForConditionalGeneration,
    AutoProcessor,
    AutoModelForCTC,
    Wav2Vec2Processor,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2CTCTokenizer
)

AUDIO_PATH = "000243_gtts.wav"

models = {
    # "Whisper Base (Multilingual)": "openai/whisper-base",
    # "Whisper Fine-tuned Amharic": "seyyaw/whisper-finetuned-amharic",
    "agkphysics_wav2vec2": "agkphysics/wav2vec2-large-xlsr-53-amharic"
}

# Load audio
speech, sr = sf.read(AUDIO_PATH)
print(f"Loaded audio: {AUDIO_PATH}, sample rate: {sr}")

device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")

for display_name, model_id in models.items():
    print("\n" + "=" * 60)
    print(f"Running ASR with: {display_name}")
    print("=" * 60)

    # Determine if it's a Whisper or Wav2Vec2 model
    is_whisper = "whisper" in model_id.lower() or "openai" in model_id.lower() or "seyyaw" in model_id.lower()
    
    # Prepare input - ensure audio is resampled to 16kHz if needed
    audio_for_processing = speech.copy()
    if sr != 16000:
        audio_for_processing = librosa.resample(audio_for_processing, orig_sr=sr, target_sr=16000)
        processing_sr = 16000
    else:
        processing_sr = sr

    if is_whisper:
        # Whisper model processing
        processor = WhisperProcessor.from_pretrained(model_id)
        model = WhisperForConditionalGeneration.from_pretrained(model_id).to(device)
        
        inputs = processor(
            audio_for_processing,
            sampling_rate=processing_sr,
            return_tensors="pt"
        ).input_features.to(device)

        # Generate transcription with better parameters
        predicted_ids = model.generate(
            inputs,
            task="transcribe",
            language="am",
            max_length=448,
            num_beams=5,
            temperature=0.0,
            return_timestamps=False,
            early_stopping=True,
            no_repeat_ngram_size=3,
            repetition_penalty=1.15
        )

        transcription = processor.batch_decode(
            predicted_ids,
            skip_special_tokens=True
        )[0]
    else:
        # Wav2Vec2 model processing
        try:
            processor = AutoProcessor.from_pretrained(model_id)
        except Exception:
            # Fallback to Wav2Vec2Processor if AutoProcessor fails
            feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_id)
            tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(model_id)
            processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)
        
        model = AutoModelForCTC.from_pretrained(model_id).to(device)
        
        # Process audio
        inputs = processor(
            audio_for_processing,
            sampling_rate=processing_sr,
            return_tensors="pt",
            padding=True
        )
        
        # Get logits
        with torch.no_grad():
            logits = model(inputs.input_values.to(device)).logits
        
        # CTC decoding
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = processor.batch_decode(predicted_ids)[0]

    print("Transcription:")
    print(transcription)

print("\nASR comparison finished.")




""