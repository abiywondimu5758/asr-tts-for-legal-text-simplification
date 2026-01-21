# Amharic ASR Streamlit App

A Streamlit application for Automatic Speech Recognition (ASR) using fine-tuned Wav2Vec2 models with LoRA adapters.

## Setup

1. Install dependencies:
```bash
pip install -r requirements_asr_app.txt
```

2. Ensure your models are in the `/models` folder structure:
```
models/
  ├── adapter_folder_1/
  │   ├── adapter_config.json
  │   ├── adapter_model.bin
  │   ├── tokenizer_config.json
  │   ├── vocab.json
  │   └── ... (other processor files)
  └── adapter_folder_2/
      └── ... (same structure)
```

3. Base model setup:
   - Option 1: Place the base model in `models/base_model/`
   - Option 2: Ensure the base model `agkphysics/wav2vec2-large-xlsr-53-amharic` is cached in HuggingFace cache (downloaded once)

## Running the App

```bash
streamlit run streamlit_asr_app.py
```

## Features

- **Model Selection**: Choose from available adapters in the `/models` folder
- **File Upload**: Upload WAV, MP3, M4A, or FLAC files for transcription
- **Voice Recording**: Record audio directly in the browser for real-time transcription
- **Local Model Loading**: All models loaded from local storage (no internet download during inference)

## Usage

1. Select an ASR model from the sidebar dropdown
2. Choose to either:
   - **Upload Audio**: Upload a WAV file and click "Transcribe Audio"
   - **Record Voice**: Click the microphone button to record, then transcribe
3. View the transcribed text in the text area

## Notes

- Audio files are automatically resampled to 16kHz (required by Wav2Vec2)
- The app uses GPU if available, otherwise falls back to CPU
- Models are cached in memory after first load for faster subsequent uses




