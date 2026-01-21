import streamlit as st
import torch
import librosa
import numpy as np
from pathlib import Path
import os
import tempfile
import warnings
import pandas as pd
import pickle
import hashlib
import json
from datetime import datetime
from typing import Optional, List
warnings.filterwarnings('ignore')

from transformers import (
    Wav2Vec2Processor,
    Wav2Vec2ForCTC,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2CTCTokenizer,
    AutoProcessor,
    AutoModelForCTC,
    AutoModelForCausalLM,
    AutoTokenizer,
)
from peft import PeftModel
import asyncio
import edge_tts
from gtts import gTTS
import io
import jiwer
import faiss
from sentence_transformers import SentenceTransformer
import google.generativeai as genai

# Configuration
MODELS_DIR = Path("models")
BASE_MODEL_NAME = "agkphysics/wav2vec2-large-xlsr-53-amharic"
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

# Model 5 (LLaMA 400M) configuration for legal simplification
# All models are in Foundation Models Course Projects/models
FOUNDATION_MODELS_DIR = Path("Foundation Models Course Projects")
FOUNDATION_MODELS_MODELS_DIR = FOUNDATION_MODELS_DIR / "models"

LLAMA_BASE_MODEL_NAME = "rasyosef/Llama-3.2-400M-Amharic-Instruct"
LLAMA_BASE_MODEL_PATH = FOUNDATION_MODELS_MODELS_DIR / "models--rasyosef--Llama-3.2-400M-Amharic-Instruct"
MODEL_5_ADAPTER_PATH = FOUNDATION_MODELS_MODELS_DIR / "llama-400m-legal-simplification5"

# RAG paths
RAG_INDEX_PATH = FOUNDATION_MODELS_DIR / "rag_pipeline" / "4_vector_db" / "faiss_index.bin"
RAG_METADATA_PATH = FOUNDATION_MODELS_DIR / "rag_pipeline" / "4_vector_db" / "metadata.parquet"

# Contrastive selector path
CONTRASTIVE_MODEL_PATH = FOUNDATION_MODELS_MODELS_DIR / "contrastive_strategy_selector"

# Test set configuration
TEST_CSV_PATH = Path("Dataset_1.5h") / "test.csv"
TEST_AUDIO_DIR = Path("Dataset_1.5h") / "audio"
EVALUATION_CACHE_DIR = Path("evaluation_cache")
EVALUATION_CACHE_DIR.mkdir(exist_ok=True)

# Cache loaded models in session state
def find_base_model():
    """Try to find base model in models directory"""
    # Check common locations for base model
    possible_paths = [
        MODELS_DIR / "base_model",
        MODELS_DIR / BASE_MODEL_NAME.replace("/", "_"),
        Path(BASE_MODEL_NAME.replace("/", "_")),
    ]
    
    for path in possible_paths:
        if path.exists() and (path / "config.json").exists():
            return str(path)
    return None

@st.cache_resource
def load_base_model_only(base_model_path=None):
    """Load base model only (without LoRA adapters) - like test_whisper_asr.py"""
    # Try to find base model locally
    if not base_model_path:
        base_model_path = find_base_model()
    
    # Try loading processor from base model location or use AutoProcessor
    try:
        if base_model_path and os.path.exists(base_model_path):
            processor = Wav2Vec2Processor.from_pretrained(
                base_model_path,
                local_files_only=True
            )
            model = Wav2Vec2ForCTC.from_pretrained(
                base_model_path,
                local_files_only=True
            )
            model_source = f"Local: {base_model_path}"
        else:
            # Load from HuggingFace using AutoProcessor (like test_whisper_asr.py)
            try:
                processor = AutoProcessor.from_pretrained(BASE_MODEL_NAME)
            except Exception:
                # Fallback to Wav2Vec2Processor components
                feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(BASE_MODEL_NAME)
                tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(BASE_MODEL_NAME)
                processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)
            
            model = AutoModelForCTC.from_pretrained(BASE_MODEL_NAME)
            model_source = f"HuggingFace: {BASE_MODEL_NAME}"
    except Exception as e:
        raise FileNotFoundError(
            f"Base model not found. Please ensure '{BASE_MODEL_NAME}' "
            f"is available in HuggingFace cache or placed in '{MODELS_DIR}/base_model'. Error: {e}"
        )
    
    # Move to device and set to eval mode
    model = model.to(DEVICE)
    model.eval()
    
    return model, processor, model_source

def load_model(adapter_path, processor_path, base_model_path=None):
    """Load base model and LoRA adapters from local paths"""
    # Load processor from adapter/processor path
    if not os.path.exists(processor_path):
        raise FileNotFoundError(f"Processor not found at {processor_path}")
    
    processor = Wav2Vec2Processor.from_pretrained(
        processor_path, 
        local_files_only=True
    )
    
    # Try to find base model locally
    if not base_model_path:
        base_model_path = find_base_model()
    
    # Load base model
    if base_model_path and os.path.exists(base_model_path):
        # Load from local path
        base_model = Wav2Vec2ForCTC.from_pretrained(
            base_model_path,
            local_files_only=True,
            ctc_loss_reduction="mean",
            pad_token_id=processor.tokenizer.pad_token_id,
            vocab_size=len(processor.tokenizer),
            ignore_mismatched_sizes=True
        )
        model_source = f"Local: {base_model_path}"
    else:
        # Try to load from HuggingFace cache
        try:
            base_model = Wav2Vec2ForCTC.from_pretrained(
                BASE_MODEL_NAME,
                ctc_loss_reduction="mean",
                pad_token_id=processor.tokenizer.pad_token_id,
                vocab_size=len(processor.tokenizer),
                local_files_only=True  # Only use cache, don't download
            )
            model_source = f"HuggingFace cache: {BASE_MODEL_NAME}"
        except Exception as e:
            raise FileNotFoundError(
                f"Base model not found locally. Please ensure '{BASE_MODEL_NAME}' "
                f"is cached in HuggingFace cache or placed in '{MODELS_DIR}/base_model'"
            )
    
    # Load LoRA adapters
    if not os.path.exists(adapter_path):
        raise FileNotFoundError(f"Adapter not found at {adapter_path}")
    
    peft_model = PeftModel.from_pretrained(base_model, adapter_path, local_files_only=True)
    
    # Merge LoRA weights into base model for inference to avoid PEFT wrapper issues
    # This is more reliable than wrapping forward methods
    model = peft_model.merge_and_unload()
    
    # Alternative: Keep as PEFT model and wrap forward method if merge doesn't work
    # But merging is cleaner and matches how models are typically used for inference
    
    # Move to device and set to eval mode
    model = model.to(DEVICE)
    model.eval()
    
    return model, processor, model_source

def find_available_adapters():
    """Find available adapter folders in the models directory and include base model option"""
    adapters = []
    
    # Add base model as the first option
    adapters.append({
        "name": f"Base Model ({BASE_MODEL_NAME})",
        "adapter_path": None,  # None indicates base model
        "processor_path": None,
        "is_base_model": True
    })
    
    if not MODELS_DIR.exists():
        return adapters
    
    # Look for folders containing adapter_config.json
    for folder in MODELS_DIR.iterdir():
        if folder.is_dir():
            adapter_config = folder / "adapter_config.json"
            if adapter_config.exists():
                adapters.append({
                    "name": folder.name,
                    "adapter_path": str(folder),
                    "processor_path": str(folder),  # Processor is saved in the same folder
                    "is_base_model": False
                })
    
    return adapters

def transcribe_audio(model, processor, audio_array, sampling_rate=16000):
    """Transcribe audio array using the model"""
    # Resample to 16kHz if needed (wav2vec2 expects 16kHz)
    if sampling_rate != 16000:
        audio_array = librosa.resample(audio_array, orig_sr=sampling_rate, target_sr=16000)
        sampling_rate = 16000
    
    # Process audio
    inputs = processor(
        audio_array,
        sampling_rate=sampling_rate,
        return_tensors="pt",
        padding=True
    )
    
    # Move inputs to device
    input_values = inputs.input_values.to(DEVICE)
    
    # Get predictions
    # Use positional argument like test_whisper_asr.py does for consistency
    with torch.no_grad():
        logits = model(input_values).logits
    
    # Decode predictions
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)[0]
    
    return transcription

# Text Normalization Functions
def normalize_amharic_text(text: str) -> str:
    """Normalize Amharic text by fixing spacing and common spelling errors
    
    This function handles:
    1. Word spacing corrections (removing unnecessary spaces within words)
    2. Common spelling corrections (legal terms, common errors)
    3. Basic punctuation improvements
    """
    if not text:
        return text
    
    import re
    normalized = text
    
    # Common word spacing patterns to fix (words that should not have internal spaces)
    # These are common patterns from ASR errors - order matters for compound patterns
    spacing_patterns = [
        # Compound words that should not have spaces
        (r'እንደሚያ\s+መለክተው', 'እንደሚያመለክተው'),
        (r'ማንኛው\s+ም\b', 'ማንኛውም'),
        (r'እንዳይ\s+ከናወን', 'እንዳይከናወን'),
        (r'በ\s+መተላለፍ', 'በመተላለፍ'),
        (r'የ\s+ሚቀጥል', 'የሚቀጥል'),
        (r'ይጠየ\s+ቃል', 'ይጠየቃል'),
        
        # Common preposition/adverb patterns (fix spaces after single character prepositions)
        (r'\bበ\s+([ሀ-፰])', r'በ\1'),  # "በ " followed by Amharic char
        (r'\bየ\s+([ሀ-፰])', r'የ\1'),  # "የ " followed by Amharic char
        (r'\bከ\s+([ሀ-፰])', r'ከ\1'),  # "ከ " followed by Amharic char
        (r'\bላይ\s+([ሀ-፰])', r'ላይ\1'),  # "ላይ " followed by Amharic char
        
        # Common verb/particle patterns
        (r'\s+ሲሆን\b', ' ሲሆን'),  # Ensure space before "ሲሆን"
        (r'\s+ወንጀል', ' ወንጀል'),
    ]
    
    # Apply spacing corrections
    for pattern, replacement in spacing_patterns:
        normalized = re.sub(pattern, replacement, normalized)
    
    # Common spelling corrections (legal terms and common errors)
    spelling_corrections = {
        "መና ወንጀል": "መናቅ ወንጀል",  # Must come before single word replacement
        "መና": "መናቅ",  # Contempt of Court (common ASR error)
    }
    
    # Apply spelling corrections
    for wrong, correct in spelling_corrections.items():
        normalized = normalized.replace(wrong, correct)
    
    # Remove extra spaces (multiple spaces to single space)
    normalized = re.sub(r' +', ' ', normalized)
    
    # Basic punctuation improvements
    # Add comma before "ሲሆን" if there's a word before it (common pattern)
    normalized = re.sub(r'(\w+) ሲሆን', r'\1፣ ሲሆን', normalized)
    
    # Ensure proper sentence ending punctuation
    normalized = normalized.strip()
    if normalized and not normalized.endswith(('።', '፤', '፥', '፦', '.', '!', '?')):
        normalized += '።'
    
    return normalized

# Model 5 (Legal Simplification) Functions
@st.cache_resource
def load_llama_base_model():
    """Load LLaMA base model for Model 5 from Foundation Models Course Projects/models"""
    snapshot_dir = LLAMA_BASE_MODEL_PATH / "snapshots"
    base_model_name = LLAMA_BASE_MODEL_NAME
    
    # Find the snapshot directory
    local_model_path = None
    if snapshot_dir.exists():
        snapshots = list(snapshot_dir.iterdir())
        if snapshots:
            for snapshot in snapshots:
                potential_path = snapshot
                model_file_bin = potential_path / "pytorch_model.bin"
                model_file_safe = potential_path / "model.safetensors"
                if model_file_bin.exists() or model_file_safe.exists():
                    local_model_path = potential_path
                    break
    
    if local_model_path and local_model_path.exists():
        # Load from local snapshot path
        try:
            tokenizer = AutoTokenizer.from_pretrained(str(local_model_path), local_files_only=True)
            model = AutoModelForCausalLM.from_pretrained(
                str(local_model_path),
                local_files_only=True,
                torch_dtype=torch.float16 if torch.cuda.is_available() else None,
                device_map="auto" if torch.cuda.is_available() else None,
                trust_remote_code=True
            )
        except Exception as e:
            # If local loading fails, try with cache_dir pointing to Foundation Models models directory
            tokenizer = AutoTokenizer.from_pretrained(base_model_name, cache_dir=str(FOUNDATION_MODELS_MODELS_DIR))
            model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                cache_dir=str(FOUNDATION_MODELS_MODELS_DIR),
                torch_dtype=torch.float16 if torch.cuda.is_available() else None,
                device_map="auto" if torch.cuda.is_available() else None,
                trust_remote_code=True
            )
    else:
        # Use cache_dir - will use existing cache if available in Foundation Models models directory
        tokenizer = AutoTokenizer.from_pretrained(base_model_name, cache_dir=str(FOUNDATION_MODELS_MODELS_DIR))
        model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            cache_dir=str(FOUNDATION_MODELS_MODELS_DIR),
            torch_dtype=torch.float16 if torch.cuda.is_available() else None,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True
        )
    
    # Set pad_token for LLaMA if needed
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        if model.config.pad_token_id is None:
            model.config.pad_token_id = tokenizer.pad_token_id
    
    if not torch.cuda.is_available():
        model = model.to(DEVICE)
    
    return model, tokenizer

@st.cache_resource
def load_model5_adapter(_base_model, adapter_path):
    """Load Model 5 adapter on base model"""
    adapter = PeftModel.from_pretrained(_base_model, str(adapter_path))
    adapter = adapter.to(DEVICE)
    adapter.eval()
    return adapter

@st.cache_resource
def load_rag_system():
    """Load RAG index and metadata"""
    index_exists = RAG_INDEX_PATH.exists()
    metadata_exists = RAG_METADATA_PATH.exists()
    
    if not index_exists or not metadata_exists:
        return None, None
    
    try:
        index = faiss.read_index(str(RAG_INDEX_PATH))
        metadata = pd.read_parquet(RAG_METADATA_PATH)
        return index, metadata
    except Exception as e:
        return None, None

@st.cache_resource
def load_contrastive_selector():
    """Load contrastive strategy selector"""
    if not CONTRASTIVE_MODEL_PATH.exists():
        return None
    
    try:
        encoder = SentenceTransformer(str(CONTRASTIVE_MODEL_PATH))
        encoder = encoder.to(DEVICE)
        
        centroids_path = CONTRASTIVE_MODEL_PATH / "centroids.pkl"
        if centroids_path.exists():
            with open(centroids_path, "rb") as f:
                centroids = pickle.load(f)
            return encoder, centroids
        return None
    except Exception as e:
        return None

def get_rag_context(query: str, index, metadata, top_k: int = 3) -> List[str]:
    """Retrieve relevant legal context using RAG"""
    try:
        # Load Gemini API key
        api_key_path = FOUNDATION_MODELS_DIR / ".gemini_api_key"
        if not api_key_path.exists():
            # Try in current directory
            api_key_path = Path(".gemini_api_key")
        
        if api_key_path.exists():
            with open(api_key_path, "r") as f:
                content = f.read().strip()
                if "=" in content:
                    api_key = content.split("=", 1)[1].strip()
                else:
                    api_key = content
                genai.configure(api_key=api_key)
        else:
            return []
        
        # Get embedding for query
        result = genai.embed_content(
            model="models/text-embedding-004",
            content=query
        )
        query_embedding = np.array(result['embedding'], dtype=np.float32).reshape(1, -1)
        
        # Search in FAISS index
        k = min(top_k, index.ntotal)
        distances, indices = index.search(query_embedding, k)
        
        # Get relevant chunks
        contexts = []
        for idx in indices[0]:
            if idx < len(metadata):
                chunk_text = metadata.iloc[idx]['text']
                contexts.append(chunk_text)
        
        return contexts
    except Exception as e:
        return []

def predict_simplification_type(sentence: str, encoder, centroids) -> str:
    """Predict simplification type using contrastive selector"""
    # Heuristic rules first
    words = sentence.split()
    word_count = len(words)
    
    if word_count > 40:
        return "sentence_splitting"
    
    boilerplate_phrases = [
        "እንደተጠበቀ ሆኖ",
        "በማንኛውም ሁኔታ",
        "ያለ አግባብ",
        "እንደተጠበቀ",
    ]
    if any(phrase in sentence for phrase in boilerplate_phrases):
        return "deletion"
    
    conjunctions = ["እና", "ወይም", "ቢሆንም", "ነገር ግን"]
    conjunction_count = sum(1 for conj in conjunctions if conj in sentence)
    if conjunction_count >= 3:
        return "structure_reordering"
    
    # Fallback to contrastive
    try:
        emb = encoder.encode([sentence], convert_to_numpy=True)[0]
        
        sims = {}
        for label, centroid in centroids.items():
            dot_product = np.dot(emb, centroid)
            norm_emb = np.linalg.norm(emb)
            norm_centroid = np.linalg.norm(centroid)
            similarity = dot_product / (norm_emb * norm_centroid) if (norm_emb * norm_centroid) > 0 else 0
            sims[label] = similarity
        
        return max(sims, key=sims.get)
    except Exception as e:
        return "vocabulary_simplification"  # default

def simplify_text(
    text: str,
    model,
    tokenizer,
    simplification_type: Optional[str] = None,
    rag_context: Optional[List[str]] = None,
    max_input_length: int = 512
) -> str:
    """Simplify legal text using Model 5"""
    # Build prompt
    base_instruction = "የሕግ ቃላትን ለግለሰቦች ለመረዳት ቀላል አማርኛ ውስጥ አቅርብ: "
    
    type_map = {
        "vocabulary_simplification": "[የቃላት ማቃለል]",
        "sentence_splitting": "[የዓረፍተ ነገር መከፋፈል]",
        "deletion": "[መሻር]",
        "structure_reordering": "[የዕውቀት ማደራጀት]",
        "definition_or_expansion": "[ፍቺ ወይም ማስፋፋት]"
    }
    if simplification_type:
        type_label = type_map.get(simplification_type, "[አጠቃላይ]")
    else:
        type_label = "[አጠቃላይ]"  # Default without contrastive
    prompt = base_instruction + type_label + " " + text
    
    # Add RAG context if provided
    if rag_context:
        context = "\n".join(rag_context[:2])  # Use top 2 contexts
        prompt = f"የሕግ አውድ: {context}\n\n{prompt}"
    
    # Tokenize
    inputs = tokenizer(
        prompt,
        max_length=max_input_length,
        truncation=True,
        padding=True,
        return_tensors="pt"
    ).to(DEVICE)
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=128,
            num_beams=4,
            early_stopping=True,
            repetition_penalty=1.5,
            no_repeat_ngram_size=4,
            length_penalty=0.6,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            do_sample=False
        )
        # Decode and extract only the generated part (remove input)
        full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Remove the input portion
        if full_output.startswith(prompt):
            simplified = full_output[len(prompt):].strip()
        else:
            simplified = full_output
        
        # Post-process: Extract only the first sentence
        sentence_endings = ['።', '፤', '፥', '፦', '.', '!', '?']
        for ending in sentence_endings:
            if ending in simplified:
                idx = simplified.find(ending)
                simplified = simplified[:idx+1].strip()
                break
    
    return simplified

# TTS Functions
async def generate_edge_tts_audio_async(text: str, voice: str) -> bytes:
    """Generate audio using edge_tts asynchronously"""
    communicate = edge_tts.Communicate(text, voice)
    audio_data = b""
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            audio_data += chunk["data"]
    return audio_data

def generate_edge_tts_audio(text: str, voice: str) -> bytes:
    """Generate audio using edge_tts (wrapper for async)"""
    try:
        return asyncio.run(generate_edge_tts_audio_async(text, voice))
    except Exception as e:
        raise Exception(f"edge_tts error: {str(e)}")

def generate_gtts_audio(text: str) -> bytes:
    """Generate audio using gTTS"""
    try:
        tts = gTTS(text=text, lang='am', slow=False)
        audio_buffer = io.BytesIO()
        tts.write_to_fp(audio_buffer)
        audio_buffer.seek(0)
        return audio_buffer.read()
    except Exception as e:
        raise Exception(f"gTTS error: {str(e)}")

def generate_tts_audio(text: str, tts_engine: str, voice: str = None) -> bytes:
    """Generate audio using the selected TTS engine"""
    if tts_engine == "edge_tts":
        if voice == "Ameha":
            voice_id = "am-ET-AmehaNeural"
        elif voice == "Mekdes":
            voice_id = "am-ET-MekdesNeural"
        else:
            voice_id = "am-ET-MekdesNeural"  # Default
        
        return generate_edge_tts_audio(text, voice_id)
    
    elif tts_engine == "gTTS":
        return generate_gtts_audio(text)
    
    else:
        raise ValueError(f"Unknown TTS engine: {tts_engine}")

# Evaluation Functions
def get_evaluation_cache_path(adapter_name: str) -> Path:
    """Get path to cached evaluation results for an adapter"""
    # Create a hash of the adapter name for filename
    cache_filename = hashlib.md5(adapter_name.encode()).hexdigest() + ".json"
    return EVALUATION_CACHE_DIR / cache_filename

def load_cached_evaluation(adapter_name: str) -> dict:
    """Load cached evaluation results if available"""
    cache_path = get_evaluation_cache_path(adapter_name)
    if cache_path.exists():
        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            return None
    return None

def save_evaluation_cache(adapter_name: str, results: dict):
    """Save evaluation results to cache"""
    cache_path = get_evaluation_cache_path(adapter_name)
    try:
        with open(cache_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
    except Exception as e:
        st.warning(f"Failed to save evaluation cache: {e}")

def load_test_set():
    """Load test set from CSV"""
    if not TEST_CSV_PATH.exists():
        raise FileNotFoundError(f"Test CSV not found at {TEST_CSV_PATH}")
    
    df = pd.read_csv(TEST_CSV_PATH)
    test_samples = []
    
    for _, row in df.iterrows():
        audio_path = TEST_AUDIO_DIR / row['file_name']
        if audio_path.exists():
            test_samples.append({
                'file_name': row['file_name'],
                'audio_path': str(audio_path),
                'transcription': str(row['transcription']).strip()
            })
        else:
            st.warning(f"Audio file not found: {audio_path}")
    
    return test_samples

def evaluate_model(model, processor, test_samples: list, adapter_name: str, progress_container=None, force_rerun: bool = False) -> dict:
    """Evaluate model on test set and return metrics
    
    Args:
        model: Loaded ASR model
        processor: Loaded processor
        test_samples: List of test samples with audio_path and transcription
        adapter_name: Name of the adapter (for caching)
        progress_container: Optional Streamlit container for progress updates
        force_rerun: If True, ignore cache and re-run evaluation
    """
    # Check cache first (unless force_rerun)
    if not force_rerun:
        cached_results = load_cached_evaluation(adapter_name)
        if cached_results is not None:
            return cached_results
    
    # Run evaluation
    predictions = []
    references = []
    file_names = []
    errors = []
    
    total_samples = len(test_samples)
    
    # Progress tracking
    progress_bar = None
    status_text = None
    
    if progress_container:
        with progress_container.container():
            progress_bar = st.progress(0)
            status_text = st.empty()
    
    for idx, sample in enumerate(test_samples):
        try:
            # Update progress
            if progress_bar is not None:
                progress = (idx + 1) / total_samples
                progress_bar.progress(progress)
            if status_text is not None:
                status_text.text(f"Processing {idx + 1}/{total_samples}: {sample['file_name']}")
            
            # Load and transcribe audio
            if not os.path.exists(sample['audio_path']):
                error_msg = f"File not found: {sample['audio_path']}"
                errors.append((sample['file_name'], error_msg))
                continue
                
            audio, sr = librosa.load(sample['audio_path'], sr=16000)
            
            if len(audio) == 0:
                error_msg = f"Empty audio file: {sample['file_name']}"
                errors.append((sample['file_name'], error_msg))
                continue
            
            # Transcribe
            transcription = transcribe_audio(model, processor, audio, sr)
            
            if not transcription or len(str(transcription).strip()) == 0:
                error_msg = f"Empty transcription for: {sample['file_name']}"
                errors.append((sample['file_name'], error_msg))
                continue
            
            predictions.append(transcription)
            references.append(sample['transcription'])
            file_names.append(sample['file_name'])
                
        except Exception as e:
            error_msg = f"Error processing {sample['file_name']}: {str(e)}"
            errors.append((sample['file_name'], error_msg))
            import traceback
            # Show first error in detail, others as warnings
            if len(errors) == 1:
                st.error(f"❌ First error (showing detail): {error_msg}")
                with st.expander("See full traceback"):
                    st.code(traceback.format_exc())
            continue
    
    # Clear progress display
    if progress_bar is not None:
        progress_bar.empty()
    if status_text is not None:
        status_text.empty()
    
    # Show summary of errors if any
    if errors:
        st.warning(f"⚠️ {len(errors)} out of {total_samples} samples failed to process")
        if len(errors) <= 10:
            for file_name, error_msg in errors:
                st.caption(f"  • **{file_name}**: {error_msg}")
        else:
            for file_name, error_msg in errors[:5]:
                st.caption(f"  • **{file_name}**: {error_msg}")
            st.caption(f"  ... and {len(errors) - 5} more errors (see expander)")
            with st.expander("View all errors"):
                for file_name, error_msg in errors:
                    st.caption(f"  • **{file_name}**: {error_msg}")
    
    # Compute metrics
    if len(predictions) > 0 and len(references) > 0:
        # Standard metrics
        wer = jiwer.wer(references, predictions)
        cer = jiwer.cer(references, predictions)
        
        # Word and Character Accuracy (inverted metrics)
        word_accuracy = 1.0 - float(wer)
        char_accuracy = 1.0 - float(cer)
        
        # Sentence Error Rate (SER) - percentage of sentences with at least one error
        perfect_sentences = sum(1 for ref, pred in zip(references, predictions) if ref == pred)
        ser = 1.0 - (perfect_sentences / len(references)) if len(references) > 0 else 0.0
        
        # Length Statistics
        ref_lengths = [len(ref) for ref in references]
        pred_lengths = [len(pred) for pred in predictions]
        
        ref_word_counts = [len(ref.split()) for ref in references]
        pred_word_counts = [len(pred.split()) for pred in predictions]
        
        length_stats = {
            'char_length': {
                'ref_mean': float(np.mean(ref_lengths)),
                'ref_median': float(np.median(ref_lengths)),
                'ref_std': float(np.std(ref_lengths)),
                'pred_mean': float(np.mean(pred_lengths)),
                'pred_median': float(np.median(pred_lengths)),
                'pred_std': float(np.std(pred_lengths)),
                'ratio_mean': float(np.mean([p/r if r > 0 else 0 for p, r in zip(pred_lengths, ref_lengths)])),
                'ratio_median': float(np.median([p/r if r > 0 else 0 for p, r in zip(pred_lengths, ref_lengths)]))
            },
            'word_length': {
                'ref_mean': float(np.mean(ref_word_counts)),
                'ref_median': float(np.median(ref_word_counts)),
                'ref_std': float(np.std(ref_word_counts)),
                'pred_mean': float(np.mean(pred_word_counts)),
                'pred_median': float(np.median(pred_word_counts)),
                'pred_std': float(np.std(pred_word_counts)),
                'ratio_mean': float(np.mean([p/r if r > 0 else 0 for p, r in zip(pred_word_counts, ref_word_counts)])),
                'ratio_median': float(np.median([p/r if r > 0 else 0 for p, r in zip(pred_word_counts, ref_word_counts)]))
            }
        }
        
        results = {
            'wer': float(wer),
            'cer': float(cer),
            'word_accuracy': word_accuracy,
            'char_accuracy': char_accuracy,
            'ser': float(ser),
            'perfect_sentences': perfect_sentences,
            'length_stats': length_stats,
            'num_samples': len(predictions),
            'total_samples': total_samples,
            'timestamp': datetime.now().isoformat(),
            'adapter_name': adapter_name,
            'predictions': predictions,
            'references': references,
            'file_names': file_names
        }
        
        # Save to cache
        save_evaluation_cache(adapter_name, results)
        
        return results
    else:
        raise ValueError("No samples were successfully processed")

def main():
    st.set_page_config(
        page_title="Amharic ASR - Speech Recognition",
        page_icon="🎤",
        layout="wide"
    )
    
    st.title("🎤 Amharic Automatic Speech Recognition")
    st.markdown("Upload a WAV file or record your voice to transcribe Amharic speech")
    
    # Find available adapters
    adapters = find_available_adapters()
    
    # Model selection
    st.sidebar.header("Model Selection")
    adapter_names = [ad["name"] for ad in adapters]
    
    if len(adapters) == 1:  # Only base model
        st.sidebar.warning("No trained adapters found. Only base model is available.")
    
    selected_adapter_name = st.sidebar.selectbox(
        "Choose ASR Model",
        adapter_names,
        help="Select base model or trained adapter to use for transcription"
    )
    
    selected_adapter = next(ad for ad in adapters if ad["name"] == selected_adapter_name)
    is_base_model = selected_adapter.get("is_base_model", False)
    
    # TTS Configuration
    st.sidebar.markdown("---")
    st.sidebar.header("Legal Simplification (Model 5)")
    enable_simplification = st.sidebar.checkbox("Enable Simplification", value=False, help="Simplify transcribed legal text using Model 5")
    
    use_rag = False
    use_contrastive = False
    if enable_simplification:
        use_rag_option = st.sidebar.selectbox(
            "RAG Intervention",
            ["Without RAG", "With RAG"],
            index=0,
            help="Use RAG for legal context retrieval"
        )
        use_rag = use_rag_option == "With RAG"
        
        contrastive_option = st.sidebar.selectbox(
            "Contrastive Learning",
            ["Without Contrastive", "With Contrastive"],
            index=0,
            help="Use contrastive learning for simplification type prediction"
        )
        use_contrastive = contrastive_option == "With Contrastive"
    
    st.sidebar.markdown("---")
    st.sidebar.header("Text-to-Speech")
    enable_tts = st.sidebar.checkbox("Enable TTS", value=True, help="Generate audio from transcribed text")
    
    if enable_tts:
        tts_engine = st.sidebar.selectbox(
            "TTS Engine",
            ["edge_tts", "gTTS"],
            index=0,
            help="Choose TTS engine for audio generation"
        )
        
        # Voice selection (only for edge_tts)
        selected_voice = None
        if tts_engine == "edge_tts":
            selected_voice = st.sidebar.selectbox(
                "Voice",
                ["Mekdes", "Ameha"],
                index=0,
                help="Choose voice for edge_tts (Mekdes or Ameha)"
            )
    
    # Load model
    with st.spinner(f"Loading model: {selected_adapter_name}..."):
        try:
            if is_base_model:
                # Load base model only (no adapters)
                model, processor, model_source = load_base_model_only()
            else:
                # Load fine-tuned model with LoRA adapters
                model, processor, model_source = load_model(
                    selected_adapter["adapter_path"],
                    selected_adapter["processor_path"]
                )
        except Exception as e:
            st.error(f"Failed to load model: {str(e)}")
            if is_base_model:
                st.info(f"Make sure the base model '{BASE_MODEL_NAME}' is available in HuggingFace cache or at '{MODELS_DIR}/base_model'")
            else:
                st.info(f"Make sure the models are available locally in the '{MODELS_DIR}' folder")
            st.stop()
    
    st.sidebar.success(f"✓ Model loaded: {selected_adapter_name}")
    st.sidebar.info(f"Base model: {model_source}\nDevice: {DEVICE}")
    
    # Main content area
    tab1, tab2, tab3 = st.tabs(["📁 Upload Audio", "🎙️ Record Voice", "📊 Model Evaluation"])
    
    with tab1:
        st.header("Upload WAV File")
        uploaded_file = st.file_uploader(
            "Choose a WAV audio file",
            type=['wav', 'mp3', 'm4a', 'flac'],
            help="Upload an audio file to transcribe"
        )
        
        if uploaded_file is not None:
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_path = tmp_file.name
            
            try:
                # Load audio
                audio, sr = librosa.load(tmp_path, sr=None)
                
                # Display audio player
                st.audio(tmp_path, format='audio/wav')
                st.info(f"Audio loaded: {len(audio)/sr:.2f} seconds, Sample rate: {sr} Hz")
                
                # Transcribe button
                if st.button("Transcribe Audio", type="primary"):
                    with st.spinner("Transcribing audio..."):
                        transcription = transcribe_audio(model, processor, audio, sr)
                    
                    st.success("Transcription complete!")
                    
                    # Store transcription in session state
                    st.session_state['transcription'] = transcription
                    st.session_state['transcription_source'] = 'upload'
                
                    # Display transcription if it exists (from session state or current transcription)
                transcription = st.session_state.get('transcription', '')
                if transcription:
                    st.subheader("📝 Transcribed Text")
                    st.text_area(
                        "Transcribed Text",
                        transcription,
                        height=100,
                        key="uploaded_transcription",
                        help="The transcribed text from the audio"
                    )
                    
                    # Legal Simplification
                    simplified_text = None
                    text_for_tts = transcription
                    if enable_simplification:
                        st.markdown("---")
                        st.subheader("⚖️ Legal Simplification")
                        
                        if st.button("Simplify Text", type="primary", key="simplify_upload"):
                            with st.spinner("Simplifying text with Model 5..."):
                                try:
                                    # Normalize the transcription first
                                    normalized_transcription = normalize_amharic_text(transcription)
                                    if normalized_transcription != transcription:
                                        st.info("📝 Text normalized (spacing and spelling corrections applied)")
                                    
                                    # Load Model 5
                                    llama_base_model, llama_tokenizer = load_llama_base_model()
                                    model5 = load_model5_adapter(llama_base_model, MODEL_5_ADAPTER_PATH)
                                    
                                    # Get simplification type if using contrastive (use normalized text)
                                    simplification_type = None
                                    if use_contrastive:
                                        contrastive_result = load_contrastive_selector()
                                        if contrastive_result:
                                            encoder, centroids = contrastive_result
                                            simplification_type = predict_simplification_type(normalized_transcription, encoder, centroids)
                                            st.info(f"Predicted simplification type: **{simplification_type}**")
                                    
                                    # Get RAG context if enabled (use normalized text)
                                    rag_context = None
                                    if use_rag:
                                        rag_index, rag_metadata = load_rag_system()
                                        if rag_index is not None and rag_metadata is not None:
                                            rag_context = get_rag_context(normalized_transcription, rag_index, rag_metadata)
                                            if rag_context:
                                                st.info(f"Retrieved {len(rag_context)} relevant legal contexts")
                                        else:
                                            st.warning("RAG system not available, running without RAG")
                                    
                                    # Simplify (use normalized text)
                                    simplified_text = simplify_text(
                                        normalized_transcription,
                                        model5,
                                        llama_tokenizer,
                                        simplification_type=simplification_type,
                                        rag_context=rag_context
                                    )
                                    
                                    st.session_state['simplified_text'] = simplified_text
                                    text_for_tts = simplified_text
                                    st.success("Simplification complete!")
                                    
                                    # Display simplified text
                                    st.text_area(
                                        "Simplified Text",
                                        simplified_text,
                                        height=100,
                                        key="simplified_upload",
                                        help="The simplified legal text"
                                    )
                                    
                                except Exception as e:
                                    st.error(f"Error during simplification: {str(e)}")
                                    import traceback
                                    with st.expander("See error details"):
                                        st.code(traceback.format_exc())
                        else:
                            # Check if simplified text exists in session state
                            if 'simplified_text' in st.session_state:
                                simplified_text = st.session_state['simplified_text']
                                text_for_tts = simplified_text
                                st.text_area(
                                    "Simplified Text",
                                    simplified_text,
                                    height=100,
                                    key="simplified_upload",
                                    help="The simplified legal text"
                                )
                    
                    # TTS Generation
                    if enable_tts:
                        st.markdown("---")
                        st.subheader("🔊 Text-to-Speech")
                        
                        if st.button("Generate Audio", type="primary", key="generate_audio_upload"):
                            with st.spinner("Generating audio..."):
                                try:
                                    # Ensure selected_voice is defined (should be from sidebar)
                                    voice_to_use = selected_voice if selected_voice else "Mekdes"
                                    audio_data = generate_tts_audio(text_for_tts, tts_engine, voice_to_use)
                                    
                                    if audio_data and len(audio_data) > 0:
                                        st.audio(audio_data, format='audio/mp3')
                                        st.success("Audio generated successfully!")
                                        
                                        # Download button
                                        st.download_button(
                                            label="Download Audio",
                                            data=audio_data,
                                            file_name="output.mp3",
                                            mime="audio/mp3",
                                            key="download_audio_upload"
                                        )
                                    else:
                                        st.error("Audio generation returned empty data.")
                                except Exception as e:
                                    st.error(f"Error generating audio: {str(e)}")
                                    import traceback
                                    with st.expander("See error details"):
                                        st.code(traceback.format_exc())
                    
                    # Clean up temporary file
                    try:
                        os.unlink(tmp_path)
                    except:
                        pass
            except Exception as e:
                st.error(f"Error processing audio: {str(e)}")
    
    with tab2:
        st.header("Record Voice")
        st.info("Click the button below to start recording. Make sure your microphone is enabled.")
        
        # Streamlit audio recorder
        audio_bytes = st.audio_input("Record your voice", label_visibility="visible")
        
        if audio_bytes is not None:
            # Save recorded audio temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                tmp_file.write(audio_bytes)
                tmp_path = tmp_file.name
            
            try:
                # Load audio
                audio, sr = librosa.load(tmp_path, sr=None)
                
                # Display audio player
                st.audio(tmp_path, format='audio/wav')
                st.info(f"Audio loaded: {len(audio)/sr:.2f} seconds, Sample rate: {sr} Hz")
                
                # Transcribe button
                if st.button("Transcribe Recording", type="primary"):
                    with st.spinner("Transcribing audio..."):
                        transcription = transcribe_audio(model, processor, audio, sr)
                    
                    st.success("Transcription complete!")
                    
                    # Store transcription in session state
                    st.session_state['transcription'] = transcription
                    st.session_state['transcription_source'] = 'record'
                
                # Display transcription if it exists (from session state or current transcription)
                transcription = st.session_state.get('transcription', '')
                if transcription:
                    st.subheader("📝 Transcribed Text")
                    st.text_area(
                        "Transcribed Text",
                        transcription,
                        height=100,
                        key="recorded_transcription",
                        help="The transcribed text from the recording"
                    )
                    
                    # Legal Simplification
                    simplified_text = None
                    text_for_tts = transcription
                    if enable_simplification:
                        st.markdown("---")
                        st.subheader("⚖️ Legal Simplification")
                        
                        if st.button("Simplify Text", type="primary", key="simplify_record"):
                            with st.spinner("Simplifying text with Model 5..."):
                                try:
                                    # Load Model 5
                                    llama_base_model, llama_tokenizer = load_llama_base_model()
                                    model5 = load_model5_adapter(llama_base_model, MODEL_5_ADAPTER_PATH)
                                    
                                    # Get simplification type if using contrastive
                                    simplification_type = None
                                    if use_contrastive:
                                        contrastive_result = load_contrastive_selector()
                                        if contrastive_result:
                                            encoder, centroids = contrastive_result
                                            simplification_type = predict_simplification_type(transcription, encoder, centroids)
                                            st.info(f"Predicted simplification type: **{simplification_type}**")
                                    
                                    # Get RAG context if enabled
                                    rag_context = None
                                    if use_rag:
                                        rag_index, rag_metadata = load_rag_system()
                                        if rag_index is not None and rag_metadata is not None:
                                            rag_context = get_rag_context(transcription, rag_index, rag_metadata)
                                            if rag_context:
                                                st.info(f"Retrieved {len(rag_context)} relevant legal contexts")
                                        else:
                                            st.warning("RAG system not available, running without RAG")
                                    
                                    # Simplify
                                    simplified_text = simplify_text(
                                        transcription,
                                        model5,
                                        llama_tokenizer,
                                        simplification_type=simplification_type,
                                        rag_context=rag_context
                                    )
                                    
                                    st.session_state['simplified_text'] = simplified_text
                                    text_for_tts = simplified_text
                                    st.success("Simplification complete!")
                                    
                                    # Display simplified text
                                    st.text_area(
                                        "Simplified Text",
                                        simplified_text,
                                        height=100,
                                        key="simplified_record",
                                        help="The simplified legal text"
                                    )
                                    
                                except Exception as e:
                                    st.error(f"Error during simplification: {str(e)}")
                                    import traceback
                                    with st.expander("See error details"):
                                        st.code(traceback.format_exc())
                        else:
                            # Check if simplified text exists in session state
                            if 'simplified_text' in st.session_state:
                                simplified_text = st.session_state['simplified_text']
                                text_for_tts = simplified_text
                                st.text_area(
                                    "Simplified Text",
                                    simplified_text,
                                    height=100,
                                    key="simplified_record",
                                    help="The simplified legal text"
                                )
                    
                    # TTS Generation
                    if enable_tts:
                        st.markdown("---")
                        st.subheader("🔊 Text-to-Speech")
                        
                        if st.button("Generate Audio", type="primary", key="generate_audio_record"):
                            with st.spinner("Generating audio..."):
                                try:
                                    # Ensure selected_voice is defined (should be from sidebar)
                                    voice_to_use = selected_voice if selected_voice else "Mekdes"
                                    audio_data = generate_tts_audio(text_for_tts, tts_engine, voice_to_use)
                                    
                                    if audio_data and len(audio_data) > 0:
                                        st.audio(audio_data, format='audio/mp3')
                                        st.success("Audio generated successfully!")
                                        
                                        # Download button
                                        st.download_button(
                                            label="Download Audio",
                                            data=audio_data,
                                            file_name="output.mp3",
                                            mime="audio/mp3",
                                            key="download_audio_record"
                                        )
                                    else:
                                        st.error("Audio generation returned empty data.")
                                except Exception as e:
                                    st.error(f"Error generating audio: {str(e)}")
                                    import traceback
                                    with st.expander("See error details"):
                                        st.code(traceback.format_exc())
                    
                    # Clean up temporary file
                    try:
                        os.unlink(tmp_path)
                    except:
                        pass
            except Exception as e:
                st.error(f"Error processing audio: {str(e)}")
    
    with tab3:
        st.header("📊 Model Evaluation")
        st.markdown("Evaluate the selected ASR model on the Dataset_1.5h test set")
        
        # Check if test set exists
        if not TEST_CSV_PATH.exists():
            st.error(f"Test CSV not found at {TEST_CSV_PATH}")
            st.info("Please ensure the Dataset_1.5h/test.csv file exists")
        elif not TEST_AUDIO_DIR.exists():
            st.error(f"Test audio directory not found at {TEST_AUDIO_DIR}")
            st.info("Please ensure the Dataset_1.5h/audio directory exists")
        else:
            # Load test set
            try:
                test_samples = load_test_set()
                st.info(f"✓ Test set loaded: {len(test_samples)} samples")
                
                # Check for cached results
                cached_results = load_cached_evaluation(selected_adapter_name)
                
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    st.subheader("Evaluation Results")
                    
                    if cached_results:
                        st.success("✓ Cached evaluation results available")
                        st.caption(f"Last evaluated: {cached_results.get('timestamp', 'Unknown')}")
                    else:
                        st.info("No cached results found. Click 'Run Evaluation' to evaluate the model.")
                
                with col2:
                    force_rerun = st.checkbox("Force Re-run", help="Ignore cache and re-run evaluation")
                
                # Initialize results variable
                results = None
                
                # Button to run evaluation
                run_evaluation = st.button("Run Evaluation", type="primary")
                
                # Get results (from cache or run evaluation)
                if run_evaluation or force_rerun:
                    # Run evaluation
                    progress_placeholder = st.empty()
                    
                    with progress_placeholder.container():
                        st.info("Running evaluation on test set...")
                    
                    try:
                        results = evaluate_model(
                            model, 
                            processor, 
                            test_samples, 
                            selected_adapter_name,
                            progress_container=progress_placeholder,
                            force_rerun=force_rerun
                        )
                        progress_placeholder.empty()
                        st.success("✓ Evaluation complete!")
                        # Reload from cache to display
                        results = load_cached_evaluation(selected_adapter_name)
                    except ValueError as e:
                        progress_placeholder.empty()
                        st.error(f"Evaluation failed: {str(e)}")
                        st.info("Check the error messages above to see which samples failed and why.")
                        results = None
                    except Exception as e:
                        progress_placeholder.empty()
                        st.error(f"Unexpected error during evaluation: {str(e)}")
                        import traceback
                        with st.expander("See full error traceback"):
                            st.code(traceback.format_exc())
                        results = None
                elif cached_results:
                    results = cached_results
                    st.info("📊 Displaying cached evaluation results. Click 'Run Evaluation' or check 'Force Re-run' to re-evaluate.")
                else:
                    st.info("Click 'Run Evaluation' to evaluate the model on the test set.")
                
                # Display results
                if results:
                    
                    # Primary Metrics display
                    st.subheader("Primary Metrics")
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric(
                            "Word Error Rate (WER)",
                            f"{results['wer']:.4f}",
                            help="Lower is better (0 = perfect transcription)"
                        )
                    
                    with col2:
                        st.metric(
                            "Character Error Rate (CER)",
                            f"{results['cer']:.4f}",
                            help="Lower is better (0 = perfect transcription)"
                        )
                    
                    with col3:
                        st.metric(
                            "Word Accuracy",
                            f"{results.get('word_accuracy', 1.0 - results['wer']):.4f}",
                            help="Higher is better (1.0 = perfect transcription)"
                        )
                    
                    with col4:
                        st.metric(
                            "Character Accuracy",
                            f"{results.get('char_accuracy', 1.0 - results['cer']):.4f}",
                            help="Higher is better (1.0 = perfect transcription)"
                        )
                    
                    # Additional Metrics
                    st.markdown("---")
                    st.subheader("Additional Metrics")
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric(
                            "Sentence Error Rate (SER)",
                            f"{results.get('ser', 0.0):.4f}",
                            help="Percentage of sentences with at least one error (lower is better)"
                        )
                    
                    with col2:
                        perfect_sentences = results.get('perfect_sentences', sum(1 for ref, pred in zip(results['references'], results['predictions']) if ref == pred))
                        total_sentences = len(results['predictions'])
                        st.metric(
                            "Perfect Sentences",
                            f"{perfect_sentences}/{total_sentences}",
                            help=f"Number of sentences with zero errors ({perfect_sentences/total_sentences*100:.1f}%)"
                        )
                    
                    with col3:
                        # Sentence Accuracy (complement of SER)
                        sentence_accuracy = 1.0 - results.get('ser', 0.0)
                        st.metric(
                            "Sentence Accuracy",
                            f"{sentence_accuracy:.4f}",
                            help="Percentage of sentences with zero errors (higher is better)"
                        )
                    
                    # Additional info
                    st.info(f"Evaluated {results['num_samples']} out of {results['total_samples']} test samples")
                    
                    # Detailed results table
                    st.markdown("---")
                    st.subheader("Detailed Results")
                    
                    # Create DataFrame with results
                    results_df = pd.DataFrame({
                        'File Name': results['file_names'],
                        'Reference': results['references'],
                        'Prediction': results['predictions']
                    })
                    
                    # Add per-sample WER and CER
                    results_df['WER'] = results_df.apply(
                        lambda row: jiwer.wer([row['Reference']], [row['Prediction']]), 
                        axis=1
                    )
                    results_df['CER'] = results_df.apply(
                        lambda row: jiwer.cer([row['Reference']], [row['Prediction']]), 
                        axis=1
                    )
                    
                    # Display table
                    st.dataframe(
                        results_df,
                        use_container_width=True,
                        hide_index=True
                    )
                    
                    # Download results
                    csv = results_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 Download Results as CSV",
                        data=csv,
                        file_name=f"evaluation_results_{selected_adapter_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
                    
                    # Summary statistics
                    st.markdown("---")
                    st.subheader("Summary Statistics")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Mean WER", f"{results_df['WER'].mean():.4f}")
                    with col2:
                        st.metric("Mean CER", f"{results_df['CER'].mean():.4f}")
                    with col3:
                        perfect_matches = (results_df['WER'] == 0).sum()
                        st.metric("Perfect Transcriptions", f"{perfect_matches}/{len(results_df)}")
                    with col4:
                        mean_word_acc = 1.0 - results_df['WER'].mean()
                        st.metric("Mean Word Accuracy", f"{mean_word_acc:.4f}")
                    
                    # Length Statistics
                    if 'length_stats' in results:
                        st.markdown("---")
                        st.subheader("Length Statistics")
                        
                        length_stats = results['length_stats']
                        
                        # Character Length Statistics
                        st.write("**Character Length Statistics:**")
                        char_col1, char_col2, char_col3 = st.columns(3)
                        with char_col1:
                            st.write("*Reference (Ground Truth):*")
                            st.write(f"Mean: {length_stats['char_length']['ref_mean']:.1f} chars")
                            st.write(f"Median: {length_stats['char_length']['ref_median']:.1f} chars")
                            st.write(f"Std Dev: {length_stats['char_length']['ref_std']:.1f} chars")
                        with char_col2:
                            st.write("*Prediction (Model Output):*")
                            st.write(f"Mean: {length_stats['char_length']['pred_mean']:.1f} chars")
                            st.write(f"Median: {length_stats['char_length']['pred_median']:.1f} chars")
                            st.write(f"Std Dev: {length_stats['char_length']['pred_std']:.1f} chars")
                        with char_col3:
                            st.write("*Length Ratio (Pred/Ref):*")
                            st.write(f"Mean: {length_stats['char_length']['ratio_mean']:.3f}")
                            st.write(f"Median: {length_stats['char_length']['ratio_median']:.3f}")
                            if length_stats['char_length']['ratio_mean'] < 0.95:
                                st.warning("⚠️ Predictions are shorter (possible missing words)")
                            elif length_stats['char_length']['ratio_mean'] > 1.05:
                                st.warning("⚠️ Predictions are longer (possible repetitions/insertions)")
                            else:
                                st.success("✓ Length ratio is close to 1.0")
                        
                        # Word Length Statistics
                        st.write("**Word Length Statistics:**")
                        word_col1, word_col2, word_col3 = st.columns(3)
                        with word_col1:
                            st.write("*Reference (Ground Truth):*")
                            st.write(f"Mean: {length_stats['word_length']['ref_mean']:.1f} words")
                            st.write(f"Median: {length_stats['word_length']['ref_median']:.1f} words")
                            st.write(f"Std Dev: {length_stats['word_length']['ref_std']:.1f} words")
                        with word_col2:
                            st.write("*Prediction (Model Output):*")
                            st.write(f"Mean: {length_stats['word_length']['pred_mean']:.1f} words")
                            st.write(f"Median: {length_stats['word_length']['pred_median']:.1f} words")
                            st.write(f"Std Dev: {length_stats['word_length']['pred_std']:.1f} words")
                        with word_col3:
                            st.write("*Length Ratio (Pred/Ref):*")
                            st.write(f"Mean: {length_stats['word_length']['ratio_mean']:.3f}")
                            st.write(f"Median: {length_stats['word_length']['ratio_median']:.3f}")
                            if length_stats['word_length']['ratio_mean'] < 0.95:
                                st.warning("⚠️ Predictions have fewer words (possible missing words)")
                            elif length_stats['word_length']['ratio_mean'] > 1.05:
                                st.warning("⚠️ Predictions have more words (possible repetitions/insertions)")
                            else:
                                st.success("✓ Word count ratio is close to 1.0")
                    
                    # Clear cache button
                    st.markdown("---")
                    if st.button("🗑️ Clear Evaluation Cache", help="Delete cached evaluation results for this model"):
                        cache_path = get_evaluation_cache_path(selected_adapter_name)
                        if cache_path.exists():
                            cache_path.unlink()
                            st.success("Cache cleared! Refresh the page to see updated results.")
                            st.rerun()
            except Exception as e:
                st.error(f"Error loading test set: {str(e)}")
                st.exception(e)

if __name__ == "__main__":
    main()
