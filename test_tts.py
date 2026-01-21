# test_piper_tts.py
import subprocess
import sys
import os
from pathlib import Path
import requests
import json

# Test cases with legal domain Amharic text (Ge'ez script - Piper supports it directly)
test_cases = [
    {
        "name": "Legal domain - contract definition",
        "text": "ውል ማለት በሁለት ወይም ከሁለት በላይ በሆኑ ሰዎች መካከል የሚደረግ ስምምነት ነው።"
    },
    {
        "name": "Legal domain - equality before law",
        "text": "ማንኛውም ሰው በሕግ ፊት እኩል ነው።"
    },
    {
        "name": "Legal domain - constitutional rights",
        "text": "አንቀፅ አምስት የኢትዮጵያ ዜጎች የፖለቲካ መብቶችን ያቀልማል።"
    },
    {
        "name": "Simple test",
        "text": "ሰላም ለሁላችሁ።"
    }
]

def check_piper_installed():
    """Check if Piper CLI is installed"""
    try:
        result = subprocess.run(
            ["piper", "--version"],
            capture_output=True,
            text=True,
            timeout=5
        )
        return result.returncode == 0, result.stdout
    except FileNotFoundError:
        return False, "Piper CLI not found"
    except subprocess.TimeoutExpired:
        return False, "Piper command timed out"

def check_piper_python():
    """Check if piper-tts Python package is available"""
    try:
        import piper_tts
        return True, "piper-tts package found"
    except ImportError:
        return False, "piper-tts package not found"

def download_piper_model(model_name="am_ET-kdl-medium", output_dir="piper_models"):
    """Download a Piper model from HuggingFace"""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    model_dir = output_path / model_name
    if model_dir.exists() and (model_dir / "model.onnx").exists():
        print(f"Model already exists at {model_dir}")
        return str(model_dir / "model.onnx"), str(model_dir / "config.json")
    
    print(f"Attempting to download model: {model_name}")
    print("NOTE: Piper models need to be downloaded manually from:")
    print(f"https://huggingface.co/rhasspy/piper-voices/tree/main/am/am_ET/{model_name}")
    print("\nOr use Piper's download script:")
    print(f"python -m piper.download --language am --output-dir {output_dir}")
    
    return None, None

def test_with_piper_cli(model_path, text, output_file):
    """Test Piper using CLI"""
    try:
        # Piper CLI command: piper --model model.onnx --text "text" --output-file output.wav
        cmd = [
            "piper",
            "--model", model_path,
            "--text", text,
            "--output-file", output_file
        ]
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if result.returncode == 0:
            return True, "Success"
        else:
            return False, f"Error: {result.stderr}"
            
    except FileNotFoundError:
        return False, "Piper CLI not found. Install with: pip install piper-tts"
    except subprocess.TimeoutExpired:
        return False, "Command timed out"
    except Exception as e:
        return False, f"Exception: {str(e)}"

def test_with_piper_python(model_path, text, output_file):
    """Test Piper using Python package"""
    try:
        import piper_tts
        
        # Initialize Piper
        engine = piper_tts.PiperVoice.load(model_path)
        
        # Generate audio
        with open(output_file, 'wb') as f:
            engine.synthesize(text, f)
        
        return True, "Success"
    except Exception as e:
        return False, f"Error: {str(e)}"

def main():
    print("=" * 70)
    print("Testing Piper TTS for Amharic")
    print("=" * 70)
    print()
    
    # Check installation
    print("Checking Piper installation...")
    piper_cli_ok, cli_msg = check_piper_installed()
    piper_python_ok, python_msg = check_piper_python()
    
    print(f"  Piper CLI: {'✓' if piper_cli_ok else '✗'} {cli_msg}")
    print(f"  Piper Python: {'✓' if piper_python_ok else '✗'} {python_msg}")
    print()
    
    if not piper_cli_ok and not piper_python_ok:
        print("=" * 70)
        print("INSTALLATION REQUIRED")
        print("=" * 70)
        print()
        print("Piper TTS is not installed. To install:")
        print()
        print("Option 1: Install Piper Python package (Recommended for testing):")
        print("  pip install piper-tts")
        print()
        print("Option 2: Install Piper CLI (for production use):")
        print("  git clone https://github.com/rhasspy/piper.git")
        print("  cd piper")
        print("  python3 -m venv venv")
        print("  source venv/bin/activate  # On Windows: venv\\Scripts\\activate")
        print("  pip install -r requirements.txt")
        print()
        print("Then download the Amharic model:")
        print("  python -m piper.download --language am --output-dir piper_models")
        print()
        print("Or download manually from:")
        print("  https://huggingface.co/rhasspy/piper-voices/tree/main/am/am_ET")
        print()
        sys.exit(1)
    
    # Try to find or download model
    print("Checking for Amharic model...")
    model_path, config_path = download_piper_model()
    
    if not model_path:
        print("\nPlease download the Amharic model first.")
        print("After downloading, update the MODEL_PATH in this script.")
        sys.exit(1)
    
    # Test with sample text
    print("\n" + "=" * 70)
    print("Testing Piper TTS with Legal Domain Amharic Text")
    print("=" * 70)
    print()
    
    output_dir = Path("piper_test_outputs")
    output_dir.mkdir(exist_ok=True)
    
    for i, test in enumerate(test_cases, 1):
        print(f"Test {i}: {test['name']}")
        print(f"Text (Ge'ez): {test['text']}")
        
        output_file = output_dir / f"test_piper_{i}.wav"
        
        # Try Python package first, then CLI
        if piper_python_ok:
            success, msg = test_with_piper_python(model_path, test['text'], str(output_file))
        elif piper_cli_ok:
            success, msg = test_with_piper_cli(model_path, test['text'], str(output_file))
        else:
            success, msg = False, "No Piper installation found"
        
        if success:
            file_size = output_file.stat().st_size / 1024
            print(f"✓ Audio saved: {output_file} ({file_size:.1f} KB)")
        else:
            print(f"✗ Failed: {msg}")
        
        print("-" * 70)
        print()
    
    print("Testing complete!")
    print(f"\nOutput files saved in: {output_dir.absolute()}")
    print("\nCompare the audio quality with:")
    print("  - MMS-TTS (facebook/mms-tts-amh)")
    print("  - edge-tts (Mekdes, Ameha voices)")
    print("  - gTTS")

if __name__ == "__main__":
    main()