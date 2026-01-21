# Setup Instructions for Amharic PDF Text Extraction

Since Abyssinica OCR is a Windows Word add-in and not programmatically accessible, we're using a Python-based solution that can handle both text-based and image-based PDFs.

## Option 1: Text Extraction (for text-based PDFs)

If your PDFs contain selectable text, this is the fastest method:

```bash
# Install pdfplumber
pip install pdfplumber
```

## Option 2: OCR Extraction (for image-based/scanned PDFs)

If your PDFs are scanned images, you'll need OCR:

### Step 1: Install Tesseract OCR

**macOS (using Homebrew):**
```bash
brew install tesseract
brew install tesseract-lang  # This includes Amharic language data
```

**Verify installation:**
```bash
tesseract --list-langs
```
You should see `amh` (Amharic) in the list.

### Step 2: Install Python OCR libraries

```bash
pip install pytesseract pdf2image pillow
```

**Note:** `pdf2image` also requires `poppler` on macOS:
```bash
brew install poppler
```

## Usage

### Test on a single PDF:
```bash
python extract_amharic_text.py Dataset/raw_legal_text/1356.pdf
```

### Extract text from a single PDF and save to file:
```bash
python extract_amharic_text.py Dataset/raw_legal_text/1356.pdf output.txt
```

### Process all PDFs in a directory:
```bash
python extract_amharic_text.py --dir Dataset/raw_legal_text Dataset/extracted_text
```

### Force OCR (skip text extraction):
```bash
python extract_amharic_text.py --dir Dataset/raw_legal_text Dataset/extracted_text --ocr
```

## How It Works

1. **First attempt:** Tries to extract text directly from PDF (if pdfplumber is installed)
2. **Fallback:** If text extraction fails or returns minimal text, automatically uses OCR
3. **Output:** Saves extracted text as UTF-8 encoded `.txt` files

## Testing Your PDFs

Run this to test if your PDFs are text-based:

```bash
python extract_amharic_text.py Dataset/raw_legal_text/1356.pdf test_output.txt
```

Then check `test_output.txt`:
- If you see readable Amharic text → PDFs are text-based (good!)
- If you see garbled text or very little content → PDFs are image-based (need OCR)

## Troubleshooting

### "Tesseract not found"
- Make sure Tesseract is installed: `brew install tesseract tesseract-lang`
- Verify: `which tesseract`

### "Amharic language not available"
- Install language pack: `brew install tesseract-lang`
- Or download manually from: https://github.com/tesseract-ocr/tessdata
- Place `amh.traineddata` in Tesseract's tessdata directory

### "pdf2image failed"
- Install poppler: `brew install poppler`
- On Linux: `sudo apt-get install poppler-utils`
- On Windows: Download from: https://github.com/oschwartz10612/poppler-windows/releases

### Permission errors with pip
- Try: `pip install --user pdfplumber pytesseract pdf2image pillow`
- Or use your virtual environment: `source venv/bin/activate` then `pip install ...`

## Next Steps

After extraction:
1. Review the extracted text files
2. Clean the text (normalize homophones, expand numbers, etc.)
3. Split into sentences
4. Filter for useful legal sentences
5. Generate audio with your TTS scripts




