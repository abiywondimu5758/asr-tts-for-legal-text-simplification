import re
from pathlib import Path
from typing import List, Set


HOMOPHONE_REPLACEMENTS = {
    # --- The 'H' Group (High frequency in legal texts) ---
    "ሐ": "ሀ", "ሑ": "ሁ", "ሒ": "ሂ", "ሓ": "ሃ", "ሔ": "ሄ", "ሕ": "ህ", "ሖ": "ሆ",
    "ኀ": "ሀ", "ኁ": "ሁ", "ኂ": "ሂ", "ኃ": "ሃ", "ኄ": "ሄ", "ኅ": "ህ", "ኆ": "ሆ",
    "ኸ": "ሀ", "ኹ": "ሁ", "ኺ": "ሂ", "ኻ": "ሃ", "ኼ": "ሄ", "ኽ": "ህ", "ኾ": "ሆ",

    # --- The 'S' Group ---
    "ሠ": "ሰ", "ሡ": "ሱ", "ሢ": "ሲ", "ሣ": "ሳ", "ሤ": "ሴ", "ሥ": "ስ", "ሦ": "ሶ",

    # --- The 'A' Group (Crucial: 1st and 4th orders of 'Ain' look the same: ዓ) ---
    "ዐ": "አ", "ዑ": "ኡ", "ዒ": "ኢ", "ዓ": "አ", "ዔ": "ኤ", "ዕ": "እ", "ዖ": "ኦ",
    "ኣ": "አ",  # Sometimes used specifically for the 4th order sound

    # --- The 'Tse' Group ---
    "ፀ": "ጸ", "ፁ": "ጹ", "ፂ": "ጺ", "ፃ": "ጻ", "ፄ": "ጼ", "ፅ": "ጽ", "ፆ": "ጾ",

    # --- Labialized Consonants (The "Wa" sounds) ---
    # Many legal documents use the compressed character (ሏ) 
    # while some ASR transcripts use the expanded form (ልዋ).
    # Normalizing to the compressed form is standard for training.
    "ሏ": "ልዋ", "ሟ": "ምዋ", "ሯ": "ርዋ", "ሷ": "ስዋ", "ሿ": "ሽዋ", 
    "ቧ": "ብዋ", "ቷ": "ትዋ", "ቿ": "ችዋ", "ኗ": "ንዋ", "ኟ": "ኝዋ", 
    "ዟ": "ዝዋ", "ዧ": "ዥዋ", "ዷ": "ድዋ", "ጇ": "ጅዋ", "ጧ": "ጥዋ", 
    "ጯ": "ጭዋ", "ጿ": "ጽዋ", "ፏ": "ፍዋ", "ኧ": "አ",

    # --- The 'K' and 'G' Labialized Group ---
    # These often appear in words like 'Bakwul' (በኩል/በኵል)
    "ኰ": "ኮ", "ኵ": "ኩ", "ኲ": "ኪ", "ኳ": "ኳ",  # ኳ is already standard
    "ጐ": "ጎ", "ጕ": "ጉ", "ጔ": "ጌ", "ጓ": "ጓ",
    "ቈ": "ቆ", "ቍ": "ቁ", "ቌ": "ቄ", "ቋ": "ቋ",

    # --- Common Transcription Cleanup ---
    "ዉ": "ው",  # Changes 'Wu' to the 6th order 'W' (Standardization)
    "ዪ": "ይ",  # Common phonetic misspelling in transcripts
}

NUMBER_WORDS = {
    0: "ዜሮ",
    1: "አንድ",
    2: "ሁለት",
    3: "ሶስት",
    4: "አራት",
    5: "አምስት",
    6: "ስድስት",
    7: "ሰባት",
    8: "ስምንት",
    9: "ዘጠኝ",
    10: "አስር",
    20: "ሃያ",
    30: "ሰላሳ",
    40: "አርባ",
    50: "ሃምሳ",
    60: "ስልሳ",
    70: "ሰባ",
    80: "ሰማንያ",
    90: "ዘጠና",
    100: "መቶ",
    1000: "ሺህ",
    10000: "አስር ሺህ",
    100000: "መቶ ሺህ"
}

GEZ_NUMERALS = {
    "፩": 1,
    "፪": 2,
    "፫": 3,
    "፬": 4,
    "፭": 5,
    "፮": 6,
    "፯": 7,
    "፰": 8,
    "፱": 9,
    "፲": 10,
    "፳": 20,
    "፴": 30,
    "፵": 40,
    "፶": 50,
    "፷": 60,
    "፸": 70,
    "፹": 80,
    "፺": 90,
    "፻": 100,
    "፼": 10000
}


def normalize_homophones(text: str) -> str:
    for old_char, new_char in HOMOPHONE_REPLACEMENTS.items():
        text = text.replace(old_char, new_char)
    return text


def number_to_amharic_words(num: int) -> str:
    if num == 0:
        return NUMBER_WORDS[0]
    if num in NUMBER_WORDS:
        return NUMBER_WORDS[num]
    
    words = []
    
    if num >= 100000:
        hundreds_thousands = num // 100000
        words.append(number_to_amharic_words(hundreds_thousands) + " መቶ ሺህ")
        num %= 100000
    
    if num >= 10000:
        tens_thousands = num // 10000
        words.append(number_to_amharic_words(tens_thousands) + " አስር ሺህ")
        num %= 10000
    
    if num >= 1000:
        thousands = num // 1000
        words.append(number_to_amharic_words(thousands) + " ሺህ")
        num %= 1000
    
    if num >= 100:
        hundreds = num // 100
        words.append(number_to_amharic_words(hundreds) + " መቶ")
        num %= 100
    
    if num >= 10:
        tens = num // 10
        words.append(NUMBER_WORDS[tens * 10])
        num %= 10
    
    if num > 0:
        words.append(NUMBER_WORDS[num])
    
    return " ".join(words)


def expand_numbers_in_text(text: str) -> str:
    def replace_number(match):
        num_str = match.group(0)
        try:
            num = int(num_str.replace(",", ""))
            return number_to_amharic_words(num)
        except ValueError:
            return num_str
    
    text = re.sub(r'\d+', replace_number, text)
    
    for gez_char, num_value in GEZ_NUMERALS.items():
        text = text.replace(gez_char, number_to_amharic_words(num_value))
    
    return text


def remove_line_number_prefix(text: str) -> str:
    """Remove line number prefixes like '343. ' or '344. ' from the beginning of text"""
    text = re.sub(r'^\d+\.\s+', '', text)
    return text


def remove_quotation_marks(text: str) -> str:
    """Remove quotation marks from text"""
    text = text.replace('"', '')
    text = text.replace('"', '')  # Different Unicode quotation marks
    text = text.replace('"', '')  # Left double quotation mark
    text = text.replace('"', '')  # Right double quotation mark
    text = text.replace(''', '')  # Left single quotation mark
    text = text.replace(''', '')  # Right single quotation mark
    return text


def normalize_punctuation(text: str) -> str:
    text = re.sub(r'[^\u1200-\u137F\s።]', '', text)
    return text


def normalize_whitespace(text: str) -> str:
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text


def clean_sentence(sentence: str) -> str:
    sentence = remove_line_number_prefix(sentence)
    sentence = remove_quotation_marks(sentence)
    sentence = normalize_homophones(sentence)
    sentence = expand_numbers_in_text(sentence)
    sentence = normalize_punctuation(sentence)
    sentence = normalize_whitespace(sentence)
    return sentence


def remove_duplicates(sentences: List[str]) -> tuple[List[str], int]:
    """
    Remove duplicate sentences from a list.
    Returns: (unique_sentences, duplicate_count)
    """
    seen = set()
    unique_sentences = []
    duplicate_count = 0
    
    for sentence in sentences:
        if sentence not in seen:
            seen.add(sentence)
            unique_sentences.append(sentence)
        else:
            duplicate_count += 1
    
    return unique_sentences, duplicate_count


def process_text_file(file_path: Path, remove_dups: bool = True) -> tuple[List[str], int]:
    """
    Process text file: clean and optionally remove duplicates.
    Returns: (cleaned_sentences, duplicate_count)
    """
    content = file_path.read_text(encoding='utf-8')
    lines = content.split('\n')
    
    cleaned_sentences = []
    for line in lines:
        line = line.strip()
        if line and line.endswith('።'):
            cleaned = clean_sentence(line)
            if cleaned and cleaned.endswith('።'):
                cleaned_sentences.append(cleaned)
    
    duplicate_count = 0
    if remove_dups:
        cleaned_sentences, duplicate_count = remove_duplicates(cleaned_sentences)
    
    return cleaned_sentences, duplicate_count


def check_normalization_issues(text: str, file_path: Path = None) -> tuple[List[str], int]:
    """
    Check for normalization issues.
    Returns: (issues_list, duplicate_count)
    """
    issues = []
    
    # Check for line number prefixes
    if re.search(r'^\d+\.\s+', text, re.MULTILINE):
        issues.append("Found line number prefixes (should be removed)")
    
    # Check for quotation marks
    if '"' in text or '"' in text or '"' in text or '"' in text:
        issues.append("Found quotation marks (should be removed)")
    
    for old_char, new_char in HOMOPHONE_REPLACEMENTS.items():
        if old_char in text:
            issues.append(f"Found non-normalized character '{old_char}' (should be '{new_char}')")
    
    if re.search(r'\d', text):
        issues.append("Found numeric digits (should be expanded to words)")
    
    for gez_char in GEZ_NUMERALS.keys():
        if gez_char in text:
            issues.append(f"Found Ge'ez numeral '{gez_char}' (should be expanded to words)")
    
    invalid_chars = re.findall(r'[^\u1200-\u137F\s።]', text)
    if invalid_chars:
        unique_invalid = list(set(invalid_chars))[:5]
        issues.append(f"Found invalid characters: {unique_invalid}")
    
    # Check for duplicates in cleaned sentences
    duplicate_count = 0
    if file_path:
        cleaned_sentences, _ = process_text_file(file_path, remove_dups=False)
        _, duplicate_count = remove_duplicates(cleaned_sentences)
        if duplicate_count > 0:
            issues.append(f"Found {duplicate_count} duplicate sentence(s) (will be removed)")
    
    return issues, duplicate_count


def normalize_file(input_file: Path, output_file: Path, check_only: bool = False):
    print(f"Processing: {input_file.name}")
    
    if not input_file.exists():
        print(f"  ERROR: File not found")
        return
    
    original_content = input_file.read_text(encoding='utf-8')
    original_sentences = [line.strip() for line in original_content.split('\n') if line.strip().endswith('።')]
    
    issues, duplicate_count_check = check_normalization_issues(original_content, input_file if check_only else None)
    if issues:
        print(f"  Found {len(issues)} normalization issues:")
        for issue in issues[:5]:
            print(f"    - {issue}")
        if len(issues) > 5:
            print(f"    ... and {len(issues) - 5} more")
    else:
        print(f"  No normalization issues found (text is clean)")
    
    if not check_only:
        cleaned_sentences, duplicate_count = process_text_file(input_file, remove_dups=True)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_text('\n'.join(cleaned_sentences) + '\n', encoding='utf-8')
        
        # Calculate removed sentences (invalid + duplicates)
        invalid_removed = len(original_sentences) - (len(cleaned_sentences) + duplicate_count)
        print(f"  Processed: {len(cleaned_sentences)} unique sentences")
        if duplicate_count > 0:
            print(f"  Removed: {duplicate_count} duplicate sentence(s)")
        if invalid_removed > 0:
            print(f"  Removed: {invalid_removed} invalid sentence(s)")
        print(f"  Saved to: {output_file.name}")
    else:
        print(f"  Check-only mode: {len(original_sentences)} sentences analyzed")
        if duplicate_count_check > 0:
            print(f"  Would remove: {duplicate_count_check} duplicate sentence(s)")
    
    print()


def main():
    import sys
    
    input_path = Path("Dataset/generated_legal_text")
    output_path = Path("Dataset/normalized_legal_text")
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "--check":
            check_only = True
            mode = "check-only"
        elif sys.argv[1] == "--normalize":
            check_only = False
            mode = "normalize"
        else:
            print("Usage: python normalize_legal_text.py [--check|--normalize]")
            print("  --check: Check files for issues without normalizing")
            print("  --normalize: Normalize files (default)")
            sys.exit(1)
    else:
        check_only = False
        mode = "normalize"
    
    if len(sys.argv) > 2:
        target_file = sys.argv[2]
        input_file = Path(target_file)
        if input_file.exists():
            if check_only:
                output_file = input_file
            else:
                output_path.mkdir(parents=True, exist_ok=True)
                output_file = output_path / input_file.name
            normalize_file(input_file, output_file, check_only)
            return
        else:
            print(f"ERROR: File not found: {target_file}")
            sys.exit(1)
    
    print("=" * 70)
    print("Amharic Legal Text Normalization")
    print("=" * 70)
    print(f"Mode: {mode}")
    print(f"Input directory: {input_path}")
    if not check_only:
        print(f"Output directory: {output_path}")
    print()
    
    if not input_path.exists():
        print(f"ERROR: Input directory not found: {input_path}")
        sys.exit(1)
    
    batch_files = sorted(input_path.glob("legal_text_batch_*.txt"))
    combined_file = input_path / "legal_text_combined.txt"
    
    if not batch_files and not combined_file.exists():
        print("No batch files found")
        sys.exit(1)
    
    if combined_file.exists():
        print(f"Processing combined file: {combined_file.name}")
        if check_only:
            output_file = combined_file
        else:
            output_file = output_path / "legal_text_combined_normalized.txt"
        normalize_file(combined_file, output_file, check_only)
    
    if batch_files:
        print(f"Processing {len(batch_files)} batch files...")
        print()
        
        for batch_file in batch_files:
            if check_only:
                output_file = batch_file
            else:
                output_file = output_path / batch_file.name
            normalize_file(batch_file, output_file, check_only)
    
    if not check_only:
        print("=" * 70)
        print("Normalization Complete")
        print("=" * 70)
        print(f"Normalized files saved to: {output_path}")
    
    if check_only:
        print("=" * 70)
        print("Check Complete")
        print("=" * 70)


if __name__ == "__main__":
    main()

