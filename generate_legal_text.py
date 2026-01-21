import time
import random
import queue
import threading
from pathlib import Path
from datetime import datetime
from typing import Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import google.generativeai as genai

API_KEYS = [
    "REPLACE WITH YOUR API KEY HERE",  # Key 1
    "REPLACE WITH YOUR API KEY HERE",  # Key 2
    "REPLACE WITH YOUR API KEY HERE",  # Key 3
    "REPLACE WITH YOUR API KEY HERE",  # Key 4
    "REPLACE WITH YOUR API KEY HERE",  # Key 5
    "REPLACE WITH YOUR API KEY HERE",  # Key 6
    "REPLACE WITH YOUR API KEY HERE",  # Key 7
    "REPLACE WITH YOUR API KEY HERE",  # Key 8
]

MODEL_NAME = "gemini-2.5-flash"

DOMAIN_VARIATIONS = [
    "FOCUS DOMAIN FOR THIS REQUEST: Emphasize Contracts and Commercial Law, with heavy focus on Conditional Statements (30%) and Legal Definitions (30%).",
    "FOCUS DOMAIN FOR THIS REQUEST: Emphasize Criminal Law and Procedures, with heavy focus on Procedural Language (30%) and Formal Legal Phrasing (30%).",
    "FOCUS DOMAIN FOR THIS REQUEST: Emphasize Constitutional and Administrative Law, with heavy focus on Complex Enumerations (30%) and Conditional Statements (25%).",
    "FOCUS DOMAIN FOR THIS REQUEST: Emphasize Property and Civil Law, with heavy focus on Legal Definitions (30%) and Procedural Language (30%).",
    "FOCUS DOMAIN FOR THIS REQUEST: Emphasize Family Law and Labor Law, with balanced distribution across all sentence types.",
    "FOCUS DOMAIN FOR THIS REQUEST: Emphasize Tax and Financial Law, with heavy focus on Complex Enumerations (30%) and Conditional Statements (30%).",
    "FOCUS DOMAIN FOR THIS REQUEST: Emphasize International Law and Treaties, with heavy focus on Formal Legal Phrasing (30%) and Legal Definitions (30%).",
    "FOCUS DOMAIN FOR THIS REQUEST: Emphasize Court Procedures and Judicial Processes, with heavy focus on Procedural Language (35%) and Formal Legal Phrasing (25%).",
    "FOCUS DOMAIN FOR THIS REQUEST: Emphasize Regulatory and Administrative Procedures, with balanced distribution and extra focus on Complex Enumerations (25%).",
]

BASE_PROMPT = """You are an expert in Amharic legal text generation for ASR (Automatic Speech Recognition) fine-tuning. 

CONTEXT:
The base ASR model (agkphysics/wav2vec2-large-xlsr-53-amharic) was trained on ALFFA corpus (general Amharic: news, conversations). It needs fine-tuning with EXTREMELY CHALLENGING legal domain text to improve performance on complex legal Amharic.

YOUR TASK:
Generate EXACTLY 350-400 (NO MORE, NO LESS) VERY DIFFICULT and COMPLEX Amharic legal sentences. This is a STRICT LIMIT - DO NOT exceed 400 sentences. Focus on maximum complexity - sentences that challenge ASR models significantly more than general speech. Ensure UNIQUE, DIVERSE content - avoid repetition of similar sentences or structures.

CRITICAL REQUIREMENTS:

1. DIFFICULTY: MAXIMUM COMPLEXITY
   - Generate ADVANCED sentences (30-50 words each) with multiple nested clauses
   - Use RARE and SPECIALIZED legal terminology
   - Include COMPLEX grammatical structures not found in everyday speech
   - Create sentences with intricate legal logic and qualifications
   - Prioritize QUALITY and COMPLEXITY over simplicity

2. DIVERSITY REQUIREMENT (CRITICAL):
   - Generate UNIQUE sentences - avoid repeating similar structures or phrases
   - Vary sentence beginnings and constructions
   - Use different legal contexts and scenarios
   - Include diverse vocabulary - don't reuse the same terms repeatedly
   - Ensure each sentence brings unique value and complexity

3. MANDATORY SENTENCE TYPES (Distribute all these structures evenly):

   A. LONG CONDITIONAL STATEMENTS (20-25% of sentences):
      - "If...then..." structures with multiple conditions
      - Conditional clauses with exceptions and qualifications
      - Nested conditional logic (if X and Y, then Z, unless A, in which case B)
      - Complex conditional legal requirements
      Example structure: "እንደዚህ ያለ ሁኔታ ከተፈጠረ እና እንደዚህ ያለ ስምምነት ከተደረሰ የሚመለስ ውጤት አለው።"

   B. LEGAL DEFINITIONS WITH MULTIPLE PARTS (20-25% of sentences):
      - Definitions that include multiple qualifying phrases
      - Legal terms defined with enumerations
      - Definitions with exceptions and special cases
      - Multi-part legal definitions with "and", "or", "including but not limited to" concepts
      Example structure: "X ማለት Y እና Z ወይም A ይሁን B ከተጠቀሰ በስተቀር ማለት ነው።"

   C. PROCEDURAL LANGUAGE (20-25% of sentences):
      - Step-by-step legal procedures
      - Requirements with specific deadlines
      - Procedural sequences with time limits
      - Legal processes with multiple stages and conditions
      - Administrative procedures with qualifications
      Example structure: "X ማድረግ አለብህ ከዚህ በፊት ወይም ከዚህ በኋላ Y ማድረግ ያስፈልጋል እና በZ ጊዜ ውስጥ ማጠናቀቅ አለበት።"

   D. COMPLEX ENUMERATIONS (15-20% of sentences):
      - Lists with multiple items and sub-items
      - Enumerations with qualifications for each item
      - Complex lists with "and", "or", "either...or" structures
      - Multi-level enumerations (first, second, third, and for each...)
      Example structure: "X የሚከተሉትን ያካትታል አንደኛ Y ሁለተኛ Z ሶስተኛ A እና አራተኛ B ሲሆኑ እያንዳንዳቸው የራሳቸው ሁኔታዎች አላቸው።"

   E. FORMAL LEGAL PHRASING (15-20% of sentences):
      - Highly formal and ceremonial legal language
      - Archaic or traditional legal expressions
      - Court language and judicial phrasing
      - Legislative and regulatory language
      - Legal declarations and pronouncements
      Example structure: "ይህ ሕግ እንደሚያስፈልገው እና እንደሚገደድ በመሆኑ የተቋቋመው የፍርድ ቤት ውሳኔ የሚተገበር እና የሚከተል መሆን አለበት።"

4. LEGAL DOMAINS TO COVER (Distribute across requests):
   - Contracts and Commercial Law
   - Civil Law and Property Rights
   - Criminal Law and Procedures
   - Constitutional Law
   - Administrative and Regulatory Law
   - Family Law
   - Procedural Law and Court Processes
   - Labor and Employment Law
   - Tax and Financial Law
   - International Law and Treaties

5. SENTENCE CHARACTERISTICS:
   - Length: 30-50 words per sentence (longer, more complex than general speech)
   - Multiple nested clauses and phrases
   - Advanced grammar: Complex subordinate clauses, relative clauses, participial phrases
   - Common legal vocabulary: Frequently use common legal terms such as አንቀጽ (article), ንኡስ (sub-), ንኡስ አንቀጽ (sub-article), and other standard legal terminology that appears frequently in legal documents
   - Rare legal vocabulary: Include terminology not found in everyday speech
   - Technical precision: Use exact legal language, not simplified versions

6. MANDATORY NORMALIZATION (Apply to ALL text):

   A. HOMOPHONE NORMALIZATION (CRITICAL - Standardize all variants):
      - H Group: Always use ሀ/ሁ/ሂ/ሃ/ሄ/ህ/ሆ (never use ሐ/ሑ/ሒ/ሓ/ሔ/ሕ/ሖ, ኀ/ኁ/ኂ/ኃ/ኄ/ኅ/ኆ, or ኸ/ኹ/ኺ/ኻ/ኼ/ኽ/ኾ)
      - S Group: Always use ሰ/ሱ/ሲ/ሳ/ሴ/ስ/ሶ (never use ሠ/ሡ/ሢ/ሣ/ሤ/ሥ/ሦ)
      - A Group: Always use አ/ኡ/ኢ/ኤ/እ/ኦ (never use ዐ/ዑ/ዒ/ዓ/ዔ/ዕ/ዖ or ኣ)
      - Tse Group: Always use ጸ/ጹ/ጺ/ጻ/ጼ/ጽ/ጾ (never use ፀ/ፁ/ፂ/ፃ/ፄ/ፅ/ፆ)
      - Labialized consonants: Use expanded form (ልዋ not ሏ, ምዋ not ሟ, ርዋ not ሯ, etc.)
      - K/G Labialized: Use standard forms (ኮ not ኰ, ኩ not ኵ, ጎ not ጐ, ጉ not ጕ, ቆ not ቈ, ቁ not ቍ)
      - Common cleanup: Always use ው (not ዉ), ይ (not ዪ)

   B. NUMBER EXPANSION (CRITICAL - Expand ALL numbers to words):
      - Arabic numerals (1, 2, 3, 25, 1988, etc.) → Expand to full Amharic words
      - Ge'ez numerals (፩, ፪, ፫, ፬, ፭, ፮, ፯, ፰, ፱, ፲, ፳, ፴, etc.) → Expand to full Amharic words
      - Article numbers: "አንቀጽ 147" → "አንቀጽ አንድ መቶ አራት አስር ሰባት"
      - Years: "2024" → "ሁለት ሺህ ሃያ አራት"
      - Dates: Expand completely (day, month, year all as words)
      - Ordinals: "1ኛ" → "አንደኛ", "2ኛ" → "ሁለተኛ"
      - Quantities and measurements: All numbers as words
      - NO DIGITS OR GE'EZ NUMERALS in final text - ONLY WRITTEN WORDS

   C. PUNCTUATION:
      - Use ONLY the sentence-ending period "።" (Arat Neteb)
      - Remove ALL other punctuation: commas, colons, semicolons, parentheses, quotation marks, brackets, dashes, etc.
      - One sentence per line (each sentence ends with "።" on its own line)

   D. WHITESPACE:
      - Single space between words
      - No multiple spaces
      - No leading/trailing spaces
      - Clean, normalized spacing

7. OUTPUT FORMAT:
   - Plain text format
   - ONE sentence per line
   - Each sentence ends with "።"
   - UTF-8 encoding
   - No headers, no numbering, no formatting
   - Ready for direct use in TTS → ASR pipeline

8. QUALITY STANDARDS:
   - Maximum complexity and difficulty
   - Authentic Ethiopian legal terminology
   - Natural flow for speech synthesis (should sound natural when spoken)
   - Advanced grammatical structures
   - Rare and specialized legal vocabulary
   - Sentences that significantly challenge ASR models
   - UNIQUE content - no repetitive patterns

CRITICAL: Generate EXACTLY 350-400 sentences - DO NOT exceed 400 sentences. Count carefully. If you reach 400 sentences, STOP immediately. This is a hard limit to prevent timeout errors.

Generate exactly 350-400 sentences now. Ensure ALL sentence types are represented with proper distribution. Apply ALL normalization rules automatically. Create UNIQUE, DIVERSE content. The output must be TTS-ready and optimized for ASR fine-tuning with maximum difficulty and complexity."""

MIN_DELAY_BETWEEN_REQUESTS = 12.0
MAX_DELAY_BETWEEN_REQUESTS = 18.0


class APIKeyManager:
    def __init__(self, api_keys):
        self.key_queue = queue.Queue()
        self.key_lock = threading.Lock()
        self.key_last_used = {}
        self.key_usage_count = {}
        
        for key in api_keys:
            self.key_queue.put(key)
            self.key_last_used[key] = 0.0
            self.key_usage_count[key] = 0
    
    def get_available_key(self) -> Optional[str]:
        try:
            key = self.key_queue.get(timeout=60)
            current_time = time.time()
            last_used = self.key_last_used[key]
            time_since_last_use = current_time - last_used
            
            if time_since_last_use < MIN_DELAY_BETWEEN_REQUESTS:
                wait_time = MIN_DELAY_BETWEEN_REQUESTS - time_since_last_use
                time.sleep(wait_time)
            
            return key
        except queue.Empty:
            return None
    
    def release_key(self, key: str):
        with self.key_lock:
            self.key_last_used[key] = time.time()
            self.key_usage_count[key] += 1
        self.key_queue.put(key)
    
    def get_usage_stats(self):
        with self.key_lock:
            return dict(self.key_usage_count)


def create_prompt_with_domain(domain_variation: str) -> str:
    prompt = BASE_PROMPT.replace(
        "YOUR TASK:",
        f"YOUR TASK:\n\n{domain_variation}\n\n"
    )
    return prompt


def generate_text(api_key: str, prompt: str, key_index: int, batch_num: int) -> Optional[str]:
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(MODEL_NAME)
        
        response = model.generate_content(prompt)
        
        if response and response.text:
            return response.text.strip()
        else:
            print(f"  ERROR: Empty response from API key {key_index + 1}")
            return None
    except Exception as e:
        print(f"  ERROR: Exception with API key {key_index + 1}: {str(e)}")
        return None


def format_output_filename(batch_num: int, key_index: int, domain_index: int) -> str:
    return f"legal_text_batch_{batch_num:03d}_key{key_index + 1}_domain{domain_index + 1}.txt"


def file_exists(output_dir: Path, batch_num: int, key_index: int, domain_index: int) -> bool:
    """Check if the output file already exists"""
    filename = format_output_filename(batch_num, key_index, domain_index)
    file_path = output_dir / filename
    return file_path.exists() and file_path.stat().st_size > 0


def process_batch_worker(api_key: str, key_index: int, batches: list, output_dir: Path, lock: threading.Lock, stats: dict):
    successful = 0
    failed = 0
    skipped = 0
    
    for batch_num, domain_index in batches:
        output_file = output_dir / format_output_filename(batch_num, key_index, domain_index)
        
        if file_exists(output_dir, batch_num, key_index, domain_index):
            with lock:
                print(f"[Batch {batch_num + 1}] Key {key_index + 1} skipping...")
                print(f"  File already exists: {output_file.name}")
            skipped += 1
            continue
        
        domain_variation = DOMAIN_VARIATIONS[domain_index]
        prompt = create_prompt_with_domain(domain_variation)
        
        with lock:
            print(f"[Batch {batch_num + 1}] Key {key_index + 1} processing...")
            domain_name = domain_variation.split('Emphasize')[1].split(',')[0].strip() if 'Emphasize' in domain_variation else 'Balanced'
            print(f"  Domain: {domain_name}")
            print(f"  Sending request to {MODEL_NAME}...")
        
        request_start = time.time()
        result = generate_text(api_key, prompt, key_index, batch_num)
        request_duration = time.time() - request_start
        
        if file_exists(output_dir, batch_num, key_index, domain_index):
            with lock:
                print(f"  Key {key_index + 1} Batch {batch_num + 1}: SKIPPED (file created by another process)")
            skipped += 1
            continue
        
        if result:
            output_file.write_text(result, encoding='utf-8')
            
            sentence_count = len([line for line in result.split('\n') if line.strip().endswith('።')])
            char_count = len(result)
            
            with lock:
                print(f"  Key {key_index + 1} Batch {batch_num + 1}: SUCCESS - {sentence_count} sentences ({char_count} chars, {request_duration:.1f}s)")
            successful += 1
        else:
            with lock:
                print(f"  Key {key_index + 1} Batch {batch_num + 1}: FAILED")
            failed += 1
        
        if batch_num < batches[-1][0]:
            delay = random.uniform(MIN_DELAY_BETWEEN_REQUESTS, MAX_DELAY_BETWEEN_REQUESTS)
            time.sleep(delay)
    
    with lock:
        stats['successful'] += successful
        stats['failed'] += failed
        stats['skipped'] += skipped
        stats['key_usage'][key_index] = len(batches)
    
    return successful, failed, skipped


def main():
    print("=" * 70)
    print("Amharic Legal Text Generation for ASR Fine-tuning")
    print("=" * 70)
    print()
    
    placeholder_keys = [key for key in API_KEYS if "YOUR_API_KEY" in key or len(key) < 20]
    if placeholder_keys:
        print("WARNING: Please configure API_KEYS in the script before running")
        print("Current API keys appear to be placeholder values")
        response = input("Continue anyway? (yes/no): ")
        if response.lower() != "yes":
            return
    
    num_batches = input("Enter number of batches to generate (default: 10): ").strip()
    try:
        num_batches = int(num_batches) if num_batches else 10
    except ValueError:
        num_batches = 10
    
    output_dir = Path("Dataset/generated_legal_text")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print()
    print(f"Configuration:")
    print(f"  Model: {MODEL_NAME}")
    print(f"  API Keys: {len(API_KEYS)}")
    print(f"  Batches: {num_batches}")
    print(f"  Output Directory: {output_dir}")
    print(f"  Sentences per batch: 350-400")
    print(f"  Execution mode: Parallel (3 workers)")
    print(f"  Delay between requests: {MIN_DELAY_BETWEEN_REQUESTS}-{MAX_DELAY_BETWEEN_REQUESTS} seconds")
    print()
    
    num_keys = len(API_KEYS)
    batches_per_key = num_batches // num_keys
    extra_batches = num_batches % num_keys
    
    batch_assignments = []
    batch_num = 0
    for key_index in range(num_keys):
        key_batches = batches_per_key + (1 if key_index < extra_batches else 0)
        batches = []
        for _ in range(key_batches):
            domain_index = batch_num % len(DOMAIN_VARIATIONS)
            batches.append((batch_num, domain_index))
            batch_num += 1
        batch_assignments.append(batches)
    
    print(f"Batch distribution:")
    for key_idx, batches in enumerate(batch_assignments):
        print(f"  Key {key_idx + 1}: {len(batches)} batches")
    print()
    
    lock = threading.Lock()
    stats = {'successful': 0, 'failed': 0, 'skipped': 0, 'key_usage': {}}
    
    start_time = time.time()
    
    print("Starting parallel generation with 3 workers...")
    print()
    
    with ThreadPoolExecutor(max_workers=num_keys) as executor:
        futures = []
        for key_index, batches in enumerate(batch_assignments):
            if batches:
                future = executor.submit(process_batch_worker, API_KEYS[key_index], key_index, batches, output_dir, lock, stats)
                futures.append(future)
        
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                with lock:
                    print(f"Worker error: {e}")
    
    total_duration = time.time() - start_time
    
    print()
    print("=" * 70)
    print("Generation Complete")
    print("=" * 70)
    print(f"Total batches requested: {num_batches}")
    print(f"Successful batches: {stats['successful']}")
    print(f"Failed batches: {stats['failed']}")
    print(f"Skipped batches (already exist): {stats['skipped']}")
    print(f"Total duration: {total_duration / 60:.1f} minutes")
    print()
    print("API Key Usage Statistics:")
    for key_idx in range(num_keys):
        count = stats['key_usage'].get(key_idx, 0)
        print(f"  Key {key_idx + 1}: {count} batches")
    print()
    
    print("Combining all files into master file...")
    master_file = output_dir / "legal_text_combined.txt"
    
    existing_sentences = set()
    total_sentences = 0
    
    all_files = sorted(output_dir.glob("legal_text_batch_*.txt"))
    if all_files:
        with open(master_file, 'w', encoding='utf-8') as master:
            for file_path in all_files:
                if file_path.name == master_file.name:
                    continue
                try:
                    content = file_path.read_text(encoding='utf-8').strip()
                    if content:
                        sentences = [line.strip() for line in content.split('\n') if line.strip().endswith('።')]
                        for sentence in sentences:
                            if sentence not in existing_sentences:
                                master.write(sentence + '\n')
                                existing_sentences.add(sentence)
                                total_sentences += 1
                except Exception as e:
                    print(f"  Warning: Could not read {file_path.name}: {e}")
        
        print(f"  Combined {len(all_files)} files into master file")
        print(f"  Total unique sentences: {total_sentences}")
        print(f"  Master file: {master_file.name}")
    else:
        print("  No batch files found to combine")
    
    print()
    print(f"Individual batch files saved to: {output_dir}")
    print(f"Master combined file: {master_file}")


if __name__ == "__main__":
    main()

