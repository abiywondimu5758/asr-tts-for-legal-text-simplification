# download_text_models.py

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

models = {
    "mT5-base": "google/mt5-base",
    "mBART-50": "facebook/mbart-large-50-many-to-many-mmt"
}

for name, model_id in models.items():
    print(f"\nDownloading {name}: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
    print(f"Finished downloading {name}")

print("\nAll text models downloaded and cached successfully.")
