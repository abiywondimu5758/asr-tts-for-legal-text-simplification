import torch
from transformers import VitsModel, AutoTokenizer
import scipy

model_id = "facebook/mms-tts-amh"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = VitsModel.from_pretrained(model_id)

# IMPORTANT: This model expects Romanized text.
# Ge'ez: "ማንኛውም ሰው በሕግ ፊት እኩል ነው።"
# Romanized: "mannanyawum sawu behig fit ikul nawu."
romanized_text = "mannanyawum sawu be hig fit ikul nawu."

inputs = tokenizer(romanized_text, return_tensors="pt")

with torch.no_grad():
    output = model(**inputs).waveform

# Save as 16kHz (Ready for Wav2Vec2 fine-tuning)
scipy.io.wavfile.write("legal_test_mms.wav", rate=model.config.sampling_rate, data=output.numpy().T)
print("Meta MMS audio saved as legal_test_mms.wav")