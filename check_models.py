import google.generativeai as genai

API_KEY = "REPLACE WITH YOUR API KEY HERE"

print("=" * 70)
print("Checking Available Gemini Models")
print("=" * 70)
print()

genai.configure(api_key=API_KEY)

try:
    models = genai.list_models()
    
    print("Available models:")
    print("-" * 70)
    
    model_list = []
    for model in models:
        model_name = model.name
        if 'models/' in model_name:
            model_name = model_name.replace('models/', '')
        
        if 'generateContent' in model.supported_generation_methods:
            model_list.append(model_name)
            print(f"  ✓ {model_name}")
            if hasattr(model, 'display_name') and model.display_name:
                print(f"    Display Name: {model.display_name}")
            if hasattr(model, 'description') and model.description:
                print(f"    Description: {model.description}")
            print()
    
    print("=" * 70)
    print(f"Total models with generateContent support: {len(model_list)}")
    print()
    print("Recommended models for text generation:")
    for model in model_list:
        if 'flash' in model.lower() or 'pro' in model.lower():
            print(f"  - {model}")
    
except Exception as e:
    print(f"Error listing models: {e}")
    print()
    print("Trying alternative method...")
    try:
        from google.generativeai import models
        models_list = models.list_models()
        for model in models_list:
            print(f"  {model}")
    except Exception as e2:
        print(f"Alternative method also failed: {e2}")

