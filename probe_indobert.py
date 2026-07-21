import torch
from transformers import BertForSequenceClassification, BertTokenizer
from pathlib import Path

model_path = Path("../model/indobert")
print(f"Loading model from {model_path}...")

try:
    model = BertForSequenceClassification.from_pretrained(str(model_path))
    tokenizer = BertTokenizer.from_pretrained(str(model_path))
    
    print("\n--- Model Config ---")
    print("id2label:", model.config.id2label)
    print("label2id:", model.config.label2id)
    print("num_labels:", model.config.num_labels)
    
    # Test samples
    texts = [
        "Aplikasi ini sangat bagus dan bermanfaat sekali!", # likely positive
        "Sangat buruk, jelek banget, kecewa berat.",        # likely negative
        "Biasa saja sih aplikasinya, standar."              # likely neutral
    ]
    
    print("\n--- Testing Inference ---")
    for text in texts:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1).numpy()[0]
        pred = int(torch.argmax(outputs.logits, dim=-1))
        print(f"Text: {text}")
        print(f"  Prediction index: {pred}")
        print(f"  Probs: {['%.4f' % p for p in probs]}")
        
except Exception as e:
    print("Error loading/running model:", e)
