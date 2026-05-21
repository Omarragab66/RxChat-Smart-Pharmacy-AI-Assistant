import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "intent_model_no_leakage"

id2label = {
    0: "contraindication",
    1: "dosage",
    2: "interaction",
    3: "side_effects"
}

tokenizer = AutoTokenizer.from_pretrained(str(MODEL_PATH))
model = AutoModelForSequenceClassification.from_pretrained(str(MODEL_PATH))

model.eval()

def predict_intent(text):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=64
    )

    with torch.no_grad():
        outputs = model(**inputs)

    predicted_id = torch.argmax(outputs.logits, dim=1).item()
    return id2label[predicted_id]


if __name__ == "__main__":
    tests = [
        "الجرعة كام للكونجستال؟",
        "ينفع اخد بنادول مع بروفين؟",
        "زيرتك بيعمل نعاس؟",
        "هل الاسبرين خطر لمرضى الضغط؟"
    ]

    for text in tests:
        print(text, "=>", predict_intent(text))




