import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score
)

# =========================
# Load Dataset
# =========================

df = pd.read_csv("data/medical_arabic_dataset_large.csv")

# =========================
# Label Mapping
# =========================

label2id = {
    "contraindication": 0,
    "dosage": 1,
    "interaction": 2,
    "side_effects": 3
}

id2label = {
    0: "contraindication",
    1: "dosage",
    2: "interaction",
    3: "side_effects"
}

# =========================
# Load Model
# =========================

MODEL_PATH = "models/intent_model"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)

model.eval()

# =========================
# Prediction Function
# =========================

def predict(text):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=64
    )

    with torch.no_grad():
        outputs = model(**inputs)

    pred_id = torch.argmax(outputs.logits, dim=1).item()

    return pred_id

# =========================
# Predict All
# =========================

y_true = []
y_pred = []

for _, row in df.iterrows():

    text = row["raw_text"]
    true_label = label2id[row["intent"]]

    pred_label = predict(text)

    y_true.append(true_label)
    y_pred.append(pred_label)

# =========================
# Metrics
# =========================

acc = accuracy_score(y_true, y_pred)

print("\\nAccuracy:\\n")
print(acc)

print("\\nClassification Report:\\n")
print(classification_report(y_true, y_pred, target_names=id2label.values()))

print("\\nConfusion Matrix:\\n")
print(confusion_matrix(y_true, y_pred))



