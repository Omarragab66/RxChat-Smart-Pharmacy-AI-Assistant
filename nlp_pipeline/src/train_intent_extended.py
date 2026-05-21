import pandas as pd
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import numpy as np

# =========================
# Load Dataset Splits
# =========================
train_df = pd.read_csv("data/train.csv")
val_df = pd.read_csv("data/val.csv")
test_df = pd.read_csv("data/test.csv")

# =========================
# Encode Labels
# =========================
label_encoder = LabelEncoder()
train_df["label"] = label_encoder.fit_transform(train_df["intent"])
val_df["label"] = label_encoder.transform(val_df["intent"])
test_df["label"] = label_encoder.transform(test_df["intent"])

id2label = {i: label for i, label in enumerate(label_encoder.classes_)}
label2id = {label: i for i, label in id2label.items()}

# =========================
# Convert to HuggingFace Dataset
# =========================
train_dataset = Dataset.from_pandas(train_df[["raw_text","label"]])
val_dataset = Dataset.from_pandas(val_df[["raw_text","label"]])
test_dataset = Dataset.from_pandas(test_df[["raw_text","label"]])

# =========================
# Load AraBERT
# =========================
model_name = "aubmindlab/bert-base-arabertv02"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=len(label2id)
)

# =========================
# Tokenization
# =========================
def tokenize_function(example):
    return tokenizer(
        example["raw_text"],
        padding="max_length",
        truncation=True,
        max_length=64
    )

train_dataset = train_dataset.map(tokenize_function, batched=True)
val_dataset = val_dataset.map(tokenize_function, batched=True)
test_dataset = test_dataset.map(tokenize_function, batched=True)

# =========================
# Metrics
# =========================
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, predictions)
    return {"accuracy": acc}

# =========================
# Training Arguments
# =========================
training_args = TrainingArguments(
    output_dir="./models/intent_model_extended",
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=50
)

# =========================
# Trainer
# =========================
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics
)

# =========================
# Train
# =========================
trainer.train()

# =========================
# Save Model
# =========================
trainer.save_model("./models/intent_model_extended")
tokenizer.save_pretrained("./models/intent_model_extended")

# =========================
# Evaluate on Test Set
# =========================
y_true = test_df["label"].tolist()
y_pred = []

for text in test_df["raw_text"]:
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=64
    )
    with torch.no_grad():
        outputs = model(**inputs)
    pred = torch.argmax(outputs.logits, dim=1).item()
    y_pred.append(pred)

print("\nAccuracy:", accuracy_score(y_true, y_pred))
print("\nClassification Report:\n", classification_report(y_true, y_pred, target_names=label_encoder.classes_))
print("\nConfusion Matrix:\n", confusion_matrix(y_true, y_pred))



