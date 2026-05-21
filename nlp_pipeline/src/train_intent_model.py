import pandas as pd
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import numpy as np
import evaluate


# =========================
# Load Dataset
# =========================

df = pd.read_csv("data/medical_arabic_dataset_large.csv")

print(df.head())
print(df["intent"].value_counts())


# =========================
# Encode Labels
# =========================

label_encoder = LabelEncoder()

df["label"] = label_encoder.fit_transform(df["intent"])

print(label_encoder.classes_)


# =========================
# Train / Test Split
# =========================

train_texts, test_texts, train_labels, test_labels = train_test_split(
    df["raw_text"],
    df["label"],
    test_size=0.2,
    random_state=42
)


train_df = pd.DataFrame({
    "text": train_texts,
    "label": train_labels
})

test_df = pd.DataFrame({
    "text": test_texts,
    "label": test_labels
})


train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)


# =========================
# Load AraBERT
# =========================

model_name = "aubmindlab/bert-base-arabertv02"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=4
)


# =========================
# Tokenization
# =========================

def tokenize_function(example):
    return tokenizer(
        example["text"],
        padding="max_length",
        truncation=True,
        max_length=64
    )


train_dataset = train_dataset.map(tokenize_function, batched=True)
test_dataset = test_dataset.map(tokenize_function, batched=True)


# =========================
# Evaluation Metric
# =========================

accuracy_metric = evaluate.load("accuracy")


def compute_metrics(eval_pred):
    logits, labels = eval_pred

    predictions = np.argmax(logits, axis=-1)

    return accuracy_metric.compute(
        predictions=predictions,
        references=labels
    )


# =========================
# Training Arguments
# =========================

training_args = TrainingArguments(
    output_dir="./models/intent_model",
    eval_strategy="epoch",
    save_strategy="no",
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=1,
    weight_decay=0.01,
    logging_steps=10
)


# =========================
# Trainer
# =========================

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics
)


# =========================
# Train
# =========================

trainer.train()


# =========================
# Save Model
# =========================

trainer.save_model("./models/intent_model")
tokenizer.save_pretrained("./models/intent_model")

print("Model training completed!")



