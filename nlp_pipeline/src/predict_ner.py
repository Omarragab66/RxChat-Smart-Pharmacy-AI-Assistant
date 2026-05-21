import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "ner_model"

id2label = {
    0: "O",
    1: "B-DRUG"
}

tokenizer = AutoTokenizer.from_pretrained(str(MODEL_PATH))
model = AutoModelForTokenClassification.from_pretrained(str(MODEL_PATH))

model.eval()

def predict_drugs_ner(text):
    words = text.replace("؟", "").replace("?", "").split()

    inputs = tokenizer(
        words,
        is_split_into_words=True,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=64
    )

    with torch.no_grad():
        outputs = model(**inputs)

    predictions = torch.argmax(outputs.logits, dim=2)[0].tolist()
    word_ids = inputs.word_ids(batch_index=0)

    found_drugs = []
    previous_word_id = None

    for token_pred, word_id in zip(predictions, word_ids):
        if word_id is None or word_id == previous_word_id:
            continue

        label = id2label[token_pred]

        if label == "B-DRUG":
            found_drugs.append(words[word_id])

        previous_word_id = word_id

    return found_drugs


if __name__ == "__main__":
    tests = [
        "ينفع اخد بنادول مع كونجستال؟",
        "هل بروفين يسبب دوخة؟",
        "جرعة زيرتك كام؟",
        "هل الاسبرين خطر لمرضى الضغط؟"
    ]

    for text in tests:
        print(text, "=>", predict_drugs_ner(text))




