import pandas as pd
import random

drugs = {
    "كونجستال": "Congestal",
    "بنادول": "Panadol",
    "بروفين": "Brufen",
    "اسبرين": "Aspirin",
    "كتافلام": "Cataflam",
    "اموكسيسيلين": "Amoxicillin",
    "باراسيتامول": "Paracetamol",
    "اوجمنتين": "Augmentin",
    "زيرتك": "Zyrtec",
    "فلاجيل": "Flagyl"
}

templates = {
    "dosage": [
        "الجرعة كام ل {drug_ar}؟",
        "اخد {drug_ar} كام مرة في اليوم؟",
        "جرعة {drug_ar} للاطفال كام؟",
        "ينفع اخد جرعة من {drug_ar}؟",
        "طريقة استخدام {drug_ar} ايه؟"
    ],
    "interaction": [
        "ينفع اخد {drug_ar} مع {drug2_ar}؟",
        "هل {drug_ar} يتاخد مع {drug2_ar}؟",
        "في تفاعل بين {drug_ar} و {drug2_ar}؟",
        "اقدر اخد {drug_ar} و {drug2_ar} مع بعض؟",
        "هل الجمع بين {drug_ar} و {drug2_ar} خطر؟"
    ],
    "side_effects": [
        "{drug_ar} بيعمل نعاس؟",
        "ايه اعراض {drug_ar} الجانبية؟",
        "{drug_ar} ممكن يسبب دوخة؟",
        "هل {drug_ar} يسبب صداع؟",
        "بعد ما اخدت {drug_ar} حسيت بتعب"
    ],
    "contraindication": [
        "هل {drug_ar} خطر لمرضى الضغط؟",
        "ينفع الحامل تاخد {drug_ar}؟",
        "هل {drug_ar} مناسب لمرضى السكر؟",
        "مين ممنوع ياخد {drug_ar}؟",
        "هل {drug_ar} خطر على الكلى؟"
    ]
}

def normalize_text(text, drug_map):
    normalized = text
    replacements = {
    "يتاخد": "يتم تناوله",
    "اخد": "أتناول",
    "ينفع": "هل يمكن",
    "بيعمل": "يسبب",
    "كام": "ما",
    "ايه": "ما هي"
}

    for old, new in replacements.items():
        normalized = normalized.replace(old, new)

    for ar, en in drug_map.items():
        normalized = normalized.replace(ar, en)

    return normalized

rows = []
drug_items = list(drugs.items())

for _ in range(3000):
    intent = random.choice(list(templates.keys()))
    template = random.choice(templates[intent])

    drug_ar, drug_en = random.choice(drug_items)
    drug2_ar, drug2_en = random.choice(drug_items)

    while drug2_ar == drug_ar:
        drug2_ar, drug2_en = random.choice(drug_items)

    raw_text = template.format(drug_ar=drug_ar, drug2_ar=drug2_ar)
    normalized_text = normalize_text(raw_text, drugs)

    if intent == "interaction":
        drug_entities = f"{drug_en}|{drug2_en}"
    else:
        drug_entities = drug_en

    rows.append({
        "raw_text": raw_text,
        "normalized_text": normalized_text,
        "intent": intent,
        "drug_entities": drug_entities
    })

df = pd.DataFrame(rows)
df.to_csv("data/medical_arabic_dataset_large.csv", index=False, encoding="utf-8-sig")

print("Dataset generated successfully")
print(df.head())
print(df.shape)



