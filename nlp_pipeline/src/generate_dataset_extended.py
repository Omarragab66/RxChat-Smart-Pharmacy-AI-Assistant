import pandas as pd
import random

# Load existing dataset
df = pd.read_csv("data/medical_arabic_dataset_large.csv")

# Define more drugs and Arabic aliases
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

# Extra templates (more diverse, semi-real)
extra_templates = {
    "dosage": [
        "كم مرة اخد {drug_ar} في اليوم؟",
        "ما هي جرعة {drug_ar} للبالغين؟",
        "جرعة {drug_ar} للأطفال",
        "متى يجب تناول {drug_ar}؟"
    ],
    "interaction": [
        "هل {drug_ar} يتفاعل مع {drug2_ar}؟",
        "يمكنني اخد {drug_ar} و {drug2_ar} مع بعض؟",
        "في مشكلة لو اخدت {drug_ar} مع {drug2_ar}؟"
    ],
    "side_effects": [
        "ما هي اعراض {drug_ar} الجانبية؟",
        "{drug_ar} يسبب نعاس؟",
        "هل {drug_ar} يسبب دوخة أو صداع؟"
    ],
    "contraindication": [
        "هل {drug_ar} خطر لمرضى الضغط؟",
        "هل يمكن للحامل تناول {drug_ar}؟",
        "هل {drug_ar} مناسب لمرضى السكر؟"
    ]
}

# Generate extra rows
rows = []

drug_items = list(drugs.items())

for _ in range(7000):  # نزيد الداتا لـ 10,000 rows إجمالي
    intent = random.choice(list(extra_templates.keys()))
    template = random.choice(extra_templates[intent])

    drug_ar, drug_en = random.choice(drug_items)
    drug2_ar, drug2_en = random.choice(drug_items)

    while drug2_ar == drug_ar:
        drug2_ar, drug2_en = random.choice(drug_items)

    raw_text = template.format(drug_ar=drug_ar, drug2_ar=drug2_ar)
    
    # Normalize text
    normalized = raw_text
    replacements = {
        "اخد": "أتناول",
        "ينفع": "هل يمكن",
        "بيعمل": "يسبب",
        "كام": "ما",
        "ايه": "ما هي",
        "يتاخد": "يتم تناوله"
    }
    for old, new in replacements.items():
        normalized = normalized.replace(old, new)
    for ar, en in drugs.items():
        normalized = normalized.replace(ar, en)

    if intent == "interaction":
        drug_entities = f"{drug_en}|{drug2_en}"
    else:
        drug_entities = drug_en

    rows.append({
        "raw_text": raw_text,
        "normalized_text": normalized,
        "intent": intent,
        "drug_entities": drug_entities
    })

extra_df = pd.DataFrame(rows)

# Combine with original dataset
combined_df = pd.concat([df, extra_df], ignore_index=True)
combined_df.to_csv("data/medical_arabic_dataset_extended.csv", index=False, encoding="utf-8-sig")

print("Extended dataset generated!")
print(combined_df.shape)
print(combined_df.head())



