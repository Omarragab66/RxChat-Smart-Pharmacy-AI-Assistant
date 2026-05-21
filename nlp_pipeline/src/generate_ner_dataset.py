import pandas as pd

df = pd.read_csv("data/medical_arabic_dataset_large.csv")

rows = []

for _, row in df.iterrows():
    text = row["raw_text"]
    drugs = str(row["drug_entities"]).split("|")

    words = text.replace("؟", "").replace("?", "").split()
    labels = ["O"] * len(words)

    # خريطة عربي للأدوية الإنجليزي
    drug_map = {
        "Congestal": ["كونجستال"],
        "Panadol": ["بنادول"],
        "Brufen": ["بروفين"],
        "Aspirin": ["اسبرين"],
        "Cataflam": ["كتافلام"],
        "Amoxicillin": ["اموكسيسيلين"],
        "Paracetamol": ["باراسيتامول"],
        "Augmentin": ["اوجمنتين"],
        "Zyrtec": ["زيرتك"],
        "Flagyl": ["فلاجيل"]
    }

    for drug in drugs:
        aliases = drug_map.get(drug, [])
        for i, word in enumerate(words):
            if word in aliases:
                labels[i] = "B-DRUG"

    rows.append({
        "tokens": words,
        "ner_tags": labels
    })

ner_df = pd.DataFrame(rows)
ner_df.to_json("data/ner_dataset.json", orient="records", force_ascii=False, lines=True)

print("NER dataset generated successfully")
print(ner_df.head())
print(ner_df.shape)



