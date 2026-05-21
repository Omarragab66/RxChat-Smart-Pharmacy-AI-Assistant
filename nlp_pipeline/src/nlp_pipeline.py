from predict_intent import predict_intent
from fuzzy_match import correct_drug_name, normalize_drug_names_in_text, extract_drugs_from_text
from predict_ner import predict_drugs_ner
def normalize_query(text, corrected_drugs):
    normalized = text

    replacements = {
        "اخد": "أتناول",
        "ينفع": "هل يمكن",
        "بيعمل": "يسبب",
        "كام": "ما",
    }

    for old, new in replacements.items():
        normalized = normalized.replace(old, new)

    normalized = normalize_drug_names_in_text(normalized)

    return normalized
    normalized = text

    replacements = {
        "اخد": "أتناول",
        "ينفع": "هل يمكن",
        "بيعمل": "يسبب",
        "كام": "ما",
    }

    # استبدال الكلمات العامة
    for old, new in replacements.items():
        normalized = normalized.replace(old, new)

    # استبدال أسماء الأدوية الغلط بالمصححة
    drug_replacements = {
        "كونجستول": "Congestal",
        "كونجستال": "Congestal",
        "بنادوول": "Panadol",
        "بنادول": "Panadol",
        "بروفين": "Brufen",
        "اسبرين": "Aspirin",
        "كتافلام": "Cataflam",
        "اموكسيسيلين": "Amoxicillin"
    }

    for old, new in drug_replacements.items():
        normalized = normalized.replace(old, new)

    return normalized


def process_query(text):
    # 1) BERT NER extracts drug names
    ner_drugs = predict_drugs_ner(text)

    drug_entities = []
    for drug in ner_drugs:
        if drug not in drug_entities:
            drug_entities.append(drug)

    # 2) Fuzzy corrects BERT output
    corrected_drugs = []
    for drug in drug_entities:
        clean_drug = drug.strip()

        if clean_drug.startswith("ال"):
            clean_drug = clean_drug[2:]

        corrected = correct_drug_name(clean_drug)

        if corrected is None:
            corrected = clean_drug

        if corrected not in corrected_drugs:
            corrected_drugs.append(corrected)

    # 3) Backup fuzzy only if BERT NER did not extract anything
    if not corrected_drugs:
        backup_drugs = extract_drugs_from_text(text)

        for drug in backup_drugs:
            if drug not in corrected_drugs:
                corrected_drugs.append(drug)

    # 4) Intent classification
    intent = predict_intent(text)

    # 5) Normalization
    normalized_query = normalize_query(text, corrected_drugs)

    return {
        "normalized_query": normalized_query,
        "intent": intent,
        "drug_entities": drug_entities,
        "corrected_drugs": corrected_drugs
    }
    # 1) BERT NER extracts drug names
    ner_drugs = predict_drugs_ner(text)

    # remove duplicates
    drug_entities = []
    for drug in ner_drugs:
        if drug not in drug_entities:
            drug_entities.append(drug)

    # 2) Fuzzy only corrects extracted names
    corrected_drugs = []
    for drug in drug_entities:
        clean_drug = drug.strip()

        # remove Arabic definite article only if it is at the beginning
        if clean_drug.startswith("ال"):
            clean_drug = clean_drug[2:]

        corrected = correct_drug_name(clean_drug)

        # لو fuzzy معرفش يصحح، متخفيش نتيجة BERT
        if corrected is None:
            corrected = clean_drug

        if corrected not in corrected_drugs:
            corrected_drugs.append(corrected)

    # 3) Intent classification
    intent = predict_intent(text)

    # 4) Normalization
    normalized_query = normalize_query(text, corrected_drugs)

    return {
        "normalized_query": normalized_query,
        "intent": intent,
        "drug_entities": drug_entities,
        "corrected_drugs": corrected_drugs
    }
# Test
if __name__ == "__main__":
    query = "ينفع اخد كونجستول مع بنادوول؟"
    result = process_query(query)
    print(result)



