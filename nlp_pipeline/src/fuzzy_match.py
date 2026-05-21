from rapidfuzz import process
import re


# Canonical names used by the system
drug_dictionary = [
    "Congestal",
    "Panadol",
    "Paracetamol",
    "Brufen",
    "Ibuprofen",
    "Aspirin",
    "Cataflam",
    "Diclofenac",
    "Amoxicillin",
    "Augmentin",
    "Zyrtec",
    "Cetirizine",
    "Flagyl",
    "Metronidazole",
    "Voltaren",
    "Antinal",
    "Motilium",
    "Nexium",
    "Omeprazole",
    "Telfast",
    "Loratadine",
    "Aerius",
    "Ventolin",
    "Salbutamol",
    "Prednisone",
    "Decancit",
    "Strepsils"
]


# Arabic / English / typo aliases -> canonical drug name
arabic_aliases = {
    # Congestal
    "كونجستال": "Congestal",
    "كونجستول": "Congestal",
    "كونجيستال": "Congestal",
    "congestal": "Congestal",

    # Panadol / Paracetamol
    "بانادول": "Panadol",
    "بنادول": "Panadol",
    "بنادوول": "Panadol",
    "panadol": "Panadol",
    "panadol extra": "Panadol",

    "باراسيتامول": "Paracetamol",
    "براسيتامول": "Paracetamol",
    "باراستامول": "Paracetamol",
    "paracetamol": "Paracetamol",
    "acetaminophen": "Paracetamol",

    # Brufen / Ibuprofen
    "بروفين": "Brufen",
    "بروفن": "Brufen",
    "brufen": "Brufen",

    "ايبوبروفين": "Ibuprofen",
    "ايبو بروفين": "Ibuprofen",
    "ibuprofen": "Ibuprofen",

    # Aspirin
    "اسبرين": "Aspirin",
    "الاسبرين": "Aspirin",
    "أسبرين": "Aspirin",
    "aspirin": "Aspirin",

    # Cataflam / Diclofenac
    "كتافلام": "Cataflam",
    "كتفلام": "Cataflam",
    "كاتافلام": "Cataflam",
    "cataflam": "Cataflam",

    "ديكلوفيناك": "Diclofenac",
    "diclofenac": "Diclofenac",

    # Amoxicillin / Augmentin
    "اموكسيسيلين": "Amoxicillin",
    "اموكسسلين": "Amoxicillin",
    "اموكسيل": "Amoxicillin",
    "اموكسلين": "Amoxicillin",
    "amoxicillin": "Amoxicillin",
    "amoxil": "Amoxicillin",

    "اوجمنتين": "Augmentin",
    "أوجمنتين": "Augmentin",
    "augmentin": "Augmentin",

    # Zyrtec / Cetirizine
    "زيرتك": "Zyrtec",
    "زرتك": "Zyrtec",
    "zyrtec": "Zyrtec",

    "سيتريزين": "Cetirizine",
    "سيتريزين": "Cetirizine",
    "cetirizine": "Cetirizine",

    # Flagyl / Metronidazole
    "فلاجيل": "Flagyl",
    "فلاچيل": "Flagyl",
    "flagyl": "Flagyl",

    "ميترونيدازول": "Metronidazole",
    "metronidazole": "Metronidazole",

    # Voltaren
    "فولتارين": "Voltaren",
    "فولترين": "Voltaren",
    "voltaren": "Voltaren",

    # Antinal
    "انتينال": "Antinal",
    "أنتينال": "Antinal",
    "antinal": "Antinal",

    # Motilium
    "موتيليوم": "Motilium",
    "موتيليم": "Motilium",
    "motilium": "Motilium",

    # Nexium / Omeprazole
    "نيكسيوم": "Nexium",
    "nexium": "Nexium",

    "اوميبرازول": "Omeprazole",
    "أوميبرازول": "Omeprazole",
    "omeprazole": "Omeprazole",

    # Telfast / Loratadine / Aerius
    "تلفاست": "Telfast",
    "telfast": "Telfast",

    "لوراتادين": "Loratadine",
    "loratadine": "Loratadine",

    "ايريوس": "Aerius",
    "أيريوس": "Aerius",
    "aerius": "Aerius",

    # Ventolin / Salbutamol
    "فنتولين": "Ventolin",
    "ventolin": "Ventolin",

    "سالبوتامول": "Salbutamol",
    "salbutamol": "Salbutamol",

    # Prednisone
    "بريدنيزون": "Prednisone",
    "prednisone": "Prednisone",

    # Decancit
    "ديكانست": "Decancit",
    "ديكانسيت": "Decancit",
    "decancit": "Decancit",

    # Strepsils
    "ستربسلز": "Strepsils",
    "ستريبسلز": "Strepsils",
    "strepsils": "Strepsils"
}


def normalize_word(word: str) -> str:
    """
    Normalize Arabic/English drug words before fuzzy matching.
    """
    word = str(word).strip().lower()

    # Remove punctuation
    word = re.sub(r"[؟?!.,،:;()\[\]{}\"']", "", word)

    # Arabic normalization
    word = word.replace("أ", "ا")
    word = word.replace("إ", "ا")
    word = word.replace("آ", "ا")
    word = word.replace("ى", "ي")
    word = word.replace("ة", "ه")
    word = word.replace("ـ", "")

    return word


# Build normalized alias dictionary
normalized_aliases = {
    normalize_word(alias): canonical
    for alias, canonical in arabic_aliases.items()
}


def correct_drug_name(word, threshold=80):
    """
    Correct a drug name extracted by NER.
    BERT extracts the entity, and this function only normalizes/corrects it.
    """
    if not word:
        return None

    clean_word = normalize_word(word)

    # Exact alias match
    if clean_word in normalized_aliases:
        return normalized_aliases[clean_word]

    # Fuzzy match against aliases
    alias_match = process.extractOne(clean_word, list(normalized_aliases.keys()))
    if alias_match and alias_match[1] >= threshold:
        return normalized_aliases[alias_match[0]]

    # Fuzzy match against canonical English names
    canonical_match = process.extractOne(clean_word, drug_dictionary)
    if canonical_match and canonical_match[1] >= threshold:
        return canonical_match[0]

    return None


def extract_drugs_from_text(text, threshold=85):
    """
    Backup only: tries to find drugs directly from text.
    Main extraction should still come from BERT NER.
    """
    found_drugs = []

    words = text.replace("؟", "").replace("?", "").split()

    for word in words:
        corrected = correct_drug_name(word, threshold=threshold)
        if corrected and corrected not in found_drugs:
            found_drugs.append(corrected)

    return found_drugs


def normalize_drug_names_in_text(text):
    """
    Replace known Arabic/typo aliases in text with canonical names.
    Useful for normalized_query.
    """
    normalized_text = text

    # Replace longer aliases first
    aliases_sorted = sorted(arabic_aliases.keys(), key=len, reverse=True)

    for alias in aliases_sorted:
        canonical = arabic_aliases[alias]
        normalized_text = normalized_text.replace(alias, canonical)

    return normalized_text


if __name__ == "__main__":
    tests = [
        "ينفع اخد كونجستول مع بنادوول؟",
        "جرعة اموكسسلين كام؟",
        "باراسيتامول لطفل وزنه 20 كيلو",
        "هل زرتك يسبب نعاس؟",
        "كتفلام ينفع مع بروفن؟",
        "فولتارين مضر للمعدة؟",
        "اوجمنتين جرعته كام؟",
        "فلاجيل قبل الاكل ولا بعده؟"
    ]

    for t in tests:
        print(t, "=>", extract_drugs_from_text(t))



