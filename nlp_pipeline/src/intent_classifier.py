def classify_intent(text):
    text = text.lower()

    dosage_words = ["جرعة", "كام", "اخد قد ايه", "كم مرة"]
    interaction_words = ["مع", "ينفع اخد", "يتاخد مع"]
    side_effect_words = ["اعراض", "دوخة", "نعاس", "بيعمل", "يسبب"]
    contraindication_words = ["حامل", "ضغط", "سكر", "ممنوع", "خطر"]

    if any(word in text for word in dosage_words):
        return "dosage"

    if any(word in text for word in interaction_words):
        return "interaction"

    if any(word in text for word in side_effect_words):
        return "side_effects"

    if any(word in text for word in contraindication_words):
        return "contraindication"

    return "unknown"


# Tests
if __name__ == "__main__":
    print(classify_intent("الجرعة كام للكونجستال؟"))
    print(classify_intent("ينفع اخد بنادول مع كونجستال؟"))
    print(classify_intent("كونجستال بيعمل نعاس؟"))
    print(classify_intent("هل الاسبرين خطر لمرضى الضغط؟"))



