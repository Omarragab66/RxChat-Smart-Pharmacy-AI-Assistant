import pandas as pd
from sklearn.model_selection import train_test_split

# Load full dataset
df = pd.read_csv("data/medical_arabic_dataset_diverse_plus_hard_fixed.csv")
# Remove exact duplicate rows
df = df.drop_duplicates(subset=["raw_text", "intent"]).reset_index(drop=True)

print("Total after removing duplicates:", df.shape)

# Train/Test split
train_df, test_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42,
    stratify=df["intent"]
)

# Check leakage
train_texts = set(train_df["raw_text"])
test_texts = set(test_df["raw_text"])

overlap = train_texts.intersection(test_texts)

print("Train size:", train_df.shape)
print("Test size:", test_df.shape)
print("Overlapping texts:", len(overlap))

if len(overlap) > 0:
    print("WARNING: Data leakage found!")
    print(list(overlap)[:10])
else:
    print("No data leakage detected.")

# Save clean splits
train_df.to_csv("data/train_no_leakage.csv", index=False, encoding="utf-8-sig")
test_df.to_csv("data/test_no_leakage.csv", index=False, encoding="utf-8-sig")

print("Saved:")
print("data/train_no_leakage.csv")
print("data/test_no_leakage.csv")



