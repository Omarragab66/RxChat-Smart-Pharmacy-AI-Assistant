import pandas as pd
from sklearn.model_selection import train_test_split

# Load extended dataset
df = pd.read_csv("data/medical_arabic_dataset_extended.csv")

# Split train / temp (validation+test)
train_df, temp_df = train_test_split(
    df,
    test_size=0.3,
    random_state=42,
    stratify=df['intent']  # يحافظ على نسبة كل intent
)

# Split temp to validation and test equally
val_df, test_df = train_test_split(
    temp_df,
    test_size=0.5,
    random_state=42,
    stratify=temp_df['intent']
)

# Save splits
train_df.to_csv("data/train.csv", index=False, encoding="utf-8-sig")
val_df.to_csv("data/val.csv", index=False, encoding="utf-8-sig")
test_df.to_csv("data/test.csv", index=False, encoding="utf-8-sig")

print("Train/Validation/Test split done!")
print("Train:", train_df.shape)
print("Validation:", val_df.shape)
print("Test:", test_df.shape)



