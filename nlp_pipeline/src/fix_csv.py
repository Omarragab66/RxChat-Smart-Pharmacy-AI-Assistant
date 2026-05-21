import pandas as pd

input_path = "data/medical_arabic_dataset_diverse_plus_hard.csv"
output_path = "data/medical_arabic_dataset_diverse_plus_hard_fixed.csv"

df = pd.read_csv(input_path, header=None)

df = df[0].str.split(",", expand=True)

df = df.iloc[1:].reset_index(drop=True)

df.columns = [
    "raw_text",
    "normalized_text",
    "intent",
    "drug_entities"
]

df.to_csv(output_path, index=False, encoding="utf-8-sig")

print(df.head())
print(df.columns)
print(df["intent"].value_counts())



