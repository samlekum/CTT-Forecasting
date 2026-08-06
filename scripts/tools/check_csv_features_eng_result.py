# ./scripts/tools/check_csv_features_eng_result.py
# Memeriksa hasil feature engineering dengan menampilkan struktur dataset, statistik fitur, serta validitas data yang dihasilkan.

import pandas as pd

df = pd.read_csv("./dataset/features_60min.csv")  # pake yang paling kecil biar cepat
print(df[["base_time", "target_time", "pixel_row", "pixel_col", "lat", "lon"]].head(10))
print()
print(df.filter(like="tbb_13").describe())
print()
print(df["target_tbb_13"].describe())
print()
print(f"Ada NaN? {df.isna().sum().sum()}")