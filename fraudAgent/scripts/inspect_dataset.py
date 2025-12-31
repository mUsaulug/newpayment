"""
Fraud Dataset Inspection Script
================================
Fraud detection dataset'ini (fraudTrain.csv & fraudTest.csv) analiz eder.
Kolonları, fraud dağılımını, graph yapısı için gerekli bilgileri çıkarır.

Çıktı: Console + data/inspection_report.txt
"""

import pandas as pd
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).parent.parent
DATA_RAW = BASE_DIR / "data" / "raw"
DATA_DIR = BASE_DIR / "data"

TRAIN_FILE = DATA_RAW / "fraudTrain.csv"
TEST_FILE = DATA_RAW / "fraudTest.csv"
REPORT_FILE = DATA_DIR / "inspection_report.txt"


def inspect_dataset():
    """Ana inspection fonksiyonu"""

    print("=" * 60)
    print("FRAUD DATASET INSPECTION")
    print("=" * 60)

    # 1. Load datasets
    print("\n[1] Loading datasets...")
    df_train = pd.read_csv(TRAIN_FILE)
    df_test = pd.read_csv(TEST_FILE)
    print(f"✓ Train loaded: {len(df_train):,} rows")
    print(f"✓ Test loaded: {len(df_test):,} rows")

    # 2. Basic info
    print("\n[2] Basic information:")
    print(f"Train shape: {df_train.shape}")
    print(f"Test shape: {df_test.shape}")
    print(f"\nColumns ({len(df_train.columns)}):")
    print(df_train.columns.tolist())

    # 3. Data types
    print("\n[3] Data types:")
    print(df_train.dtypes)

    # 4. Fraud distribution
    print("\n[4] Fraud distribution:")
    train_fraud_dist = df_train["is_fraud"].value_counts(normalize=True)
    test_fraud_dist = df_test["is_fraud"].value_counts(normalize=True)

    print(f"\nTrain:")
    print(f"  Not fraud: {train_fraud_dist[0] * 100:.2f}%")
    print(f"  Fraud:     {train_fraud_dist[1] * 100:.2f}%")
    print(f"  Fraud count: {df_train['is_fraud'].sum():,}")

    print(f"\nTest:")
    print(f"  Not fraud: {test_fraud_dist[0] * 100:.2f}%")
    print(f"  Fraud:     {test_fraud_dist[1] * 100:.2f}%")
    print(f"  Fraud count: {df_test['is_fraud'].sum():,}")

    # 5. Graph construction info
    print("\n[5] Graph construction information:")
    train_unique_cards = df_train["cc_num"].nunique()
    train_unique_merchants = df_train["merchant"].nunique()
    test_unique_cards = df_test["cc_num"].nunique()
    test_unique_merchants = df_test["merchant"].nunique()

    print(f"\nTrain:")
    print(f"  Unique cards (cc_num): {train_unique_cards:,}")
    print(f"  Unique merchants: {train_unique_merchants:,}")
    print(f"  Total transactions (edges): {len(df_train):,}")

    print(f"\nTest:")
    print(f"  Unique cards (cc_num): {test_unique_cards:,}")
    print(f"  Unique merchants: {test_unique_merchants:,}")
    print(f"  Total transactions (edges): {len(df_test):,}")

    # 6. Kategorik kolonların unique değer sayıları
    print("\n[6] Categorical column cardinality:")
    categorical_cols = ["merchant", "category", "gender", "job", "city", "state"]
    for col in categorical_cols:
        print(f"  {col}: {df_train[col].nunique():,} unique values")

    # 7. Numeric kolonların özeti
    print("\n[7] Numeric columns summary:")
    numeric_cols = ["amt", "lat", "long", "city_pop", "merch_lat", "merch_long"]
    print(df_train[numeric_cols].describe())

    # 8. Zaman aralığı
    print("\n[8] Date range:")
    df_train["trans_datetime"] = pd.to_datetime(df_train["trans_date_trans_time"])
    df_test["trans_datetime"] = pd.to_datetime(df_test["trans_date_trans_time"])

    print(f"\nTrain: {df_train['trans_datetime'].min()} to {df_train['trans_datetime'].max()}")
    print(f"Test:  {df_test['trans_datetime'].min()} to {df_test['trans_datetime'].max()}")

    # 9. Missing values check
    print("\n[9] Missing values:")
    train_nulls = df_train.isnull().sum()
    if train_nulls.sum() == 0:
        print("  ✓ No missing values in train")
    else:
        print("  Missing values found:")
        print(train_nulls[train_nulls > 0])

    test_nulls = df_test.isnull().sum()
    if test_nulls.sum() == 0:
        print("  ✓ No missing values in test")
    else:
        print("  Missing values found:")
        print(test_nulls[test_nulls > 0])

    # 10. Write report file
    print("\n[10] Writing inspection report...")
    write_report(
        df_train, df_test,
        train_fraud_dist, test_fraud_dist,
        train_unique_cards, train_unique_merchants,
        test_unique_cards, test_unique_merchants
    )
    print(f"✓ Report saved to: {REPORT_FILE}")

    print("\n" + "=" * 60)
    print("INSPECTION COMPLETE")
    print("=" * 60)


def write_report(df_train, df_test, train_fraud_dist, test_fraud_dist,
                 train_cards, train_merchants, test_cards, test_merchants):
    """Inspection report'u dosyaya yaz"""

    with open(REPORT_FILE, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("FRAUD DATASET INSPECTION REPORT\n")
        f.write("=" * 70 + "\n\n")

        f.write("1. DATASET SIZE\n")
        f.write("-" * 70 + "\n")
        f.write(f"Train: {len(df_train):,} rows, {len(df_train.columns)} columns\n")
        f.write(f"Test:  {len(df_test):,} rows, {len(df_test.columns)} columns\n\n")

        f.write("2. FRAUD DISTRIBUTION\n")
        f.write("-" * 70 + "\n")
        f.write(f"Train:\n")
        f.write(
            f"  Not fraud: {train_fraud_dist[0] * 100:.2f}% ({(1 - train_fraud_dist[1]) * len(df_train):,.0f} samples)\n")
        f.write(
            f"  Fraud:     {train_fraud_dist[1] * 100:.2f}% ({train_fraud_dist[1] * len(df_train):,.0f} samples)\n\n")
        f.write(f"Test:\n")
        f.write(
            f"  Not fraud: {test_fraud_dist[0] * 100:.2f}% ({(1 - test_fraud_dist[1]) * len(df_test):,.0f} samples)\n")
        f.write(f"  Fraud:     {test_fraud_dist[1] * 100:.2f}% ({test_fraud_dist[1] * len(df_test):,.0f} samples)\n\n")

        f.write("3. GRAPH STRUCTURE INFO\n")
        f.write("-" * 70 + "\n")
        f.write(f"Train:\n")
        f.write(f"  Card nodes (cc_num):   {train_cards:,}\n")
        f.write(f"  Merchant nodes:        {train_merchants:,}\n")
        f.write(f"  Transaction edges:     {len(df_train):,}\n\n")
        f.write(f"Test:\n")
        f.write(f"  Card nodes (cc_num):   {test_cards:,}\n")
        f.write(f"  Merchant nodes:        {test_merchants:,}\n")
        f.write(f"  Transaction edges:     {len(df_test):,}\n\n")

        f.write("4. COLUMN CLASSIFICATION\n")
        f.write("-" * 70 + "\n")
        f.write("DROP (will not use):\n")
        f.write("  - Unnamed: 0 (index)\n")
        f.write("  - trans_num (unique ID, no pattern)\n")
        f.write("  - first, last, street (PII)\n\n")

        f.write("TIME (feature engineering needed):\n")
        f.write("  - trans_date_trans_time → hour, day, weekend\n")
        f.write("  - dob → age\n")
        f.write("  - unix_time (redundant with trans_date_trans_time)\n\n")

        f.write("CATEGORICAL (encoding needed):\n")
        f.write(f"  - merchant ({df_train['merchant'].nunique():,} unique)\n")
        f.write(f"  - category ({df_train['category'].nunique()} unique)\n")
        f.write(f"  - gender ({df_train['gender'].nunique()} unique)\n")
        f.write(f"  - job ({df_train['job'].nunique():,} unique)\n")
        f.write(f"  - city, state, zip\n\n")

        f.write("NUMERIC (direct or transform):\n")
        f.write("  - amt (log-transform)\n")
        f.write("  - lat, long (customer location)\n")
        f.write("  - merch_lat, merch_long (merchant location)\n")
        f.write("  - city_pop (log-transform)\n\n")

        f.write("LABEL:\n")
        f.write("  - is_fraud (target variable)\n\n")

        f.write("5. NEXT STEPS\n")
        f.write("-" * 70 + "\n")
        f.write("1. Build graph structure (PyTorch Geometric)\n")
        f.write("2. Engineer tabular features (time, geo, behavioral)\n")
        f.write("3. Train GNN model on graph\n")
        f.write("4. Train XGBoost on tabular features\n")
        f.write("5. Train meta-model (Logistic Regression) for ensemble\n\n")

        f.write("=" * 70 + "\n")


if __name__ == "__main__":
    inspect_dataset()
