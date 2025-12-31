"""
Tabular Feature Engineering Script
===================================
Transaction data'dan XGBoost için tabular features oluşturur.

Feature kategorileri:
1. Time features (hour, day, weekend, age)
2. Amount features (log, z-score)
3. Geo features (distance, city_pop)
4. Behavioral/velocity features (rolling counts, recency)
5. Categorical encoding (category, gender, state, merchant)

Çıktı: data/processed/train_features_tabular.parquet
       data/processed/test_features_tabular.parquet
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from sklearn.preprocessing import LabelEncoder
import warnings

warnings.filterwarnings('ignore')

from utils.logger import get_logger

# Paths
BASE_DIR = Path(__file__).parent.parent
DATA_RAW = BASE_DIR / "data" / "raw"
DATA_PROCESSED = BASE_DIR / "data" / "processed"

TRAIN_FILE = DATA_RAW / "fraudTrain.csv"
TEST_FILE = DATA_RAW / "fraudTest.csv"


def haversine_distance(lat1, lon1, lat2, lon2):
    """Haversine formülü ile iki nokta arası mesafe (km)"""
    R = 6371  # Earth radius in km
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c


def create_time_features(df, logger):
    """Zaman tabanlı feature'lar"""
    logger.info("[1] Creating time features...")

    # Parse datetime
    df['trans_datetime'] = pd.to_datetime(df['trans_date_trans_time'])
    df['dob_datetime'] = pd.to_datetime(df['dob'])

    # Time features
    df['hour'] = df['trans_datetime'].dt.hour
    df['dayofweek'] = df['trans_datetime'].dt.dayofweek  # 0=Monday, 6=Sunday
    df['day'] = df['trans_datetime'].dt.day
    df['month'] = df['trans_datetime'].dt.month
    df['year'] = df['trans_datetime'].dt.year

    # Binary features
    df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)
    df['is_night'] = ((df['hour'] >= 22) | (df['hour'] <= 5)).astype(int)

    # Age (years)
    df['age'] = (df['trans_datetime'] - df['dob_datetime']).dt.days / 365.25

    logger.info(f"  ✓ Created {8} time features")

    return df


def create_amount_features(df, logger):
    """Tutar tabanlı feature'lar"""
    logger.info("[2] Creating amount features...")

    # Log transform
    df['amt_log'] = np.log1p(df['amt'])

    # Z-score (global)
    df['amt_zscore'] = (df['amt'] - df['amt'].mean()) / df['amt'].std()

    # Amount buckets
    df['amt_bucket'] = pd.cut(df['amt'],
                              bins=[0, 10, 50, 100, 500, float('inf')],
                              labels=['very_low', 'low', 'medium', 'high', 'very_high'])

    logger.info(f"  ✓ Created {3} amount features")

    return df


def create_geo_features(df, logger):
    """Coğrafi feature'lar"""
    logger.info("[3] Creating geo features...")

    # Haversine distance
    df['distance_km'] = haversine_distance(
        df['lat'].values, df['long'].values,
        df['merch_lat'].values, df['merch_long'].values
    )

    # Log-transform distance
    df['distance_log'] = np.log1p(df['distance_km'])

    # City population (log)
    df['city_pop_log'] = np.log1p(df['city_pop'])

    # Distance buckets
    df['distance_bucket'] = pd.cut(df['distance_km'],
                                   bins=[0, 1, 10, 50, 200, float('inf')],
                                   labels=['same_city', 'nearby', 'regional', 'far', 'very_far'])

    logger.info(f"  ✓ Created {4} geo features")

    return df


def create_behavioral_features(df, logger, is_train=True):
    """
    Davranışsal/velocity feature'lar (card bazında) - OPTIMIZE EDİLMİŞ

    Vectorized operations kullanarak çok hızlı çalışır (10-15 saniye)
    Fraud detection gücü yüksek tutulmuştur
    """
    logger.info("[4] Creating behavioral/velocity features...")

    # Temporal sıralama
    df = df.sort_values(['cc_num', 'trans_datetime']).reset_index(drop=True)

    # Card bazında grupla
    card_groups = df.groupby('cc_num')

    # 1. Transaction count per card (total)
    df['card_tx_count'] = card_groups['cc_num'].transform('count')

    # 2. Recency: Son işlemden bu yana geçen süre (saniye)
    df['time_since_last_tx'] = card_groups['trans_datetime'].diff().dt.total_seconds()
    df['time_since_last_tx'] = df['time_since_last_tx'].fillna(999999)  # İlk işlem için büyük değer

    # 3. Average amount per card
    df['card_avg_amt'] = card_groups['amt'].transform('mean')

    # 4. Deviation from card's average
    df['amt_deviation_from_card_avg'] = df['amt'] - df['card_avg_amt']

    # 5. Normalize time since last tx (log)
    df['time_since_last_tx_log'] = np.log1p(df['time_since_last_tx'])

    # 6. Transaction sequence number (velocity proxy)
    df['card_tx_sequence'] = card_groups.cumcount() + 1

    # 7. Recent activity indicator (son 10 işlem içinde mi?)
    df['is_recent_active'] = (df['card_tx_sequence'] > df['card_tx_count'] - 10).astype(int)

    # 8. Rolling features (optimize edilmiş versiyon)
    # Her kartın son 3 işleminin ortalama tutarı
    df['amt_rolling_mean_3'] = card_groups['amt'].transform(lambda x: x.rolling(3, min_periods=1).mean())

    # Son 3 işlemin standart sapması (volatility)
    df['amt_rolling_std_3'] = card_groups['amt'].transform(lambda x: x.rolling(3, min_periods=1).std().fillna(0))

    logger.info(f"  ✓ Created {9} behavioral features (optimized, vectorized)")

    return df


def encode_categorical_features(df, logger, encoders=None, is_train=True):
    """
    Kategorik feature encoding

    Args:
        encoders: Train'den gelen encoder'lar (test için)
        is_train: Train/test ayırımı
    """
    logger.info("[5] Encoding categorical features...")

    categorical_cols = ['category', 'gender', 'state', 'amt_bucket', 'distance_bucket']

    if is_train:
        encoders = {}
        for col in categorical_cols:
            le = LabelEncoder()
            df[f'{col}_encoded'] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
        logger.info(f"  ✓ Encoded {len(categorical_cols)} categorical columns (TRAIN)")
    else:
        for col in categorical_cols:
            le = encoders[col]
            # Test'te train'de olmayan kategoriler olabilir, onları -1 yap
            df[f'{col}_encoded'] = df[col].astype(str).apply(
                lambda x: le.transform([x])[0] if x in le.classes_ else -1
            )
        logger.info(f"  ✓ Encoded {len(categorical_cols)} categorical columns (TEST)")

    # Merchant frequency encoding (high cardinality için)
    if is_train:
        merchant_counts = df['merchant'].value_counts()
        merchant_freq_map = (merchant_counts / len(df)).to_dict()
        encoders['merchant_freq'] = merchant_freq_map
    else:
        merchant_freq_map = encoders['merchant_freq']

    df['merchant_freq'] = df['merchant'].map(merchant_freq_map).fillna(0)

    logger.info(f"  ✓ Added merchant frequency encoding")

    return df, encoders


def select_final_features(df, logger):
    """Final feature set seçimi"""
    logger.info("[6] Selecting final feature set...")

    # DROP kolonlar
    drop_cols = [
        'Unnamed: 0', 'trans_num', 'first', 'last', 'street',
        'trans_date_trans_time', 'trans_datetime', 'dob', 'dob_datetime',
        'unix_time', 'cc_num', 'merchant', 'city', 'zip',
        'category', 'gender', 'state', 'amt_bucket', 'distance_bucket',
        'job'  # High cardinality, şimdilik drop
    ]

    # Feature columns
    feature_cols = [col for col in df.columns if col not in drop_cols + ['is_fraud']]

    X = df[feature_cols].copy()
    y = df['is_fraud'].copy()

    logger.info(f"  ✓ Final feature count: {len(feature_cols)}")
    logger.info(f"  ✓ Feature list: {feature_cols}")

    return X, y, feature_cols


def build_features(df, logger, encoders=None, is_train=True):
    """Tüm feature engineering pipeline"""

    logger.info(f"\n{'=' * 60}")
    logger.info(f"Building {'TRAIN' if is_train else 'TEST'} features...")
    logger.info(f"{'=' * 60}")

    # 1. Time features
    df = create_time_features(df, logger)

    # 2. Amount features
    df = create_amount_features(df, logger)

    # 3. Geo features
    df = create_geo_features(df, logger)

    # 4. Behavioral features
    df = create_behavioral_features(df, logger, is_train)

    # 5. Categorical encoding
    df, encoders = encode_categorical_features(df, logger, encoders, is_train)

    # 6. Final feature selection
    X, y, feature_cols = select_final_features(df, logger)

    logger.info(f"\n✓ Feature engineering complete!")
    logger.info(f"  Shape: {X.shape}")
    logger.info(f"  Fraud ratio: {y.mean():.4f}")

    return X, y, feature_cols, encoders


def main():
    """Ana fonksiyon"""

    # Logger başlat
    logger = get_logger("build_tabular_features")

    try:
        logger.info("TABULAR FEATURE ENGINEERING PIPELINE")

        # Parameters
        logger.log_parameters({
            "train_file": str(TRAIN_FILE),
            "test_file": str(TEST_FILE),
            "output_dir": str(DATA_PROCESSED)
        })

        # 1. Load data
        logger.info("\n[STEP 1] Loading datasets...")
        df_train = pd.read_csv(TRAIN_FILE)
        df_test = pd.read_csv(TEST_FILE)
        logger.info(f"✓ Train: {len(df_train):,} rows")
        logger.info(f"✓ Test: {len(df_test):,} rows")

        # 2. Build train features
        logger.info("\n[STEP 2] Building TRAIN features...")
        X_train, y_train, feature_cols, encoders = build_features(
            df_train, logger, is_train=True
        )

        # 3. Build test features
        logger.info("\n[STEP 3] Building TEST features...")
        X_test, y_test, _, _ = build_features(
            df_test, logger, encoders=encoders, is_train=False
        )

        # 4. Save features
        logger.info("\n[STEP 4] Saving features...")

        train_features_path = DATA_PROCESSED / "train_features_tabular.parquet"
        train_labels_path = DATA_PROCESSED / "train_labels.parquet"
        test_features_path = DATA_PROCESSED / "test_features_tabular.parquet"
        test_labels_path = DATA_PROCESSED / "test_labels.parquet"
        feature_cols_path = DATA_PROCESSED / "feature_columns.txt"

        X_train.to_parquet(train_features_path, index=False)
        y_train.to_frame('is_fraud').to_parquet(train_labels_path, index=False)
        X_test.to_parquet(test_features_path, index=False)
        y_test.to_frame('is_fraud').to_parquet(test_labels_path, index=False)

        # Save feature column names
        with open(feature_cols_path, 'w') as f:
            f.write('\n'.join(feature_cols))

        # Log artifacts
        logger.log_artifact(train_features_path, "Training tabular features")
        logger.log_artifact(train_labels_path, "Training labels")
        logger.log_artifact(test_features_path, "Test tabular features")
        logger.log_artifact(test_labels_path, "Test labels")
        logger.log_artifact(feature_cols_path, "Feature column names")

        # Log metrics
        logger.log_metrics({
            "train_samples": len(X_train),
            "train_features": X_train.shape[1],
            "train_fraud_ratio": y_train.mean(),
            "test_samples": len(X_test),
            "test_features": X_test.shape[1],
            "test_fraud_ratio": y_test.mean(),
            "feature_count": len(feature_cols)
        })

        logger.info("TABULAR FEATURE ENGINEERING COMPLETE!")
        logger.finalize(status="success")

    except Exception as e:
        logger.error(f"Error occurred: {str(e)}")
        logger.finalize(status="failed")
        raise


if __name__ == "__main__":
    main()
