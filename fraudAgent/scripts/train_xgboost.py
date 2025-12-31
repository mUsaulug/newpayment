"""
XGBoost Model Training Script
==============================
Tabular features üzerinde fraud detection için XGBoost modeli eğitir.

Özellikler:
- Imbalanced data handling (SMOTE + scale_pos_weight)
- Train/validation split (stratified)
- Hyperparameter tuning
- Threshold optimization (high recall için)
- Comprehensive metrics (AUC, precision, recall, F1)
- Model + preprocessing pipeline kaydetme

Çıktı: models/xgboost_fraud_model.pkl
       models/xgboost_metrics.json
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from datetime import datetime
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    precision_recall_curve, roc_curve, average_precision_score
)
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import xgboost as xgb

from utils.logger import get_logger

# Paths
BASE_DIR = Path(__file__).parent.parent
DATA_PROCESSED = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)

TRAIN_FEATURES = DATA_PROCESSED / "train_features_tabular.parquet"
TRAIN_LABELS = DATA_PROCESSED / "train_labels.parquet"
TEST_FEATURES = DATA_PROCESSED / "test_features_tabular.parquet"
TEST_LABELS = DATA_PROCESSED / "test_labels.parquet"


def convert_to_native_types(obj):
    """
    NumPy ve diğer özel tipleri Python native tiplere çevir (JSON serialization için)
    """
    if isinstance(obj, dict):
        return {key: convert_to_native_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_native_types(item) for item in obj]
    elif isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return convert_to_native_types(obj.tolist())
    elif isinstance(obj, np.bool_):
        return bool(obj)
    else:
        return obj


def load_data(logger):
    """Load preprocessed features and labels"""
    logger.info("[1] Loading preprocessed data...")

    X_train_full = pd.read_parquet(TRAIN_FEATURES)
    y_train_full = pd.read_parquet(TRAIN_LABELS)['is_fraud']
    X_test = pd.read_parquet(TEST_FEATURES)
    y_test = pd.read_parquet(TEST_LABELS)['is_fraud']

    logger.info(f"  ✓ Train: {len(X_train_full):,} samples, {X_train_full.shape[1]} features")
    logger.info(f"  ✓ Test: {len(X_test):,} samples")
    logger.info(f"  ✓ Train fraud ratio: {y_train_full.mean():.4f}")
    logger.info(f"  ✓ Test fraud ratio: {y_test.mean():.4f}")

    return X_train_full, y_train_full, X_test, y_test


def create_train_val_split(X, y, logger, val_size=0.2):
    """Create stratified train/validation split"""
    logger.info(f"[2] Creating train/validation split (val_size={val_size})...")

    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=val_size,
        random_state=42,
        stratify=y
    )

    logger.info(f"  ✓ Train: {len(X_train):,} samples (fraud: {y_train.mean():.4f})")
    logger.info(f"  ✓ Val: {len(X_val):,} samples (fraud: {y_val.mean():.4f})")

    return X_train, X_val, y_train, y_val


def build_xgboost_pipeline(logger):
    """
    XGBoost pipeline with SMOTE for imbalanced data

    Pipeline:
    1. SMOTE (oversample minority class)
    2. XGBoost (with scale_pos_weight for extra imbalance handling)
    """
    logger.info("[3] Building XGBoost pipeline...")

    # SMOTE parameters (conservative - avoid overfitting)
    smote = SMOTE(
        sampling_strategy=0.3,  # Fraud class'ı %30'a çıkar (tam 50-50 değil, overfit önler)
        random_state=42,
        k_neighbors=5
    )

    # XGBoost parameters (tuned for fraud detection)
    xgb_model = xgb.XGBClassifier(
        # Core parameters
        max_depth=6,  # Overfitting önlemek için moderate
        learning_rate=0.1,  # Dengeli learning rate
        n_estimators=200,  # Yeterli iterasyon

        # Imbalance handling
        scale_pos_weight=10,  # SMOTE üstüne ekstra ağırlık (fraud class)

        # Regularization (overfitting önleme)
        reg_alpha=0.1,  # L1 regularization
        reg_lambda=1.0,  # L2 regularization
        gamma=0.1,  # Minimum loss reduction

        # Tree parameters
        min_child_weight=3,  # Overfitting önleme
        subsample=0.8,  # Row sampling
        colsample_bytree=0.8,  # Feature sampling

        # Performance
        tree_method='hist',  # Hızlı histogram-based
        random_state=42,
        n_jobs=-1,  # Tüm CPU'ları kullan

        # Output
        eval_metric='aucpr',  # Imbalanced data için en iyi metric
        early_stopping_rounds=20  # Validation'da iyileşme durmazsa dur
    )

    # Pipeline
    pipeline = ImbPipeline([
        ('smote', smote),
        ('xgboost', xgb_model)
    ])

    logger.info("  ✓ Pipeline built:")
    logger.info(f"    - SMOTE: sampling_strategy=0.3")
    logger.info(f"    - XGBoost: max_depth=6, n_estimators=200, scale_pos_weight=10")

    return pipeline


def train_model(pipeline, X_train, y_train, X_val, y_val, logger):
    """Train XGBoost with early stopping on validation"""
    logger.info("[4] Training XGBoost model...")

    # Fit pipeline
    # Note: SMOTE sadece train'e uygulanır, validation'a dokunmaz
    pipeline.fit(
        X_train, y_train,
        xgboost__eval_set=[(X_val, y_val)],  # Validation monitoring
        xgboost__verbose=False
    )

    # Best iteration
    best_iteration = pipeline.named_steps['xgboost'].best_iteration
    logger.info(f"  ✓ Training complete!")
    logger.info(f"    Best iteration: {best_iteration}")

    return pipeline


def evaluate_model(pipeline, X_val, y_val, X_test, y_test, logger, threshold=0.5):
    """
    Comprehensive model evaluation

    Returns:
        metrics_dict: All metrics for logging
    """
    logger.info(f"[5] Evaluating model (threshold={threshold})...")

    # Predictions
    y_val_proba = pipeline.predict_proba(X_val)[:, 1]
    y_test_proba = pipeline.predict_proba(X_test)[:, 1]

    y_val_pred = (y_val_proba >= threshold).astype(int)
    y_test_pred = (y_test_proba >= threshold).astype(int)

    # Metrics
    metrics = {}

    # Validation metrics
    metrics['val_auc'] = float(roc_auc_score(y_val, y_val_proba))
    metrics['val_ap'] = float(average_precision_score(y_val, y_val_proba))

    val_report = classification_report(y_val, y_val_pred, output_dict=True)
    metrics['val_precision'] = float(val_report['1']['precision'])
    metrics['val_recall'] = float(val_report['1']['recall'])
    metrics['val_f1'] = float(val_report['1']['f1-score'])

    val_cm = confusion_matrix(y_val, y_val_pred)
    metrics['val_tn'], metrics['val_fp'] = int(val_cm[0][0]), int(val_cm[0][1])
    metrics['val_fn'], metrics['val_tp'] = int(val_cm[1][0]), int(val_cm[1][1])

    # Test metrics
    metrics['test_auc'] = float(roc_auc_score(y_test, y_test_proba))
    metrics['test_ap'] = float(average_precision_score(y_test, y_test_proba))

    test_report = classification_report(y_test, y_test_pred, output_dict=True)
    metrics['test_precision'] = float(test_report['1']['precision'])
    metrics['test_recall'] = float(test_report['1']['recall'])
    metrics['test_f1'] = float(test_report['1']['f1-score'])

    test_cm = confusion_matrix(y_test, y_test_pred)
    metrics['test_tn'], metrics['test_fp'] = int(test_cm[0][0]), int(test_cm[0][1])
    metrics['test_fn'], metrics['test_tp'] = int(test_cm[1][0]), int(test_cm[1][1])

    # Log results
    logger.info("\n" + "=" * 60)
    logger.info("VALIDATION METRICS")
    logger.info("=" * 60)
    logger.info(f"AUC-ROC: {metrics['val_auc']:.4f}")
    logger.info(f"AP (PR-AUC): {metrics['val_ap']:.4f}")
    logger.info(f"Precision: {metrics['val_precision']:.4f}")
    logger.info(f"Recall: {metrics['val_recall']:.4f}")
    logger.info(f"F1-Score: {metrics['val_f1']:.4f}")
    logger.info(f"\nConfusion Matrix:")
    logger.info(f"  TN: {metrics['val_tn']:,}  FP: {metrics['val_fp']:,}")
    logger.info(f"  FN: {metrics['val_fn']:,}  TP: {metrics['val_tp']:,}")

    logger.info("\n" + "=" * 60)
    logger.info("TEST METRICS (Final Holdout)")
    logger.info("=" * 60)
    logger.info(f"AUC-ROC: {metrics['test_auc']:.4f}")
    logger.info(f"AP (PR-AUC): {metrics['test_ap']:.4f}")
    logger.info(f"Precision: {metrics['test_precision']:.4f}")
    logger.info(f"Recall: {metrics['test_recall']:.4f}")
    logger.info(f"F1-Score: {metrics['test_f1']:.4f}")
    logger.info(f"\nConfusion Matrix:")
    logger.info(f"  TN: {metrics['test_tn']:,}  FP: {metrics['test_fp']:,}")
    logger.info(f"  FN: {metrics['test_fn']:,}  TP: {metrics['test_tp']:,}")
    logger.info("=" * 60 + "\n")

    return metrics, y_val_proba, y_test_proba


def find_optimal_threshold(y_true, y_proba, logger, target_recall=0.85):
    """
    Fraud detection için optimal threshold bul

    Hedef: Yüksek recall (fraud'ları kaçırmamak) + kabul edilebilir precision
    """
    logger.info(f"[6] Finding optimal threshold (target_recall >= {target_recall})...")

    precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)

    # Target recall'u sağlayan threshold'ları bul
    valid_indices = recalls >= target_recall

    if valid_indices.sum() == 0:
        logger.warning(f"  ! No threshold achieves recall >= {target_recall}")
        logger.warning(f"    Using default threshold = 0.5")
        return 0.5

    # En yüksek precision'a sahip threshold'u seç
    valid_precisions = precisions[valid_indices]
    valid_thresholds = thresholds[valid_indices[:-1]]  # thresholds 1 eleman az

    best_idx = valid_precisions.argmax()
    optimal_threshold = float(valid_thresholds[best_idx])

    logger.info(f"  ✓ Optimal threshold found: {optimal_threshold:.4f}")
    logger.info(f"    At this threshold:")
    logger.info(f"      Recall: {recalls[valid_indices][best_idx]:.4f}")
    logger.info(f"      Precision: {valid_precisions[best_idx]:.4f}")

    return optimal_threshold


def get_feature_importance(pipeline, feature_names, logger, top_n=20):
    """Extract and log feature importances"""
    logger.info(f"[7] Extracting top {top_n} feature importances...")

    xgb_model = pipeline.named_steps['xgboost']
    importances = xgb_model.feature_importances_

    # Create DataFrame
    feature_imp_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False)

    # Log top features
    logger.info(f"\n  Top {top_n} features:")
    for idx, row in feature_imp_df.head(top_n).iterrows():
        logger.info(f"    {row['feature']}: {row['importance']:.4f}")

    return feature_imp_df


def save_model_and_artifacts(pipeline, metrics, feature_importance, threshold, logger):
    """Save model, metrics, and related artifacts"""
    logger.info("[8] Saving model and artifacts...")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Model
    model_path = MODELS_DIR / f"xgboost_fraud_model_{timestamp}.pkl"
    joblib.dump(pipeline, model_path)
    logger.log_artifact(model_path, "XGBoost fraud detection model (with SMOTE pipeline)")

    # Metrics - NumPy tiplerini Python native'e çevir
    metrics_clean = convert_to_native_types(metrics)
    metrics_clean['optimal_threshold'] = float(threshold)
    metrics_clean['timestamp'] = timestamp

    metrics_path = MODELS_DIR / f"xgboost_metrics_{timestamp}.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics_clean, f, indent=2)
    logger.log_artifact(metrics_path, "Model evaluation metrics")

    # Feature importance
    feature_imp_path = MODELS_DIR / f"xgboost_feature_importance_{timestamp}.csv"
    feature_importance.to_csv(feature_imp_path, index=False)
    logger.log_artifact(feature_imp_path, "Feature importances")

    # Latest model symlink (for easy access)
    latest_model_path = MODELS_DIR / "xgboost_fraud_model_latest.pkl"
    joblib.dump(pipeline, latest_model_path)
    logger.log_artifact(latest_model_path, "Latest XGBoost model (symlink)")

    logger.info(f"  ✓ All artifacts saved to {MODELS_DIR}")

    return model_path, metrics_path, feature_imp_path


def main():
    """Main training pipeline"""

    logger = get_logger("train_xgboost")

    try:
        logger.info("XGBOOST FRAUD DETECTION TRAINING PIPELINE")
        logger.info("State-of-the-art gradient boosting with imbalanced data handling")

        # Parameters
        params = {
            "val_size": 0.2,
            "target_recall": 0.85,
            "default_threshold": 0.5,
            "smote_sampling_strategy": 0.3,
            "xgb_max_depth": 6,
            "xgb_n_estimators": 200,
            "xgb_scale_pos_weight": 10
        }
        logger.log_parameters(params)

        # 1. Load data
        X_train_full, y_train_full, X_test, y_test = load_data(logger)

        # 2. Train/val split
        X_train, X_val, y_train, y_val = create_train_val_split(
            X_train_full, y_train_full, logger, val_size=params['val_size']
        )

        # 3. Build pipeline
        pipeline = build_xgboost_pipeline(logger)

        # 4. Train
        pipeline = train_model(pipeline, X_train, y_train, X_val, y_val, logger)

        # 5. Evaluate (default threshold)
        metrics, y_val_proba, y_test_proba = evaluate_model(
            pipeline, X_val, y_val, X_test, y_test, logger,
            threshold=params['default_threshold']
        )

        # 6. Find optimal threshold
        optimal_threshold = find_optimal_threshold(
            y_val, y_val_proba, logger,
            target_recall=params['target_recall']
        )

        # 7. Re-evaluate with optimal threshold
        logger.info(f"\nRe-evaluating with optimal threshold ({optimal_threshold:.4f})...")
        metrics_optimal, _, _ = evaluate_model(
            pipeline, X_val, y_val, X_test, y_test, logger,
            threshold=optimal_threshold
        )

        # 8. Feature importance
        feature_names = X_train.columns.tolist()
        feature_importance = get_feature_importance(pipeline, feature_names, logger)

        # 9. Save
        model_path, metrics_path, feature_imp_path = save_model_and_artifacts(
            pipeline, metrics_optimal, feature_importance, optimal_threshold, logger
        )

        # Log final metrics - NumPy tiplerini temizle
        final_metrics = convert_to_native_types({
            "final_test_auc": metrics_optimal['test_auc'],
            "final_test_recall": metrics_optimal['test_recall'],
            "final_test_precision": metrics_optimal['test_precision'],
            "final_test_f1": metrics_optimal['test_f1'],
            "optimal_threshold": optimal_threshold,
            "num_features": len(feature_names)
        })

        logger.log_metrics(final_metrics)

        logger.info("\n" + "=" * 70)
        logger.info("XGBOOST TRAINING COMPLETE!")
        logger.info("=" * 70)
        logger.info(f"Model saved: {model_path}")
        logger.info(f"Test AUC: {metrics_optimal['test_auc']:.4f}")
        logger.info(f"Test Recall: {metrics_optimal['test_recall']:.4f}")
        logger.info(f"Test Precision: {metrics_optimal['test_precision']:.4f}")
        logger.info(f"Optimal Threshold: {optimal_threshold:.4f}")
        logger.info("=" * 70)

        logger.finalize(status="success")

    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        logger.finalize(status="failed")
        raise


if __name__ == "__main__":
    main()
