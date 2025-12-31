"""
Comprehensive Model Evaluation Script
======================================
XGBoost ve GNN modellerini test dataset üzerinde karşılaştırır.

Özellikler:
- Detaylı metrics (AUC, Precision, Recall, F1)
- Confusion matrix karşılaştırması
- Error analysis (hangi fraudları kaçırdılar?)
- Disagreement analysis (modeller nerede farklı tahmin etti?)
- Risk distribution
- Comprehensive report + visualizations

Süre: ~10-15 dakika
"""

import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from torch_geometric.loader import NeighborLoader
import pandas as pd
import numpy as np
import json
from pathlib import Path
from datetime import datetime
from sklearn.metrics import (
    roc_auc_score, precision_recall_curve, auc,
    classification_report, confusion_matrix,
    roc_curve, average_precision_score
)
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from utils.logger import get_logger

# Paths
BASE_DIR = Path(__file__).parent.parent
DATA_PROCESSED = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# Model paths
XGBOOST_MODEL = MODELS_DIR / "xgboost_fraud_model_latest.pkl"
GNN_MODEL = MODELS_DIR / "gnn_fraud_model_latest.pt"

# Test data
# Test data
# Test data
TEST_FEATURES = DATA_PROCESSED / "test_features_tabular.parquet"
TEST_LABELS = DATA_PROCESSED / "test_labels.parquet"
TEST_GRAPH = DATA_PROCESSED / "graph_test.pt"



# Thresholds
XGBOOST_THRESHOLD = 0.57
GNN_THRESHOLD = 0.6069  # Original from training


# Device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class GraphSAGE_FraudDetector(nn.Module):
    """GNN Model"""

    def __init__(self, in_channels, hidden_channels=128, num_layers=3, dropout=0.3):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout

        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        self.convs.append(SAGEConv(in_channels, hidden_channels))
        self.batch_norms.append(nn.BatchNorm1d(hidden_channels))

        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
            self.batch_norms.append(nn.BatchNorm1d(hidden_channels))

        self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        self.batch_norms.append(nn.BatchNorm1d(hidden_channels))

        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, 1)
        )

    def forward(self, x, edge_index):
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            x = self.batch_norms[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        out = self.classifier(x)
        return out.squeeze()


def load_models(logger):
    """Load both XGBoost and GNN models"""
    logger.info("[1] Loading models...")

    # XGBoost
    logger.info("  Loading XGBoost model...")
    try:
        import joblib
        xgb_model_data = joblib.load(XGBOOST_MODEL)
    except:
        with open(XGBOOST_MODEL, 'rb') as f:
            xgb_model_data = pickle.load(f)

    if isinstance(xgb_model_data, dict):
        if 'model' in xgb_model_data:
            xgb_model = xgb_model_data['model']
        else:
            xgb_model = list(xgb_model_data.values())[0]
    else:
        xgb_model = xgb_model_data

    logger.info(f"    ✓ XGBoost loaded")

    # GNN
    logger.info("  Loading GNN model...")
    checkpoint = torch.load(GNN_MODEL, map_location=DEVICE, weights_only=False)
    config = checkpoint['model_config']

    gnn_model = GraphSAGE_FraudDetector(
        in_channels=config['in_channels'],
        hidden_channels=config['hidden_channels'],
        num_layers=config['num_layers'],
        dropout=config['dropout']
    )

    gnn_model.load_state_dict(checkpoint['model_state_dict'])
    gnn_model.to(DEVICE)
    gnn_model.eval()

    logger.info(f"    ✓ GNN loaded ({sum(p.numel() for p in gnn_model.parameters()):,} params)")

    return xgb_model, gnn_model


def load_test_data(logger):
    """Load test dataset"""
    logger.info("\n[2] Loading test data...")

    # Tabular data (for XGBoost) - PARQUET format
    test_features_path = DATA_PROCESSED / "test_features_tabular.parquet"
    test_labels_path = DATA_PROCESSED / "test_labels.parquet"

    X_test = pd.read_parquet(test_features_path)
    y_test = pd.read_parquet(test_labels_path).values.ravel()

    logger.info(f"  ✓ Tabular test set: {len(X_test):,} samples")
    logger.info(f"  ✓ Features shape: {X_test.shape}")

    # Graph data (for GNN)
    graph_test = torch.load(TEST_GRAPH, weights_only=False)

    logger.info(f"  ✓ Graph test set: {graph_test.num_nodes:,} nodes, {graph_test.num_edges:,} edges")
    logger.info(f"  ✓ Fraud ratio: {y_test.mean():.4f}")

    return X_test, y_test, graph_test


def predict_xgboost(model, X_test, logger):
    """XGBoost predictions"""
    logger.info("\n[3] XGBoost predictions...")

    proba = model.predict_proba(X_test)[:, 1]
    preds = (proba >= XGBOOST_THRESHOLD).astype(int)

    logger.info(f"  ✓ Generated {len(proba):,} predictions")
    logger.info(f"  ✓ Threshold: {XGBOOST_THRESHOLD:.4f}")

    return proba, preds


@torch.no_grad()
def predict_gnn(model, graph_data, logger):
    """GNN predictions"""
    logger.info("\n[4] GNN predictions...")

    model.eval()

    # Create loader
    test_indices = torch.arange(graph_data.num_nodes)
    loader = NeighborLoader(
        graph_data,
        num_neighbors=[15, 10],
        batch_size=1024,
        input_nodes=test_indices,
        shuffle=False,
        num_workers=0
    )

    all_probs = []

    for batch in tqdm(loader, desc="  Predicting"):
        batch = batch.to(DEVICE)

        out = model(batch.x, batch.edge_index)
        batch_size_actual = batch.batch_size
        out = out[:batch_size_actual]

        probs = torch.sigmoid(out)
        all_probs.append(probs.cpu())

    all_probs = torch.cat(all_probs).numpy()
    preds = (all_probs >= GNN_THRESHOLD).astype(int)

    logger.info(f"  ✓ Generated {len(all_probs):,} predictions")
    logger.info(f"  ✓ Threshold: {GNN_THRESHOLD:.4f}")

    return all_probs, preds


def calculate_metrics(y_true, y_proba, y_pred, model_name):
    """Calculate comprehensive metrics"""

    metrics = {
        'model': model_name,
        'auc_roc': float(roc_auc_score(y_true, y_proba)),
        'auc_pr': float(average_precision_score(y_true, y_proba))
    }

    # Classification report
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)

    metrics['precision'] = float(report['1']['precision']) if '1' in report else 0.0
    metrics['recall'] = float(report['1']['recall']) if '1' in report else 0.0
    metrics['f1'] = float(report['1']['f1-score']) if '1' in report else 0.0

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape == (2, 2):
        metrics['tn'] = int(cm[0][0])
        metrics['fp'] = int(cm[0][1])
        metrics['fn'] = int(cm[1][0])
        metrics['tp'] = int(cm[1][1])

        # Additional metrics
        metrics['specificity'] = metrics['tn'] / (metrics['tn'] + metrics['fp']) if (metrics['tn'] + metrics[
            'fp']) > 0 else 0
        metrics['false_positive_rate'] = metrics['fp'] / (metrics['fp'] + metrics['tn']) if (metrics['fp'] + metrics[
            'tn']) > 0 else 0

    return metrics


def error_analysis(y_true, xgb_pred, gnn_pred, xgb_proba, gnn_proba):
    """Analyze prediction errors"""

    analysis = {
        'total_samples': len(y_true),
        'total_fraud': int(y_true.sum()),
        'total_legit': int((y_true == 0).sum())
    }

    # XGBoost errors
    xgb_fn = ((y_true == 1) & (xgb_pred == 0)).sum()
    xgb_fp = ((y_true == 0) & (xgb_pred == 1)).sum()

    # GNN errors
    gnn_fn = ((y_true == 1) & (gnn_pred == 0)).sum()
    gnn_fp = ((y_true == 0) & (gnn_pred == 1)).sum()

    analysis['xgboost_false_negatives'] = int(xgb_fn)
    analysis['xgboost_false_positives'] = int(xgb_fp)
    analysis['gnn_false_negatives'] = int(gnn_fn)
    analysis['gnn_false_positives'] = int(gnn_fp)

    # Disagreement analysis
    disagreement = (xgb_pred != gnn_pred).sum()
    analysis['disagreement_count'] = int(disagreement)
    analysis['disagreement_rate'] = float(disagreement / len(y_true))

    # Both wrong
    both_wrong = ((xgb_pred != y_true) & (gnn_pred != y_true)).sum()
    analysis['both_wrong'] = int(both_wrong)

    # XGBoost right, GNN wrong
    xgb_right_gnn_wrong = ((xgb_pred == y_true) & (gnn_pred != y_true)).sum()
    analysis['xgb_better'] = int(xgb_right_gnn_wrong)

    # GNN right, XGBoost wrong
    gnn_right_xgb_wrong = ((gnn_pred == y_true) & (xgb_pred != y_true)).sum()
    analysis['gnn_better'] = int(gnn_right_xgb_wrong)

    # Both right
    both_right = ((xgb_pred == y_true) & (gnn_pred == y_true)).sum()
    analysis['both_right'] = int(both_right)

    return analysis


def create_comparison_dataframe(y_true, xgb_proba, xgb_pred, gnn_proba, gnn_pred):
    """Create detailed comparison DataFrame"""

    df = pd.DataFrame({
        'true_label': y_true,
        'xgboost_probability': xgb_proba,
        'xgboost_prediction': xgb_pred,
        'gnn_probability': gnn_proba,
        'gnn_prediction': gnn_pred,
        'xgboost_correct': (xgb_pred == y_true).astype(int),
        'gnn_correct': (gnn_pred == y_true).astype(int),
        'disagreement': (xgb_pred != gnn_pred).astype(int),
        'probability_diff': np.abs(xgb_proba - gnn_proba)
    })

    return df


def plot_confusion_matrices(xgb_metrics, gnn_metrics, output_path):
    """Plot confusion matrices side by side"""

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # XGBoost
    cm_xgb = np.array([
        [xgb_metrics['tn'], xgb_metrics['fp']],
        [xgb_metrics['fn'], xgb_metrics['tp']]
    ])

    sns.heatmap(cm_xgb, annot=True, fmt='d', cmap='Blues', ax=axes[0],
                xticklabels=['Legit', 'Fraud'], yticklabels=['Legit', 'Fraud'])
    axes[0].set_title(f"XGBoost\nF1: {xgb_metrics['f1']:.4f}", fontsize=12, fontweight='bold')
    axes[0].set_ylabel('True Label')
    axes[0].set_xlabel('Predicted Label')

    # GNN
    cm_gnn = np.array([
        [gnn_metrics['tn'], gnn_metrics['fp']],
        [gnn_metrics['fn'], gnn_metrics['tp']]
    ])

    sns.heatmap(cm_gnn, annot=True, fmt='d', cmap='Greens', ax=axes[1],
                xticklabels=['Legit', 'Fraud'], yticklabels=['Legit', 'Fraud'])
    axes[1].set_title(f"GNN (Optimized)\nF1: {gnn_metrics['f1']:.4f}", fontsize=12, fontweight='bold')
    axes[1].set_ylabel('True Label')
    axes[1].set_xlabel('Predicted Label')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_roc_curves(y_true, xgb_proba, gnn_proba, xgb_metrics, gnn_metrics, output_path):
    """Plot ROC curves"""

    # Calculate ROC curves
    fpr_xgb, tpr_xgb, _ = roc_curve(y_true, xgb_proba)
    fpr_gnn, tpr_gnn, _ = roc_curve(y_true, gnn_proba)

    plt.figure(figsize=(10, 8))

    plt.plot(fpr_xgb, tpr_xgb, 'b-', linewidth=2,
             label=f"XGBoost (AUC = {xgb_metrics['auc_roc']:.4f})")
    plt.plot(fpr_gnn, tpr_gnn, 'g-', linewidth=2,
             label=f"GNN Optimized (AUC = {gnn_metrics['auc_roc']:.4f})")
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')

    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curves Comparison', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=11)
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_precision_recall_curves(y_true, xgb_proba, gnn_proba, xgb_metrics, gnn_metrics, output_path):
    """Plot Precision-Recall curves"""

    # Calculate PR curves
    precision_xgb, recall_xgb, _ = precision_recall_curve(y_true, xgb_proba)
    precision_gnn, recall_gnn, _ = precision_recall_curve(y_true, gnn_proba)

    plt.figure(figsize=(10, 8))

    plt.plot(recall_xgb, precision_xgb, 'b-', linewidth=2,
             label=f"XGBoost (AP = {xgb_metrics['auc_pr']:.4f})")
    plt.plot(recall_gnn, precision_gnn, 'g-', linewidth=2,
             label=f"GNN Optimized (AP = {gnn_metrics['auc_pr']:.4f})")

    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Precision-Recall Curves Comparison', fontsize=14, fontweight='bold')
    plt.legend(loc='lower left', fontsize=11)
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def save_results(xgb_metrics, gnn_metrics, error_analysis_results, comparison_df, logger):
    """Save all results"""

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # JSON results
    results = {
        'timestamp': timestamp,
        'xgboost': xgb_metrics,
        'gnn': gnn_metrics,
        'error_analysis': error_analysis_results,
        'summary': {
            'winner_auc': 'GNN' if gnn_metrics['auc_roc'] > xgb_metrics['auc_roc'] else 'XGBoost',
            'winner_f1': 'GNN' if gnn_metrics['f1'] > xgb_metrics['f1'] else 'XGBoost',
            'winner_recall': 'GNN' if gnn_metrics['recall'] > xgb_metrics['recall'] else 'XGBoost',
            'winner_precision': 'GNN' if gnn_metrics['precision'] > xgb_metrics['precision'] else 'XGBoost'
        }
    }

    results_path = RESULTS_DIR / f"evaluation_results_{timestamp}.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    logger.info(f"\n  ✓ Results saved: {results_path}")

    # CSV comparison
    csv_path = RESULTS_DIR / f"predictions_comparison_{timestamp}.csv"
    comparison_df.to_csv(csv_path, index=False)
    logger.info(f"  ✓ Predictions saved: {csv_path}")

    # Plots
    cm_path = RESULTS_DIR / f"confusion_matrices_{timestamp}.png"
    roc_path = RESULTS_DIR / f"roc_curves_{timestamp}.png"
    pr_path = RESULTS_DIR / f"precision_recall_curves_{timestamp}.png"

    return results_path, csv_path, cm_path, roc_path, pr_path


def print_summary(xgb_metrics, gnn_metrics, error_analysis_results, logger):
    """Print comprehensive summary"""

    logger.info("\n" + "=" * 80)
    logger.info("COMPREHENSIVE MODEL EVALUATION RESULTS")
    logger.info("=" * 80)

    logger.info("\n📊 PERFORMANCE METRICS:")
    logger.info("-" * 80)
    logger.info(f"{'Metric':<20} {'XGBoost':<15} {'GNN (Optimized)':<15} {'Winner':<15}")
    logger.info("-" * 80)

    metrics_to_compare = [
        ('AUC-ROC', 'auc_roc'),
        ('AUC-PR', 'auc_pr'),
        ('Precision', 'precision'),
        ('Recall', 'recall'),
        ('F1-Score', 'f1'),
        ('Specificity', 'specificity')
    ]

    for name, key in metrics_to_compare:
        xgb_val = xgb_metrics[key]
        gnn_val = gnn_metrics[key]
        winner = '🔥 GNN' if gnn_val > xgb_val else '✅ XGBoost' if xgb_val > gnn_val else '🤝 Tie'
        logger.info(f"{name:<20} {xgb_val:<15.4f} {gnn_val:<15.4f} {winner:<15}")

    logger.info("\n📈 CONFUSION MATRIX COMPARISON:")
    logger.info("-" * 80)
    logger.info(f"{'Metric':<20} {'XGBoost':<15} {'GNN (Optimized)':<15}")
    logger.info("-" * 80)
    logger.info(f"{'True Positives':<20} {xgb_metrics['tp']:<15,} {gnn_metrics['tp']:<15,}")
    logger.info(f"{'False Negatives':<20} {xgb_metrics['fn']:<15,} {gnn_metrics['fn']:<15,}")
    logger.info(f"{'False Positives':<20} {xgb_metrics['fp']:<15,} {gnn_metrics['fp']:<15,}")
    logger.info(f"{'True Negatives':<20} {xgb_metrics['tn']:<15,} {gnn_metrics['tn']:<15,}")

    logger.info("\n🔍 ERROR ANALYSIS:")
    logger.info("-" * 80)
    logger.info(f"Total samples: {error_analysis_results['total_samples']:,}")
    logger.info(f"Total fraud: {error_analysis_results['total_fraud']:,}")
    logger.info(f"Total legitimate: {error_analysis_results['total_legit']:,}")
    logger.info(
        f"\nBoth models correct: {error_analysis_results['both_right']:,} ({error_analysis_results['both_right'] / error_analysis_results['total_samples'] * 100:.2f}%)")
    logger.info(
        f"Both models wrong: {error_analysis_results['both_wrong']:,} ({error_analysis_results['both_wrong'] / error_analysis_results['total_samples'] * 100:.2f}%)")
    logger.info(f"XGBoost better: {error_analysis_results['xgb_better']:,}")
    logger.info(f"GNN better: {error_analysis_results['gnn_better']:,}")
    logger.info(
        f"\nDisagreement: {error_analysis_results['disagreement_count']:,} ({error_analysis_results['disagreement_rate'] * 100:.2f}%)")

    logger.info("\n" + "=" * 80)


def main():
    """Main evaluation pipeline"""

    logger = get_logger("evaluate_models")

    try:
        logger.info("=" * 80)
        logger.info("COMPREHENSIVE MODEL EVALUATION")
        logger.info("=" * 80)

        # Load models
        xgb_model, gnn_model = load_models(logger)

        # Load test data
        X_test, y_test, graph_test = load_test_data(logger)

        # XGBoost predictions
        xgb_proba, xgb_pred = predict_xgboost(xgb_model, X_test, logger)

        # GNN predictions
        gnn_proba, gnn_pred = predict_gnn(gnn_model, graph_test, logger)

        # Calculate metrics
        logger.info("\n[5] Calculating metrics...")
        xgb_metrics = calculate_metrics(y_test, xgb_proba, xgb_pred, 'XGBoost')
        gnn_metrics = calculate_metrics(y_test, gnn_proba, gnn_pred, 'GNN')
        logger.info("  ✓ Metrics calculated")

        # Error analysis
        logger.info("\n[6] Performing error analysis...")
        error_results = error_analysis(y_test, xgb_pred, gnn_pred, xgb_proba, gnn_proba)
        logger.info("  ✓ Error analysis complete")

        # Create comparison dataframe
        logger.info("\n[7] Creating comparison dataframe...")
        comparison_df = create_comparison_dataframe(y_test, xgb_proba, xgb_pred, gnn_proba, gnn_pred)
        logger.info("  ✓ Comparison dataframe created")

        # Save results
        logger.info("\n[8] Saving results...")
        results_path, csv_path, cm_path, roc_path, pr_path = save_results(
            xgb_metrics, gnn_metrics, error_results, comparison_df, logger
        )

        # Create plots
        logger.info("\n[9] Creating visualizations...")
        plot_confusion_matrices(xgb_metrics, gnn_metrics, cm_path)
        logger.info(f"  ✓ Confusion matrices: {cm_path}")

        plot_roc_curves(y_test, xgb_proba, gnn_proba, xgb_metrics, gnn_metrics, roc_path)
        logger.info(f"  ✓ ROC curves: {roc_path}")

        plot_precision_recall_curves(y_test, xgb_proba, gnn_proba, xgb_metrics, gnn_metrics, pr_path)
        logger.info(f"  ✓ Precision-Recall curves: {pr_path}")

        # Print summary
        print_summary(xgb_metrics, gnn_metrics, error_results, logger)

        logger.info("\n" + "=" * 80)
        logger.info("✅ EVALUATION COMPLETE!")
        logger.info("=" * 80)
        logger.info(f"\nResults directory: {RESULTS_DIR}")
        logger.info(f"  - Metrics: {results_path.name}")
        logger.info(f"  - Predictions: {csv_path.name}")
        logger.info(f"  - Confusion Matrices: {cm_path.name}")
        logger.info(f"  - ROC Curves: {roc_path.name}")
        logger.info(f"  - PR Curves: {pr_path.name}")
        logger.info("=" * 80)

        logger.finalize(status="success")

    except Exception as e:
        logger.error(f"Evaluation failed: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        logger.finalize(status="failed")
        raise


if __name__ == "__main__":
    main()
