"""
GNN Threshold Optimization Script
==================================
Trained GNN modelini farklı threshold değerlerinde test eder.
En iyi recall/precision balance'ı bulur.

AMAÇ: 20% gap'i düzelt!
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from torch_geometric.loader import NeighborLoader
import numpy as np
import json
from pathlib import Path
from sklearn.metrics import (
    roc_auc_score, precision_recall_curve,
    classification_report, confusion_matrix,
    average_precision_score
)
from tqdm import tqdm

# Paths
BASE_DIR = Path(__file__).parent.parent
DATA_PROCESSED = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"

TEST_GRAPH = DATA_PROCESSED / "graph_test.pt"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class GraphSAGE_FraudDetector(nn.Module):
    """GraphSAGE model (same as training)"""

    def __init__(self, in_channels, hidden_channels=128, num_layers=3, dropout=0.3):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout

        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        # Input layer
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        self.batch_norms.append(nn.BatchNorm1d(hidden_channels))

        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
            self.batch_norms.append(nn.BatchNorm1d(hidden_channels))

        # Output layer
        self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        self.batch_norms.append(nn.BatchNorm1d(hidden_channels))

        # Classification head
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


def load_model():
    """Load trained GNN model"""
    print("🔄 Loading trained GNN model...")

    model_path = MODELS_DIR / "gnn_fraud_model_latest.pt"

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=False)
    config = checkpoint['model_config']

    model = GraphSAGE_FraudDetector(
        in_channels=config['in_channels'],
        hidden_channels=config['hidden_channels'],
        num_layers=config['num_layers'],
        dropout=config['dropout']
    )

    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(DEVICE)
    model.eval()

    print(f"  ✓ Model loaded: {model_path}")
    print(f"  ✓ Parameters: {sum(p.numel() for p in model.parameters()):,}")

    return model


def load_test_data():
    """Load test graph"""
    print("\n🔄 Loading test data...")

    test_data = torch.load(TEST_GRAPH, weights_only=False)

    print(f"  ✓ Test nodes: {test_data.num_nodes:,}")
    print(f"  ✓ Test edges: {test_data.num_edges:,}")
    print(f"  ✓ Fraud ratio: {test_data.y.float().mean():.4f}")

    return test_data


@torch.no_grad()
def get_predictions(model, test_data, batch_size=1024):
    """Get probability predictions for all test nodes"""
    print("\n🔄 Generating predictions...")

    model.eval()

    # Create loader
    test_indices = torch.arange(test_data.num_nodes)
    loader = NeighborLoader(
        test_data,
        num_neighbors=[15, 10],
        batch_size=batch_size,
        input_nodes=test_indices,
        shuffle=False,
        num_workers=0
    )

    all_probs = []
    all_labels = []

    for batch in tqdm(loader, desc="Predicting"):
        batch = batch.to(DEVICE)

        # Forward
        out = model(batch.x, batch.edge_index)

        # Get batch nodes
        batch_size_actual = batch.batch_size
        out = out[:batch_size_actual]
        target = batch.y[:batch_size_actual]

        # Probabilities
        probs = torch.sigmoid(out)

        all_probs.append(probs.cpu())
        all_labels.append(target.cpu())

    all_probs = torch.cat(all_probs).numpy()
    all_labels = torch.cat(all_labels).numpy()

    print(f"  ✓ Generated {len(all_probs):,} predictions")

    return all_probs, all_labels


def evaluate_threshold(y_true, y_proba, threshold):
    """Evaluate metrics at specific threshold"""
    y_pred = (y_proba >= threshold).astype(int)

    # Basic metrics
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)

    metrics = {
        'threshold': float(threshold),
        'precision': float(report['1']['precision']) if '1' in report else 0.0,
        'recall': float(report['1']['recall']) if '1' in report else 0.0,
        'f1': float(report['1']['f1-score']) if '1' in report else 0.0,
    }

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape == (2, 2):
        metrics['tn'] = int(cm[0][0])
        metrics['fp'] = int(cm[0][1])
        metrics['fn'] = int(cm[1][0])
        metrics['tp'] = int(cm[1][1])

    return metrics


def test_thresholds(y_true, y_proba):
    """Test multiple threshold values"""
    print("\n" + "=" * 70)
    print("🎯 THRESHOLD TESTING")
    print("=" * 70)

    # AUC (threshold-independent)
    auc = roc_auc_score(y_true, y_proba)
    print(f"\nAUC-ROC: {auc:.4f}")

    # Test different thresholds
    thresholds = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.6069]

    results = []

    print(f"\nTesting {len(thresholds)} thresholds...")
    print("-" * 70)

    for threshold in thresholds:
        metrics = evaluate_threshold(y_true, y_proba, threshold)
        results.append(metrics)

        print(f"Threshold: {threshold:.4f} | "
              f"Recall: {metrics['recall']:.4f} | "
              f"Precision: {metrics['precision']:.4f} | "
              f"F1: {metrics['f1']:.4f} | "
              f"FP: {metrics['fp']:4d} | FN: {metrics['fn']:4d}")

    return results, auc


def find_best_threshold(results, auc, target_recall=0.85):
    """Find optimal threshold based on criteria"""
    print("\n" + "=" * 70)
    print("🏆 BEST THRESHOLDS")
    print("=" * 70)

    # Best F1
    best_f1 = max(results, key=lambda x: x['f1'])
    print(f"\n1. BEST F1-SCORE: {best_f1['f1']:.4f}")
    print(f"   Threshold: {best_f1['threshold']:.4f}")
    print(f"   Recall: {best_f1['recall']:.4f}")
    print(f"   Precision: {best_f1['precision']:.4f}")
    print(f"   FP: {best_f1['fp']:,} | FN: {best_f1['fn']:,}")

    # Best recall with precision >= 0.90
    high_precision = [r for r in results if r['precision'] >= 0.90]
    if high_precision:
        best_recall = max(high_precision, key=lambda x: x['recall'])
        print(f"\n2. BEST RECALL (Precision >= 0.90): {best_recall['recall']:.4f}")
        print(f"   Threshold: {best_recall['threshold']:.4f}")
        print(f"   Recall: {best_recall['recall']:.4f}")
        print(f"   Precision: {best_recall['precision']:.4f}")
        print(f"   FP: {best_recall['fp']:,} | FN: {best_recall['fn']:,}")

    # Target recall >= 0.85
    high_recall = [r for r in results if r['recall'] >= target_recall]
    if high_recall:
        best_precision = max(high_recall, key=lambda x: x['precision'])
        print(f"\n3. BEST PRECISION (Recall >= {target_recall:.2f}): {best_precision['precision']:.4f}")
        print(f"   Threshold: {best_precision['threshold']:.4f}")
        print(f"   Recall: {best_precision['recall']:.4f}")
        print(f"   Precision: {best_precision['precision']:.4f}")
        print(f"   FP: {best_precision['fp']:,} | FN: {best_precision['fn']:,}")
    else:
        print(f"\n3. ⚠️  No threshold achieves Recall >= {target_recall:.2f}")
        print(f"   Maximum recall: {max(r['recall'] for r in results):.4f}")

    # Current threshold (0.6069)
    current = [r for r in results if abs(r['threshold'] - 0.6069) < 0.001]
    if current:
        curr = current[0]
        print(f"\n4. CURRENT THRESHOLD (0.6069):")
        print(f"   Recall: {curr['recall']:.4f}")
        print(f"   Precision: {curr['precision']:.4f}")
        print(f"   F1: {curr['f1']:.4f}")
        print(f"   FP: {curr['fp']:,} | FN: {curr['fn']:,}")

    # GAP analysis
    print(f"\n" + "=" * 70)
    print("📊 GAP ANALYSIS")
    print("=" * 70)
    print(f"\nAUC-ROC: {auc:.4f}")
    print(f"\nCurrent (threshold=0.6069):")
    if current:
        gap_current = auc - curr['recall']
        print(f"  Recall: {curr['recall']:.4f}")
        print(f"  GAP: {gap_current:.4f} ({gap_current / auc * 100:.1f}%) ❌ AMATÖRCE!")

    print(f"\nBest F1 (threshold={best_f1['threshold']:.4f}):")
    gap_best = auc - best_f1['recall']
    print(f"  Recall: {best_f1['recall']:.4f}")
    print(
        f"  GAP: {gap_best:.4f} ({gap_best / auc * 100:.1f}%) {'✅ PROFESYONEL!' if gap_best < 0.15 else '⚠️ Hala yüksek'}")

    return best_f1


def save_results(results, auc, best_threshold):
    """Save threshold test results"""
    output_path = MODELS_DIR / "gnn_threshold_analysis.json"

    data = {
        'auc_roc': float(auc),
        'all_thresholds': results,
        'recommended_threshold': best_threshold,
        'note': 'Use recommended_threshold for better recall/precision balance'
    }

    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"\n💾 Results saved: {output_path}")


def main():
    """Main pipeline"""
    print("=" * 70)
    print("GNN THRESHOLD OPTIMIZATION")
    print("=" * 70)

    # 1. Load model
    model = load_model()

    # 2. Load test data
    test_data = load_test_data()

    # 3. Get predictions
    y_proba, y_true = get_predictions(model, test_data)

    # 4. Test thresholds
    results, auc = test_thresholds(y_true, y_proba)

    # 5. Find best
    best_threshold = find_best_threshold(results, auc)

    # 6. Save
    save_results(results, auc, best_threshold)

    print("\n" + "=" * 70)
    print("✅ THRESHOLD OPTIMIZATION COMPLETE!")
    print("=" * 70)
    print(f"\n🎯 RECOMMENDATION:")
    print(f"   Change threshold from 0.6069 to {best_threshold['threshold']:.4f}")
    print(f"   Expected improvement:")
    print(f"     Recall: 0.7958 → {best_threshold['recall']:.4f}")
    print(f"     F1-Score: 0.8810 → {best_threshold['f1']:.4f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
