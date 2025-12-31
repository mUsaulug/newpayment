"""
GNN (Graph Neural Network) Model Training Script - OPTIMIZED
==================================================
Transaction graph üzerinde fraud detection için GNN modeli eğitir.

OPTIMIZATIONS:
- Reduced patience (3 epochs)
- Reduced max epochs (15)
- Larger batch size (1024) → Faster training
- Target training time: 30-45 minutes

Özellikler:
- GraphSAGE architecture (scalable, inductive)
- Imbalanced data handling (weighted loss + focal loss)
- Mini-batch training (memory efficiency)
- Early stopping & validation monitoring
- Comprehensive metrics (AUC, precision, recall, F1)
- Model checkpointing

Çıktı: models/gnn_fraud_model.pt
       models/gnn_metrics.json
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from torch_geometric.loader import NeighborLoader
import numpy as np
import json
from pathlib import Path
from datetime import datetime
from sklearn.metrics import (
    roc_auc_score, precision_recall_curve,
    classification_report, confusion_matrix,
    average_precision_score
)
from tqdm import tqdm

from utils.logger import get_logger

# Paths
BASE_DIR = Path(__file__).parent.parent
DATA_PROCESSED = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)

TRAIN_GRAPH = DATA_PROCESSED / "graph_train.pt"
TEST_GRAPH = DATA_PROCESSED / "graph_test.pt"

# Device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def convert_to_native_types(obj):
    """NumPy ve Torch tiplerini Python native tiplere çevir"""
    if isinstance(obj, dict):
        return {key: convert_to_native_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_native_types(item) for item in obj]
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return convert_to_native_types(obj.tolist())
    elif isinstance(obj, torch.Tensor):
        return convert_to_native_types(obj.detach().cpu().numpy())
    else:
        return obj


class GraphSAGE_FraudDetector(nn.Module):
    """
    GraphSAGE-based fraud detection model

    Architecture:
    - 3 GraphSAGE layers (mean aggregation)
    - Batch normalization (stability)
    - Dropout (overfitting prevention)
    - Binary classification head
    """

    def __init__(self, in_channels, hidden_channels=128, num_layers=3, dropout=0.3):
        super().__init__()

        self.num_layers = num_layers
        self.dropout = dropout

        # GraphSAGE layers
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
        """Forward pass"""

        # GraphSAGE layers with batch norm and dropout
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            x = self.batch_norms[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        # Classification
        out = self.classifier(x)

        return out.squeeze()


class FocalLoss(nn.Module):
    """
    Focal Loss for imbalanced classification

    Focuses training on hard examples (misclassified samples)
    """

    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean()


def load_graph_data(logger):
    """Load preprocessed graph data"""
    logger.info("[1] Loading graph data...")

    train_data = torch.load(TRAIN_GRAPH, weights_only=False)
    test_data = torch.load(TEST_GRAPH, weights_only=False)

    logger.info(f"  ✓ Train graph: {train_data.num_nodes:,} nodes, {train_data.num_edges:,} edges")
    logger.info(f"  ✓ Test graph: {test_data.num_nodes:,} nodes, {test_data.num_edges:,} edges")
    logger.info(f"  ✓ Node features: {train_data.x.shape[1]} dimensions")
    logger.info(f"  ✓ Train fraud ratio: {train_data.y.float().mean():.4f}")
    logger.info(f"  ✓ Test fraud ratio: {test_data.y.float().mean():.4f}")

    return train_data, test_data


def create_data_loaders(train_data, batch_size=1024, num_neighbors=[15, 10]):
    """
    Create mini-batch loaders for GNN training

    NeighborLoader samples neighbors for each node (scalable training)

    OPTIMIZED: batch_size=1024 (2x larger → faster training)
    """

    # Split train into train/val (80/20)
    num_nodes = train_data.num_nodes
    indices = torch.randperm(num_nodes)

    train_size = int(0.8 * num_nodes)
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    # Train loader
    train_loader = NeighborLoader(
        train_data,
        num_neighbors=num_neighbors,
        batch_size=batch_size,
        input_nodes=train_indices,
        shuffle=True,
        num_workers=0  # Windows için 0
    )

    # Validation loader
    val_loader = NeighborLoader(
        train_data,
        num_neighbors=num_neighbors,
        batch_size=batch_size,
        input_nodes=val_indices,
        shuffle=False,
        num_workers=0
    )

    return train_loader, val_loader, train_indices, val_indices


def train_epoch(model, loader, optimizer, criterion, device, logger):
    """Train for one epoch"""
    model.train()

    total_loss = 0
    num_batches = 0

    for batch in tqdm(loader, desc="Training", leave=False):
        batch = batch.to(device)

        optimizer.zero_grad()

        # Forward
        out = model(batch.x, batch.edge_index)

        # Get batch nodes (not all nodes in subgraph)
        batch_size = batch.batch_size
        out = out[:batch_size]
        target = batch.y[:batch_size].float()

        # Loss
        loss = criterion(out, target)

        # Backward
        loss.backward()

        # Gradient clipping (stability)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / num_batches


@torch.no_grad()
def evaluate(model, data, indices, device, batch_size=1024, num_neighbors=[15, 10]):
    """Evaluate model on given node indices"""
    model.eval()

    # Create loader for evaluation
    loader = NeighborLoader(
        data,
        num_neighbors=num_neighbors,
        batch_size=batch_size,
        input_nodes=indices,
        shuffle=False,
        num_workers=0
    )

    all_preds = []
    all_probs = []
    all_labels = []

    for batch in tqdm(loader, desc="Evaluating", leave=False):
        batch = batch.to(device)

        # Forward
        out = model(batch.x, batch.edge_index)

        # Get batch nodes
        batch_size_actual = batch.batch_size
        out = out[:batch_size_actual]
        target = batch.y[:batch_size_actual]

        # Predictions
        probs = torch.sigmoid(out)
        preds = (probs > 0.5).long()

        all_probs.append(probs.cpu())
        all_preds.append(preds.cpu())
        all_labels.append(target.cpu())

    # Concatenate
    all_probs = torch.cat(all_probs).numpy()
    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()

    # Metrics
    metrics = {}

    # Check if we have both classes
    if len(np.unique(all_labels)) < 2:
        metrics['auc'] = 0.0
        metrics['ap'] = 0.0
        metrics['precision'] = 0.0
        metrics['recall'] = 0.0
        metrics['f1'] = 0.0
        metrics['tn'], metrics['fp'] = 0, 0
        metrics['fn'], metrics['tp'] = 0, 0
    else:
        metrics['auc'] = float(roc_auc_score(all_labels, all_probs))
        metrics['ap'] = float(average_precision_score(all_labels, all_probs))

        report = classification_report(all_labels, all_preds, output_dict=True, zero_division=0)
        metrics['precision'] = float(report['1']['precision']) if '1' in report else 0.0
        metrics['recall'] = float(report['1']['recall']) if '1' in report else 0.0
        metrics['f1'] = float(report['1']['f1-score']) if '1' in report else 0.0

        cm = confusion_matrix(all_labels, all_preds)
        if cm.shape == (2, 2):
            metrics['tn'], metrics['fp'] = int(cm[0][0]), int(cm[0][1])
            metrics['fn'], metrics['tp'] = int(cm[1][0]), int(cm[1][1])
        else:
            metrics['tn'], metrics['fp'] = 0, 0
            metrics['fn'], metrics['tp'] = 0, 0

    return metrics, all_probs, all_labels


def find_optimal_threshold(y_true, y_proba, target_recall=0.85):
    """Find optimal threshold for high recall"""
    if len(np.unique(y_true)) < 2:
        return 0.5

    precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)

    valid_indices = recalls >= target_recall

    if valid_indices.sum() == 0:
        return 0.5

    valid_precisions = precisions[valid_indices]
    valid_thresholds = thresholds[valid_indices[:-1]]

    if len(valid_thresholds) == 0:
        return 0.5

    best_idx = valid_precisions.argmax()
    optimal_threshold = float(valid_thresholds[best_idx])

    return optimal_threshold


def train_gnn(model, train_loader, val_loader, train_data, val_indices, device, logger,
              num_epochs=15, patience=3, lr=0.001):
    """
    Full GNN training loop with early stopping

    OPTIMIZED:
    - num_epochs=15 (reduced from 30)
    - patience=3 (reduced from 7) → Stops faster if no improvement
    """

    logger.info("[3] Training GNN model...")

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=2, verbose=True
    )

    # Loss function (Focal Loss for imbalanced data)
    criterion = FocalLoss(alpha=0.25, gamma=2.0)

    # Early stopping
    best_val_auc = 0
    patience_counter = 0
    best_model_state = None

    # Training history
    history = {
        'train_loss': [],
        'val_auc': [],
        'val_ap': [],
        'val_f1': []
    }

    logger.info(f"  Device: {device}")
    logger.info(f"  Epochs: {num_epochs}")
    logger.info(f"  Learning rate: {lr}")
    logger.info(f"  Early stopping patience: {patience}")

    for epoch in range(1, num_epochs + 1):
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, logger)

        # Validate
        val_metrics, _, _ = evaluate(model, train_data, val_indices, device)

        # Update history
        history['train_loss'].append(train_loss)
        history['val_auc'].append(val_metrics['auc'])
        history['val_ap'].append(val_metrics['ap'])
        history['val_f1'].append(val_metrics['f1'])

        # Learning rate scheduling
        scheduler.step(val_metrics['auc'])

        # Log progress
        logger.info(f"  Epoch {epoch:03d} | "
                    f"Loss: {train_loss:.4f} | "
                    f"Val AUC: {val_metrics['auc']:.4f} | "
                    f"Val F1: {val_metrics['f1']:.4f}")

        # Early stopping check
        if val_metrics['auc'] > best_val_auc:
            best_val_auc = val_metrics['auc']
            patience_counter = 0
            best_model_state = model.state_dict().copy()
            logger.info(f"    → New best model! (AUC: {best_val_auc:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"  Early stopping triggered at epoch {epoch}")
                break

    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    logger.info(f"  ✓ Training complete! Best val AUC: {best_val_auc:.4f}")

    return model, history


def evaluate_final_model(model, train_data, test_data, val_indices, device, logger):
    """Final evaluation on validation and test sets"""

    logger.info("[4] Final model evaluation...")

    # Validation
    val_metrics, val_probs, val_labels = evaluate(model, train_data, val_indices, device)

    # Test (all test nodes)
    test_indices = torch.arange(test_data.num_nodes)
    test_metrics, test_probs, test_labels = evaluate(model, test_data, test_indices, device)

    # Find optimal threshold
    optimal_threshold = find_optimal_threshold(val_labels, val_probs, target_recall=0.85)
    logger.info(f"  ✓ Optimal threshold: {optimal_threshold:.4f}")

    # Re-evaluate with optimal threshold
    val_preds_opt = (val_probs >= optimal_threshold).astype(int)
    test_preds_opt = (test_probs >= optimal_threshold).astype(int)

    # Update metrics
    if len(np.unique(val_labels)) >= 2:
        val_report = classification_report(val_labels, val_preds_opt, output_dict=True, zero_division=0)
        val_metrics['precision'] = float(val_report['1']['precision']) if '1' in val_report else 0.0
        val_metrics['recall'] = float(val_report['1']['recall']) if '1' in val_report else 0.0
        val_metrics['f1'] = float(val_report['1']['f1-score']) if '1' in val_report else 0.0

        val_cm = confusion_matrix(val_labels, val_preds_opt)
        if val_cm.shape == (2, 2):
            val_metrics['tn'], val_metrics['fp'] = int(val_cm[0][0]), int(val_cm[0][1])
            val_metrics['fn'], val_metrics['tp'] = int(val_cm[1][0]), int(val_cm[1][1])

    if len(np.unique(test_labels)) >= 2:
        test_report = classification_report(test_labels, test_preds_opt, output_dict=True, zero_division=0)
        test_metrics['precision'] = float(test_report['1']['precision']) if '1' in test_report else 0.0
        test_metrics['recall'] = float(test_report['1']['recall']) if '1' in test_report else 0.0
        test_metrics['f1'] = float(test_report['1']['f1-score']) if '1' in test_report else 0.0

        test_cm = confusion_matrix(test_labels, test_preds_opt)
        if test_cm.shape == (2, 2):
            test_metrics['tn'], test_metrics['fp'] = int(test_cm[0][0]), int(test_cm[0][1])
            test_metrics['fn'], test_metrics['tp'] = int(test_cm[1][0]), int(test_cm[1][1])

    # Log results
    logger.info("\n" + "=" * 60)
    logger.info("VALIDATION METRICS (Optimal Threshold)")
    logger.info("=" * 60)
    logger.info(f"AUC-ROC: {val_metrics['auc']:.4f}")
    logger.info(f"AP (PR-AUC): {val_metrics['ap']:.4f}")
    logger.info(f"Precision: {val_metrics['precision']:.4f}")
    logger.info(f"Recall: {val_metrics['recall']:.4f}")
    logger.info(f"F1-Score: {val_metrics['f1']:.4f}")
    if 'tn' in val_metrics and val_metrics['tn'] > 0:
        logger.info(f"\nConfusion Matrix:")
        logger.info(f"  TN: {val_metrics['tn']:,}  FP: {val_metrics['fp']:,}")
        logger.info(f"  FN: {val_metrics['fn']:,}  TP: {val_metrics['tp']:,}")

    logger.info("\n" + "=" * 60)
    logger.info("TEST METRICS (Final Holdout)")
    logger.info("=" * 60)
    logger.info(f"AUC-ROC: {test_metrics['auc']:.4f}")
    logger.info(f"AP (PR-AUC): {test_metrics['ap']:.4f}")
    logger.info(f"Precision: {test_metrics['precision']:.4f}")
    logger.info(f"Recall: {test_metrics['recall']:.4f}")
    logger.info(f"F1-Score: {test_metrics['f1']:.4f}")
    if 'tn' in test_metrics and test_metrics['tn'] > 0:
        logger.info(f"\nConfusion Matrix:")
        logger.info(f"  TN: {test_metrics['tn']:,}  FP: {test_metrics['fp']:,}")
        logger.info(f"  FN: {test_metrics['fn']:,}  TP: {test_metrics['tp']:,}")
    logger.info("=" * 60 + "\n")

    return val_metrics, test_metrics, optimal_threshold


def save_model_and_artifacts(model, val_metrics, test_metrics, optimal_threshold, history, logger):
    """Save model, metrics, and artifacts"""
    logger.info("[5] Saving model and artifacts...")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Model
    model_path = MODELS_DIR / f"gnn_fraud_model_{timestamp}.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_config': {
            'in_channels': model.convs[0].in_channels,
            'hidden_channels': model.convs[0].out_channels,
            'num_layers': model.num_layers,
            'dropout': model.dropout
        }
    }, model_path)
    logger.log_artifact(model_path, "GNN fraud detection model")

    # Metrics
    metrics_data = convert_to_native_types({
        'validation': val_metrics,
        'test': test_metrics,
        'optimal_threshold': optimal_threshold,
        'timestamp': timestamp,
        'history': history
    })

    metrics_path = MODELS_DIR / f"gnn_metrics_{timestamp}.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics_data, f, indent=2)
    logger.log_artifact(metrics_path, "GNN model metrics")

    # Latest model
    latest_model_path = MODELS_DIR / "gnn_fraud_model_latest.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_config': {
            'in_channels': model.convs[0].in_channels,
            'hidden_channels': model.convs[0].out_channels,
            'num_layers': model.num_layers,
            'dropout': model.dropout
        }
    }, latest_model_path)
    logger.log_artifact(latest_model_path, "Latest GNN model")

    logger.info(f"  ✓ All artifacts saved to {MODELS_DIR}")

    return model_path, metrics_path


def main():
    """Main training pipeline"""

    logger = get_logger("train_gnn")

    try:
        logger.info("GNN FRAUD DETECTION TRAINING PIPELINE - OPTIMIZED")
        logger.info("GraphSAGE-based fraud detection on transaction graphs")
        logger.info("Target: 30-45 minute training time")

        # Parameters - OPTIMIZED
        params = {
            "hidden_channels": 128,
            "num_layers": 3,
            "dropout": 0.3,
            "batch_size": 1024,  # ← 2x LARGER (faster)
            "num_neighbors": [15, 10],
            "num_epochs": 15,  # ← REDUCED (was 30)
            "patience": 3,  # ← REDUCED (was 7)
            "learning_rate": 0.001,
            "target_recall": 0.85
        }
        logger.log_parameters(params)

        # 1. Load data
        train_data, test_data = load_graph_data(logger)

        # 2. Create data loaders
        logger.info("[2] Creating mini-batch loaders...")
        train_loader, val_loader, train_indices, val_indices = create_data_loaders(
            train_data,
            batch_size=params['batch_size'],
            num_neighbors=params['num_neighbors']
        )
        logger.info(f"  ✓ Train batches: {len(train_loader)}")
        logger.info(f"  ✓ Val batches: {len(val_loader)}")

        # 3. Build model
        in_channels = train_data.x.shape[1]
        model = GraphSAGE_FraudDetector(
            in_channels=in_channels,
            hidden_channels=params['hidden_channels'],
            num_layers=params['num_layers'],
            dropout=params['dropout']
        ).to(DEVICE)

        num_params = sum(p.numel() for p in model.parameters())
        logger.info(f"  ✓ Model built: {num_params:,} parameters")

        # 4. Train
        model, history = train_gnn(
            model, train_loader, val_loader, train_data, val_indices, DEVICE, logger,
            num_epochs=params['num_epochs'],
            patience=params['patience'],
            lr=params['learning_rate']
        )

        # 5. Final evaluation
        val_metrics, test_metrics, optimal_threshold = evaluate_final_model(
            model, train_data, test_data, val_indices, DEVICE, logger
        )

        # 6. Save
        model_path, metrics_path = save_model_and_artifacts(
            model, val_metrics, test_metrics, optimal_threshold, history, logger
        )

        # Log final metrics
        final_metrics = convert_to_native_types({
            "final_test_auc": test_metrics['auc'],
            "final_test_recall": test_metrics['recall'],
            "final_test_precision": test_metrics['precision'],
            "final_test_f1": test_metrics['f1'],
            "optimal_threshold": optimal_threshold,
            "num_parameters": num_params
        })
        logger.log_metrics(final_metrics)

        logger.info("\n" + "=" * 70)
        logger.info("GNN TRAINING COMPLETE!")
        logger.info("=" * 70)
        logger.info(f"Model saved: {model_path}")
        logger.info(f"Test AUC: {test_metrics['auc']:.4f}")
        logger.info(f"Test Recall: {test_metrics['recall']:.4f}")
        logger.info(f"Test Precision: {test_metrics['precision']:.4f}")
        logger.info(f"Optimal Threshold: {optimal_threshold:.4f}")
        logger.info("=" * 70)

        logger.finalize(status="success")

    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        logger.finalize(status="failed")
        raise


if __name__ == "__main__":
    main()
