"""
Fraud Detection Inference Script
=================================
Trained models (XGBoost + GNN) ile fraud prediction yapar.

Usage:
    # Single transaction
    python scripts/predict_fraud.py --amount 5000 --sender_risk 0.8 --receiver_risk 0.3

    # Batch prediction from file
    python scripts/predict_fraud.py --input transactions.csv --output predictions.csv

    # Choose model
    python scripts/predict_fraud.py --model gnn --amount 5000 ...
"""

import argparse
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from torch_geometric.data import Data
import pandas as pd
import numpy as np
import json
from pathlib import Path
from datetime import datetime

# Paths
BASE_DIR = Path(__file__).parent.parent
MODELS_DIR = BASE_DIR / "models"

# Model paths
XGBOOST_MODEL = MODELS_DIR / "xgboost_fraud_model_latest.pkl"
GNN_MODEL = MODELS_DIR / "gnn_fraud_model_latest.pt"

# Thresholds (optimized)
XGBOOST_THRESHOLD = 0.2716
GNN_THRESHOLD = 0.45  # Updated from 0.6069!

# Device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Feature names (same for both models)
FEATURE_NAMES = [
    'amount',
    'sender_risk_score',
    'receiver_risk_score',
    'amount_sender_avg_ratio',
    'hour',
    'day_of_week',
    'sender_tx_count',
    'receiver_tx_count',
    'sender_unique_receivers',
    'receiver_unique_senders'
]


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


class FraudDetector:
    """Unified fraud detection interface"""

    def __init__(self, model_type='xgboost'):
        """
        Args:
            model_type: 'xgboost' or 'gnn'
        """
        self.model_type = model_type
        self.feature_names = FEATURE_NAMES  # ← FIX: Always set feature_names

        if model_type == 'xgboost':
            self.model = self._load_xgboost()
            self.threshold = XGBOOST_THRESHOLD
        elif model_type == 'gnn':
            self.model = self._load_gnn()
            self.threshold = GNN_THRESHOLD
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        print(f"✅ {model_type.upper()} model loaded (threshold: {self.threshold:.4f})")

    def _load_xgboost(self):
        """Load XGBoost model"""
        if not XGBOOST_MODEL.exists():
            raise FileNotFoundError(f"XGBoost model not found: {XGBOOST_MODEL}")

        try:
            # Try with joblib first (recommended for XGBoost)
            import joblib
            model_data = joblib.load(XGBOOST_MODEL)
        except:
            # Fallback to pickle with encoding
            with open(XGBOOST_MODEL, 'rb') as f:
                try:
                    model_data = pickle.load(f, encoding='latin1')
                except:
                    model_data = pickle.load(f)

        return model_data['model']

    def _load_xgboost(self):
        """Load XGBoost model"""
        if not XGBOOST_MODEL.exists():
            raise FileNotFoundError(f"XGBoost model not found: {XGBOOST_MODEL}")

        try:
            # Try with joblib first (recommended for XGBoost)
            import joblib
            model_data = joblib.load(XGBOOST_MODEL)
        except:
            # Fallback to pickle with encoding
            with open(XGBOOST_MODEL, 'rb') as f:
                try:
                    model_data = pickle.load(f, encoding='latin1')
                except:
                    model_data = pickle.load(f)

        # Check if it's a dict or direct model
        if isinstance(model_data, dict):
            # Try different key names
            if 'model' in model_data:
                return model_data['model']
            elif 'classifier' in model_data:
                return model_data['classifier']
            else:
                # Return first value if dict has other structure
                return list(model_data.values())[0]
        else:
            # Direct model object (Pipeline or XGBoost)
            return model_data

    def predict_xgboost(self, features):
        """Predict with XGBoost"""
        # Convert to DataFrame with correct feature names
        if isinstance(features, dict):
            df = pd.DataFrame([features])
        else:
            df = pd.DataFrame([features], columns=self.feature_names)

        # Get probabilities
        proba = self.model.predict_proba(df)[:, 1]

        return proba[0]

    def predict_gnn(self, features):
        """Predict with GNN"""
        # Convert features to array if dict
        if isinstance(features, dict):
            features_array = [features.get(k, 0.0) for k in self.feature_names]
        else:
            features_array = features

        # Create minimal graph (single node with self-loop)
        x = torch.tensor([features_array], dtype=torch.float32)
        edge_index = torch.tensor([[0], [0]], dtype=torch.long)  # Self-loop

        data = Data(x=x, edge_index=edge_index)
        data = data.to(DEVICE)

        with torch.no_grad():
            out = self.model(data.x, data.edge_index)
            proba = torch.sigmoid(out)

        return proba.cpu().item()

    def predict(self, features):
        """
        Predict fraud probability

        Args:
            features: dict or array-like of transaction features

        Returns:
            dict: {
                'probability': float,
                'prediction': str,
                'risk_level': str,
                'threshold': float,
                'confidence': float
            }
        """
        # Get probability
        if self.model_type == 'xgboost':
            proba = self.predict_xgboost(features)
        else:
            proba = self.predict_gnn(features)

        proba = float(proba)
        prediction = proba >= self.threshold

        # Risk level
        if proba >= 0.8:
            risk_level = "CRITICAL"
        elif proba >= 0.6:
            risk_level = "HIGH"
        elif proba >= 0.4:
            risk_level = "MEDIUM"
        elif proba >= 0.2:
            risk_level = "LOW"
        else:
            risk_level = "MINIMAL"

        return {
            'model': self.model_type,
            'probability': round(proba, 4),
            'prediction': 'FRAUD' if prediction else 'LEGITIMATE',
            'risk_level': risk_level,
            'threshold': self.threshold,
            'confidence': round(abs(proba - self.threshold), 4)
        }


def predict_single_transaction(args):
    """Predict single transaction from CLI args"""

    # Build features dict
    features = {
        'amount': args.amount,
        'sender_risk_score': args.sender_risk,
        'receiver_risk_score': args.receiver_risk,
        'amount_sender_avg_ratio': args.amount_ratio if args.amount_ratio else 1.0,
        'hour': args.hour if args.hour else 12,
        'day_of_week': args.day if args.day else 3,
        'sender_tx_count': args.sender_count if args.sender_count else 10,
        'receiver_tx_count': args.receiver_count if args.receiver_count else 10,
        'sender_unique_receivers': args.sender_unique if args.sender_unique else 5,
        'receiver_unique_senders': args.receiver_unique if args.receiver_unique else 5
    }

    # Load detector
    detector = FraudDetector(model_type=args.model)

    # Predict
    result = detector.predict(features)

    # Display results
    print("\n" + "=" * 70)
    print("🔍 FRAUD DETECTION RESULT")
    print("=" * 70)
    print(f"Model: {result['model'].upper()}")
    print(f"Fraud Probability: {result['probability']:.4f}")
    print(f"Prediction: {result['prediction']}")
    print(f"Risk Level: {result['risk_level']}")
    print(f"Threshold: {result['threshold']:.4f}")
    print(f"Confidence: {result['confidence']:.4f}")
    print("=" * 70)

    # Transaction details
    print("\n📋 Transaction Details:")
    print(f"  Amount: ${features['amount']:,.2f}")
    print(f"  Sender Risk: {features['sender_risk_score']:.2f}")
    print(f"  Receiver Risk: {features['receiver_risk_score']:.2f}")
    print(f"  Hour: {features['hour']}")
    print(f"  Day: {features['day_of_week']}")

    # Save to file if requested
    if args.output:
        output_data = {
            'timestamp': datetime.now().isoformat(),
            'features': features,
            'result': result
        }

        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2)

        print(f"\n💾 Result saved to: {args.output}")

    return result


def predict_batch(args):
    """Predict batch of transactions from CSV"""

    print(f"\n📂 Loading transactions from: {args.input}")

    # Load CSV
    df = pd.read_csv(args.input)
    print(f"  ✓ Loaded {len(df):,} transactions")

    # Load detector
    detector = FraudDetector(model_type=args.model)

    # Predict
    print(f"\n🔄 Running predictions with {args.model.upper()}...")

    results = []
    for idx, row in df.iterrows():
        features = row.to_dict()
        result = detector.predict(features)

        results.append({
            'transaction_id': idx,
            'probability': result['probability'],
            'prediction': result['prediction'],
            'risk_level': result['risk_level']
        })

        if (idx + 1) % 100 == 0:
            print(f"  Processed {idx + 1:,} / {len(df):,} transactions...")

    # Create results DataFrame
    results_df = pd.DataFrame(results)

    # Merge with original data
    output_df = pd.concat([df, results_df], axis=1)

    # Save
    output_df.to_csv(args.output, index=False)

    print(f"\n✅ Predictions complete!")
    print(f"💾 Results saved to: {args.output}")

    # Summary
    fraud_count = (results_df['prediction'] == 'FRAUD').sum()
    print(f"\n📊 Summary:")
    print(f"  Total transactions: {len(df):,}")
    print(f"  Predicted fraud: {fraud_count:,} ({fraud_count / len(df) * 100:.2f}%)")
    print(f"  Predicted legitimate: {len(df) - fraud_count:,}")

    # Risk distribution
    print(f"\n🎯 Risk Distribution:")
    for risk in ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW', 'MINIMAL']:
        count = (results_df['risk_level'] == risk).sum()
        if count > 0:
            print(f"  {risk}: {count:,} ({count / len(df) * 100:.2f}%)")


def main():
    parser = argparse.ArgumentParser(
        description='Fraud Detection Inference',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single transaction prediction
  python scripts/predict_fraud.py --amount 5000 --sender_risk 0.8 --receiver_risk 0.3

  # Use GNN model
  python scripts/predict_fraud.py --model gnn --amount 5000 --sender_risk 0.8 --receiver_risk 0.3

  # Batch prediction
  python scripts/predict_fraud.py --input transactions.csv --output predictions.csv

  # Batch with GNN
  python scripts/predict_fraud.py --model gnn --input transactions.csv --output predictions.csv
        """
    )

    # Model selection
    parser.add_argument('--model', type=str, default='xgboost', choices=['xgboost', 'gnn'],
                        help='Model to use (default: xgboost)')

    # Input modes
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--input', type=str, help='Input CSV file for batch prediction')
    group.add_argument('--amount', type=float, help='Transaction amount')

    # Single transaction features
    parser.add_argument('--sender_risk', type=float, help='Sender risk score (0-1)')
    parser.add_argument('--receiver_risk', type=float, help='Receiver risk score (0-1)')
    parser.add_argument('--amount_ratio', type=float, help='Amount/sender avg ratio')
    parser.add_argument('--hour', type=int, help='Hour of day (0-23)')
    parser.add_argument('--day', type=int, help='Day of week (0-6)')
    parser.add_argument('--sender_count', type=int, help='Sender transaction count')
    parser.add_argument('--receiver_count', type=int, help='Receiver transaction count')
    parser.add_argument('--sender_unique', type=int, help='Sender unique receivers')
    parser.add_argument('--receiver_unique', type=int, help='Receiver unique senders')

    # Output
    parser.add_argument('--output', type=str, help='Output file path')

    args = parser.parse_args()

    # Run prediction
    if args.input:
        # Batch mode
        if not args.output:
            args.output = args.input.replace('.csv', '_predictions.csv')
        predict_batch(args)
    else:
        # Single mode
        if not args.sender_risk or not args.receiver_risk:
            parser.error("--sender_risk and --receiver_risk are required for single transaction")
        predict_single_transaction(args)


if __name__ == "__main__":
    main()
