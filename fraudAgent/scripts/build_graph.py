"""
Graph Construction Script (Transaction-Level)
==============================================
Her transaction'ı bir node olarak modelleyen fraud detection graph'ı oluşturur.

Graph Structure:
- Nodes: Individual transactions (1.8M+ nodes)
- Node Features: Transaction attributes (amount, time, location, merchant, etc.)
- Edges:
  1. Temporal edges (same card, sequential transactions)
  2. Merchant edges (same merchant, temporal window)
  3. Spatial edges (geographic proximity, optional)
- Labels: is_fraud (binary)

Output: data/processed/graph_train.pt
        data/processed/graph_test.pt
"""

import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data
from pathlib import Path
from datetime import datetime, timedelta
from scipy.spatial.distance import cdist
from tqdm import tqdm

from utils.logger import get_logger

# Paths
BASE_DIR = Path(__file__).parent.parent
DATA_RAW = BASE_DIR / "data" / "raw"
DATA_PROCESSED = BASE_DIR / "data" / "processed"

TRAIN_FILE = DATA_RAW / "fraudTrain.csv"
TEST_FILE = DATA_RAW / "fraudTest.csv"


def load_and_prepare_data(file_path, logger):
    """Load CSV and prepare basic features"""
    logger.info(f"Loading data from {file_path.name}...")

    df = pd.read_csv(file_path)

    # Parse datetime
    df['trans_datetime'] = pd.to_datetime(df['trans_date_trans_time'])
    df['dob_datetime'] = pd.to_datetime(df['dob'])

    # Sort by time (important for temporal edges)
    df = df.sort_values('trans_datetime').reset_index(drop=True)

    logger.info(f"  ✓ Loaded {len(df):,} transactions")
    logger.info(f"  ✓ Fraud ratio: {df['is_fraud'].mean():.4f}")

    return df


def create_node_features(df, logger):
    """
    Create node features for each transaction

    Features (simple but effective):
    1. Normalized amount (log-scaled)
    2. Hour of day (0-23)
    3. Day of week (0-6)
    4. Is weekend (0/1)
    5. Is night (0/1)
    6. Age (years)
    7. Distance from home (km, normalized)
    8. City population (log-scaled)
    9. Category (encoded)
    10. Gender (encoded)

    Returns:
        torch.Tensor: [num_transactions, num_features]
    """
    logger.info("[1] Creating node features...")

    features = []

    # 1. Normalized amount (log-scaled)
    amt_log = np.log1p(df['amt'].values)
    amt_norm = (amt_log - amt_log.mean()) / (amt_log.std() + 1e-8)
    features.append(amt_norm)

    # 2-3. Time features
    hour = df['trans_datetime'].dt.hour.values / 23.0  # Normalize to [0, 1]
    day_of_week = df['trans_datetime'].dt.dayofweek.values / 6.0
    features.extend([hour, day_of_week])

    # 4-5. Binary time features
    is_weekend = (df['trans_datetime'].dt.dayofweek >= 5).astype(float).values
    is_night = ((df['trans_datetime'].dt.hour >= 22) | (df['trans_datetime'].dt.hour <= 5)).astype(float).values
    features.extend([is_weekend, is_night])

    # 6. Age
    age = ((df['trans_datetime'] - df['dob_datetime']).dt.days / 365.25).values
    age_norm = (age - age.mean()) / (age.std() + 1e-8)
    features.append(age_norm)

    # 7. Distance (haversine)
    def haversine_vectorized(lat1, lon1, lat2, lon2):
        R = 6371  # Earth radius in km
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
        c = 2 * np.arcsin(np.sqrt(a))
        return R * c

    distance = haversine_vectorized(
        df['lat'].values, df['long'].values,
        df['merch_lat'].values, df['merch_long'].values
    )
    distance_log = np.log1p(distance)
    distance_norm = (distance_log - distance_log.mean()) / (distance_log.std() + 1e-8)
    features.append(distance_norm)

    # 8. City population (log-scaled)
    city_pop_log = np.log1p(df['city_pop'].values)
    city_pop_norm = (city_pop_log - city_pop_log.mean()) / (city_pop_log.std() + 1e-8)
    features.append(city_pop_norm)

    # 9-10. Categorical features (simple encoding)
    from sklearn.preprocessing import LabelEncoder

    category_encoder = LabelEncoder()
    category_encoded = category_encoder.fit_transform(df['category'].astype(str))
    category_norm = category_encoded / category_encoded.max()
    features.append(category_norm)

    gender_encoded = (df['gender'] == 'M').astype(float).values
    features.append(gender_encoded)

    # Stack features
    X = np.column_stack(features)
    X_tensor = torch.FloatTensor(X)

    logger.info(f"  ✓ Created node features: {X_tensor.shape}")
    logger.info(f"    Features: amount, hour, day, weekend, night, age, distance, city_pop, category, gender")

    return X_tensor


def create_temporal_edges(df, logger, max_time_gap_hours=24):
    """
    Create temporal edges: same card, sequential transactions

    Edge if:
    - Same credit card
    - Time gap < max_time_gap_hours

    Returns:
        edge_index: [2, num_edges] tensor
    """
    logger.info("[2] Creating temporal edges (same card, sequential)...")

    edge_list = []

    # Group by card
    grouped = df.groupby('cc_num')

    for card_id, group in tqdm(grouped, desc="  Processing cards"):
        indices = group.index.tolist()
        times = group['trans_datetime'].tolist()

        # Connect sequential transactions if within time window
        for i in range(len(indices) - 1):
            current_idx = indices[i]
            next_idx = indices[i + 1]

            time_gap = (times[i + 1] - times[i]).total_seconds() / 3600  # hours

            if time_gap <= max_time_gap_hours:
                # Bidirectional edge
                edge_list.append([current_idx, next_idx])
                edge_list.append([next_idx, current_idx])

    if len(edge_list) == 0:
        logger.warning("  ! No temporal edges created")
        edge_index = torch.empty((2, 0), dtype=torch.long)
    else:
        edge_index = torch.LongTensor(edge_list).t()

    logger.info(f"  ✓ Created {edge_index.shape[1]:,} temporal edges")

    return edge_index


def create_merchant_edges(df, logger, max_time_gap_hours=24, max_edges_per_merchant=100):
    """
    Create merchant edges: same merchant, temporal window

    Edge if:
    - Same merchant
    - Time gap < max_time_gap_hours
    - Limit edges per merchant (scalability)

    Returns:
        edge_index: [2, num_edges] tensor
    """
    logger.info("[3] Creating merchant edges (same merchant, temporal window)...")

    edge_list = []

    # Group by merchant
    grouped = df.groupby('merchant')

    for merchant_id, group in tqdm(grouped, desc="  Processing merchants"):
        indices = group.index.tolist()
        times = group['trans_datetime'].tolist()

        # Limit edges if too many transactions per merchant
        if len(indices) > max_edges_per_merchant:
            # Sample random pairs
            sample_size = min(max_edges_per_merchant, len(indices))
            sampled_indices = np.random.choice(len(indices), sample_size, replace=False)
            indices = [indices[i] for i in sampled_indices]
            times = [times[i] for i in sampled_indices]

        # Connect transactions within time window
        for i in range(len(indices)):
            for j in range(i + 1, len(indices)):
                time_gap = abs((times[j] - times[i]).total_seconds() / 3600)

                if time_gap <= max_time_gap_hours:
                    # Bidirectional edge
                    edge_list.append([indices[i], indices[j]])
                    edge_list.append([indices[j], indices[i]])

    if len(edge_list) == 0:
        logger.warning("  ! No merchant edges created")
        edge_index = torch.empty((2, 0), dtype=torch.long)
    else:
        edge_index = torch.LongTensor(edge_list).t()

    logger.info(f"  ✓ Created {edge_index.shape[1]:,} merchant edges")

    return edge_index


def combine_edges(edge_indices, logger):
    """Combine multiple edge index tensors"""
    logger.info("[4] Combining edge types...")

    valid_edges = [e for e in edge_indices if e.shape[1] > 0]

    if len(valid_edges) == 0:
        logger.warning("  ! No edges to combine")
        return torch.empty((2, 0), dtype=torch.long)

    combined = torch.cat(valid_edges, dim=1)

    # Remove duplicate edges
    combined = combined.unique(dim=1)

    logger.info(f"  ✓ Total edges: {combined.shape[1]:,}")

    return combined


def create_graph(df, logger, split_name="train"):
    """
    Create PyTorch Geometric graph from transaction data

    Returns:
        Data: PyG Data object
    """
    logger.info(f"\n{'=' * 60}")
    logger.info(f"Building {split_name.upper()} graph...")
    logger.info(f"{'=' * 60}")

    # 1. Node features
    x = create_node_features(df, logger)

    # 2. Temporal edges (same card)
    temporal_edges = create_temporal_edges(df, logger, max_time_gap_hours=24)

    # 3. Merchant edges (same merchant)
    merchant_edges = create_merchant_edges(df, logger, max_time_gap_hours=24, max_edges_per_merchant=50)

    # 4. Combine edges
    edge_index = combine_edges([temporal_edges, merchant_edges], logger)

    # 5. Labels
    y = torch.LongTensor(df['is_fraud'].values)

    # 6. Create PyG Data object
    data = Data(
        x=x,
        edge_index=edge_index,
        y=y,
        num_nodes=len(df)
    )

    logger.info(f"\n✓ Graph created successfully!")
    logger.info(f"  Nodes: {data.num_nodes:,}")
    logger.info(f"  Edges: {data.num_edges:,}")
    logger.info(f"  Node features: {data.x.shape[1]}")
    logger.info(f"  Fraud ratio: {data.y.float().mean():.4f}")
    logger.info(f"  Avg degree: {data.num_edges / data.num_nodes:.2f}")

    return data


def save_graph(data, output_path, logger, description):
    """Save graph to disk"""
    torch.save(data, output_path)
    logger.log_artifact(output_path, description)
    logger.info(f"  ✓ Saved to {output_path}")


def main():
    """Main pipeline"""

    logger = get_logger("build_graph")

    try:
        logger.info("TRANSACTION-LEVEL GRAPH CONSTRUCTION PIPELINE")
        logger.info("Building fraud detection graphs with temporal and merchant edges")

        # Parameters
        params = {
            "temporal_window_hours": 24,
            "merchant_window_hours": 24,
            "max_merchant_edges": 50,
            "node_features": 10
        }
        logger.log_parameters(params)

        # 1. Load data
        logger.info("\n[STEP 1] Loading datasets...")
        df_train = load_and_prepare_data(TRAIN_FILE, logger)
        df_test = load_and_prepare_data(TEST_FILE, logger)

        # 2. Build train graph
        logger.info("\n[STEP 2] Building TRAIN graph...")
        train_graph = create_graph(df_train, logger, split_name="train")

        # 3. Build test graph
        logger.info("\n[STEP 3] Building TEST graph...")
        test_graph = create_graph(df_test, logger, split_name="test")

        # 4. Save graphs
        logger.info("\n[STEP 4] Saving graphs...")

        train_output = DATA_PROCESSED / "graph_train.pt"
        test_output = DATA_PROCESSED / "graph_test.pt"

        save_graph(train_graph, train_output, logger, "Training transaction graph")
        save_graph(test_graph, test_output, logger, "Test transaction graph")

        # Log metrics
        logger.log_metrics({
            "train_nodes": int(train_graph.num_nodes),
            "train_edges": int(train_graph.num_edges),
            "train_avg_degree": float(train_graph.num_edges / train_graph.num_nodes),
            "test_nodes": int(test_graph.num_nodes),
            "test_edges": int(test_graph.num_edges),
            "test_avg_degree": float(test_graph.num_edges / test_graph.num_nodes),
            "node_features": int(train_graph.x.shape[1])
        })

        logger.info("\nGRAPH CONSTRUCTION COMPLETE!")
        logger.finalize(status="success")

    except Exception as e:
        logger.error(f"Graph construction failed: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        logger.finalize(status="failed")
        raise


if __name__ == "__main__":
    main()
