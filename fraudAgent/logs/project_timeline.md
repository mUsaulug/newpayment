# Fraud ML Service - Project Timeline

Bu dosya tüm script çalıştırmalarını kronolojik olarak kaydeder.

---

## 2025-12-29 17:43:46 - build_graph

**Run ID**: `20251229_174346`  
**Status**: success  
**Duration**: 52.84s  

**Parameters**:
- train_file: `C:\Users\monster\Desktop\PyhtonLesson\fraudAgent\data\raw\fraudTrain.csv`
- test_file: `C:\Users\monster\Desktop\PyhtonLesson\fraudAgent\data\raw\fraudTest.csv`
- output_dir: `C:\Users\monster\Desktop\PyhtonLesson\fraudAgent\data\processed`

**Metrics**:
- train_nodes: `1676`
- train_edges: `1296675`
- train_fraud_ratio: `0.0057886517606675625`
- test_nodes: `1676`
- test_edges: `555556`
- test_fraud_ratio: `0.0035675971303135157`
- num_cards: `983`
- num_merchants: `693`

**Artifacts**:
- Training graph (PyTorch Geometric): `C:\Users\monster\Desktop\PyhtonLesson\fraudAgent\data\processed\graph_train.pt`
- Test graph (PyTorch Geometric): `C:\Users\monster\Desktop\PyhtonLesson\fraudAgent\data\processed\graph_test.pt`
- Node ID mappings (card/merchant): `C:\Users\monster\Desktop\PyhtonLesson\fraudAgent\data\processed\node_mappings.pkl`

---

## 2025-12-29 21:40:34 - build_tabular_features

**Run ID**: `20251229_214034`  
**Status**: failed  
**Duration**: 180.89s  

**Parameters**:
- train_file: `C:\Users\monster\Desktop\PyhtonLesson\fraudAgent\data\raw\fraudTrain.csv`
- test_file: `C:\Users\monster\Desktop\PyhtonLesson\fraudAgent\data\raw\fraudTest.csv`
- output_dir: `C:\Users\monster\Desktop\PyhtonLesson\fraudAgent\data\processed`

---

## 2025-12-29 21:44:52 - build_tabular_features

**Run ID**: `20251229_214452`  
**Status**: success  
**Duration**: 187.06s  

**Parameters**:
- train_file: `C:\Users\monster\Desktop\PyhtonLesson\fraudAgent\data\raw\fraudTrain.csv`
- test_file: `C:\Users\monster\Desktop\PyhtonLesson\fraudAgent\data\raw\fraudTest.csv`
- output_dir: `C:\Users\monster\Desktop\PyhtonLesson\fraudAgent\data\processed`

**Metrics**:
- train_samples: `1296675`
- train_features: `34`
- train_fraud_ratio: `0.005788651743883394`
- test_samples: `555719`
- test_features: `34`
- test_fraud_ratio: `0.0038598644278853163`
- feature_count: `34`

**Artifacts**:
- Training tabular features: `C:\Users\monster\Desktop\PyhtonLesson\fraudAgent\data\processed\train_features_tabular.parquet`
- Training labels: `C:\Users\monster\Desktop\PyhtonLesson\fraudAgent\data\processed\train_labels.parquet`
- Test tabular features: `C:\Users\monster\Desktop\PyhtonLesson\fraudAgent\data\processed\test_features_tabular.parquet`
- Test labels: `C:\Users\monster\Desktop\PyhtonLesson\fraudAgent\data\processed\test_labels.parquet`
- Feature column names: `C:\Users\monster\Desktop\PyhtonLesson\fraudAgent\data\processed\feature_columns.txt`

---

## 2025-12-29 21:53:33 - train_xgboost

**Run ID**: `20251229_215333`  
**Status**: failed  
**Duration**: 24.68s  

**Parameters**:
- val_size: `0.2`
- target_recall: `0.85`
- default_threshold: `0.5`
- smote_sampling_strategy: `0.3`
- xgb_max_depth: `6`
- xgb_n_estimators: `200`
- xgb_scale_pos_weight: `10`

**Artifacts**:
- XGBoost fraud detection model (with SMOTE pipeline): `C:\Users\monster\Desktop\PyhtonLesson\fraudAgent\models\xgboost_fraud_model_20251229_215358.pkl`

---

## 2025-12-29 22:00:57 - train_xgboost

**Run ID**: `20251229_220057`  
**Status**: success  
**Duration**: 23.78s  

**Parameters**:
- val_size: `0.2`
- target_recall: `0.85`
- default_threshold: `0.5`
- smote_sampling_strategy: `0.3`
- xgb_max_depth: `6`
- xgb_n_estimators: `200`
- xgb_scale_pos_weight: `10`

**Metrics**:
- final_test_auc: `0.9964477628876608`
- final_test_recall: `0.8298368298368298`
- final_test_precision: `0.9222797927461139`
- final_test_f1: `0.8736196319018404`
- optimal_threshold: `0.9630523920059204`
- num_features: `34`

**Artifacts**:
- XGBoost fraud detection model (with SMOTE pipeline): `C:\Users\monster\Desktop\PyhtonLesson\fraudAgent\models\xgboost_fraud_model_20251229_220121.pkl`
- Model evaluation metrics: `C:\Users\monster\Desktop\PyhtonLesson\fraudAgent\models\xgboost_metrics_20251229_220121.json`
- Feature importances: `C:\Users\monster\Desktop\PyhtonLesson\fraudAgent\models\xgboost_feature_importance_20251229_220121.csv`
- Latest XGBoost model (symlink): `C:\Users\monster\Desktop\PyhtonLesson\fraudAgent\models\xgboost_fraud_model_latest.pkl`

---

## 2025-12-29 22:03:25 - train_xgboost

**Run ID**: `20251229_220325`  
**Status**: success  
**Duration**: 37.46s  

**Parameters**:
- val_size: `0.2`
- target_recall: `0.85`
- default_threshold: `0.5`
- smote_sampling_strategy: `0.3`
- xgb_max_depth: `6`
- xgb_n_estimators: `200`
- xgb_scale_pos_weight: `10`

**Metrics**:
- final_test_auc: `0.9964477628876608`
- final_test_recall: `0.8298368298368298`
- final_test_precision: `0.9222797927461139`
- final_test_f1: `0.8736196319018404`
- optimal_threshold: `0.9630523920059204`
- num_features: `34`

**Artifacts**:
- XGBoost fraud detection model (with SMOTE pipeline): `C:\Users\monster\Desktop\PyhtonLesson\fraudAgent\models\xgboost_fraud_model_20251229_220403.pkl`
- Model evaluation metrics: `C:\Users\monster\Desktop\PyhtonLesson\fraudAgent\models\xgboost_metrics_20251229_220403.json`
- Feature importances: `C:\Users\monster\Desktop\PyhtonLesson\fraudAgent\models\xgboost_feature_importance_20251229_220403.csv`
- Latest XGBoost model (symlink): `C:\Users\monster\Desktop\PyhtonLesson\fraudAgent\models\xgboost_fraud_model_latest.pkl`

---

## 2025-12-29 22:18:35 - train_gnn

**Run ID**: `20251229_221835`  
**Status**: failed  
**Duration**: 5.40s  

**Parameters**:
- hidden_channels: `128`
- num_layers: `3`
- dropout: `0.3`
- batch_size: `1024`
- num_neighbors: `[10, 5]`
- num_epochs: `50`
- patience: `10`
- learning_rate: `0.001`
- target_recall: `0.85`

---

## 2025-12-29 22:51:54 - build_graph

**Run ID**: `20251229_225154`  
**Status**: success  
**Duration**: 48.13s  

**Parameters**:
- temporal_window_hours: `24`
- merchant_window_hours: `24`
- max_merchant_edges: `50`
- node_features: `10`

**Metrics**:
- train_nodes: `1296675`
- train_edges: `2382782`
- train_avg_degree: `1.8376092698633042`
- test_nodes: `555719`
- test_edges: `1061320`
- test_avg_degree: `1.909814132682165`
- node_features: `10`

**Artifacts**:
- Training transaction graph: `C:\Users\monster\Desktop\PyhtonLesson\fraudAgent\data\processed\graph_train.pt`
- Test transaction graph: `C:\Users\monster\Desktop\PyhtonLesson\fraudAgent\data\processed\graph_test.pt`

---

## 2025-12-30 02:29:29 - train_gnn

**Run ID**: `20251230_022929`  
**Status**: success  
**Duration**: 1070.03s  

**Parameters**:
- hidden_channels: `128`
- num_layers: `3`
- dropout: `0.3`
- batch_size: `1024`
- num_neighbors: `[15, 10]`
- num_epochs: `15`
- patience: `3`
- learning_rate: `0.001`
- target_recall: `0.85`

**Metrics**:
- final_test_auc: `0.9991963129895909`
- final_test_recall: `0.7958041958041958`
- final_test_precision: `0.9867052023121388`
- final_test_f1: `0.8810322580645161`
- optimal_threshold: `0.6068904995918274`
- num_parameters: `77569`

**Artifacts**:
- GNN fraud detection model: `C:\Users\monster\Desktop\PyhtonLesson\fraudAgent\models\gnn_fraud_model_20251230_024719.pt`
- GNN model metrics: `C:\Users\monster\Desktop\PyhtonLesson\fraudAgent\models\gnn_metrics_20251230_024719.json`
- Latest GNN model: `C:\Users\monster\Desktop\PyhtonLesson\fraudAgent\models\gnn_fraud_model_latest.pt`

---

## 2025-12-31 02:52:33 - evaluate_models

**Run ID**: `20251231_025233`  
**Status**: failed  
**Duration**: 2.33s  

---

## 2025-12-31 02:55:06 - evaluate_models

**Run ID**: `20251231_025506`  
**Status**: success  
**Duration**: 27.94s  

---

## 2025-12-31 02:59:10 - evaluate_models

**Run ID**: `20251231_025910`  
**Status**: success  
**Duration**: 177.86s  

---

## 2025-12-31 03:22:23 - evaluate_models

**Run ID**: `20251231_032223`  
**Status**: success  
**Duration**: 29.03s  

---

## 2025-12-31 03:23:48 - evaluate_models

**Run ID**: `20251231_032348`  
**Status**: success  
**Duration**: 25.05s  

---

## 2025-12-31 03:25:13 - evaluate_models

**Run ID**: `20251231_032513`  
**Status**: success  
**Duration**: 24.80s  

---

