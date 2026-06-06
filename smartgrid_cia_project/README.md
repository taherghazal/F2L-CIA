# Smart Grid CIA Optimization Project - Dataset 1 Update

This version updates the earlier research project to support **Dataset 1: Kaggle Smart Grid Intrusion Detection Dataset**.

Dataset reference:
- Kaggle dataset: Smart Grid Intrusion Detection Dataset by hussainsheikh03
- URL: https://www.kaggle.com/datasets/hussainsheikh03/smart-grid-intrusion-detection-dataset

## What's new
- Added dataset ingestion pipeline for CSV-based intrusion detection data
- Added preprocessing for numeric/categorical smart-grid security features
- Added binary/multiclass target preparation logic
- Added fog-layer local classifiers for intrusion detection
- Added cloud-layer federated aggregation workflow for classifier parameters
- Preserved AES-GCM edge security and CIA metric reporting

## Expected dataset placement
Place the Kaggle CSV file(s) inside:

```bash
data/
```

You can then run:

```bash
python -m src.main_dataset1 --data_dir data
```

## If column names differ
The loader includes flexible target-column detection, but you may edit `src/dataset1_loader.py` to match the exact CSV schema after downloading the dataset.
