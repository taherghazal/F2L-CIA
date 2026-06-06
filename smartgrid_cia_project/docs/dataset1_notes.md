# Dataset 1 Integration Notes

This update adapts the project to the Kaggle Smart Grid Intrusion Detection Dataset.

## Integration logic
- CSV files are loaded from a local `data/` folder.
- The loader automatically searches for common label columns such as `label`, `class`, `attack`, or `target`.
- Categorical columns are label-encoded.
- Numeric nulls are median-imputed.
- Data are partitioned across fog nodes to emulate decentralized local training.
- Each fog node performs anomaly filtering, local intrusion classification, and reliability scoring.
- The cloud layer aggregates local feature-importance vectors using weighted federated aggregation.

## User action needed
Download the dataset manually from Kaggle and place the CSV file(s) in the `data/` folder before execution.
