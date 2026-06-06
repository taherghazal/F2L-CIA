from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

TARGET_CANDIDATES = ['label', 'Label', 'class', 'Class', 'attack', 'Attack', 'target', 'Target']


def find_csv_files(data_dir):
    data_dir = Path(data_dir)
    return sorted(list(data_dir.glob('*.csv')))


def load_dataset1(data_dir):
    csv_files = find_csv_files(data_dir)
    if not csv_files:
        raise FileNotFoundError('No CSV files found in data directory')
    frames = [pd.read_csv(f) for f in csv_files]
    df = pd.concat(frames, ignore_index=True)

    target_col = None
    for c in TARGET_CANDIDATES:
        if c in df.columns:
            target_col = c
            break
    if target_col is None:
        raise ValueError(f'No target column found. Tried: {TARGET_CANDIDATES}')

    y = df[target_col].astype(str)
    X = df.drop(columns=[target_col]).copy()

    for col in X.columns:
        if X[col].dtype == 'object':
            X[col] = X[col].astype(str).fillna('missing')
            X[col] = LabelEncoder().fit_transform(X[col])
        else:
            X[col] = X[col].fillna(X[col].median())

    y_enc = LabelEncoder().fit_transform(y)
    processed = X.copy()
    processed['target'] = y_enc
    return processed, target_col, sorted(y.unique())


def split_across_fog_nodes(df, num_nodes=8):
    partitions = []
    for node_id in range(num_nodes):
        node_df = df.iloc[node_id::num_nodes].reset_index(drop=True)
        partitions.append((node_id, node_df))
    return partitions
