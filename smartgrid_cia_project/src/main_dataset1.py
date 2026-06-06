import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

from src.dataset1_loader import load_dataset1, split_across_fog_nodes
from src.fog_layer import FogNodeClassifier
from src.cloud_layer import federated_aggregation

RESULTS_DIR = Path('results_dataset1')
RESULTS_DIR.mkdir(exist_ok=True)


def run(data_dir, num_nodes=8):
    df, target_col, labels = load_dataset1(data_dir)
    parts = split_across_fog_nodes(df, num_nodes=num_nodes)

    local_updates = []
    metrics = []
    scored_frames = []
    for node_id, part in parts:
        if len(part) < 10:
            continue
        fog = FogNodeClassifier(node_id=node_id, contamination=0.05, random_state=42 + node_id)
        scored = fog.score_and_filter(part)
        upd = fog.train_local(scored)
        local_updates.append(upd)
        scored_frames.append(scored)
        metrics.append({
            'node': node_id,
            'accuracy': upd['accuracy'],
            'f1': upd['f1'],
            'lambda_k': upd['lambda_k'],
            'trusted_rate': upd['trusted_rate'],
            'mean_anomaly_score': upd['mean_anomaly_score'],
        })

    metrics_df = pd.DataFrame(metrics)
    scored_df = pd.concat(scored_frames, ignore_index=True)
    agg = federated_aggregation(local_updates, beta=1.5)

    summary = pd.DataFrame([{
        'dataset': 'Kaggle Smart Grid Intrusion Detection Dataset',
        'detected_target_column': target_col,
        'num_classes': len(labels),
        'mean_accuracy': metrics_df['accuracy'].mean(),
        'mean_f1': metrics_df['f1'].mean(),
        'mean_reliability': metrics_df['lambda_k'].mean(),
        'mean_trusted_rate': metrics_df['trusted_rate'].mean(),
        'federated_omega': agg['omega'],
        'federated_rho': agg['rho'],
        'federated_score': agg['f_fed'],
    }])

    metrics_df.to_csv(RESULTS_DIR / 'fog_node_metrics.csv', index=False)
    scored_df.to_csv(RESULTS_DIR / 'scored_packets_dataset1.csv', index=False)
    summary.to_csv(RESULTS_DIR / 'summary_dataset1.csv', index=False)

    plt.figure(figsize=(8,4))
    plt.bar(metrics_df['node'].astype(str), metrics_df['accuracy'], color='teal')
    plt.title('Fog Node Accuracy - Dataset 1')
    plt.xlabel('Fog Node')
    plt.ylabel('Accuracy')
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'fog_node_accuracy.png', dpi=200)
    plt.close()

    plt.figure(figsize=(8,4))
    plt.scatter(metrics_df['trusted_rate'], metrics_df['f1'], color='darkred')
    for _, row in metrics_df.iterrows():
        plt.annotate(f"N{int(row['node'])}", (row['trusted_rate'], row['f1']))
    plt.title('Trusted Rate vs F1 - Dataset 1')
    plt.xlabel('Trusted Packet Rate')
    plt.ylabel('Weighted F1')
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'trusted_rate_vs_f1.png', dpi=200)
    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--num_nodes', type=int, default=8)
    args = parser.parse_args()
    run(args.data_dir, args.num_nodes)
