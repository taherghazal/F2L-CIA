import numpy as np
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split


class FogNodeClassifier:
    def __init__(self, node_id, contamination=0.05, random_state=42):
        self.node_id = node_id
        self.detector = IsolationForest(contamination=contamination, random_state=int(random_state))
        self.model = RandomForestClassifier(n_estimators=50, random_state=int(random_state))

    def score_and_filter(self, df):
        X = df.drop(columns=['target']).values
        self.detector.fit(X)
        anomaly_score = -self.detector.score_samples(X)
        trusted = (self.detector.predict(X) == 1).astype(int)
        out = df.copy()
        out['anomaly_score'] = anomaly_score
        out['trusted_packet'] = trusted
        return out

    def train_local(self, df):
        trusted = df[df['trusted_packet'] == 1].copy()
        if len(trusted) < 20:
            trusted = df.copy()
            trusted['trusted_packet'] = 1
        X = trusted.drop(columns=['target', 'anomaly_score', 'trusted_packet']).values
        y = trusted['target'].values
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y if len(np.unique(y)) > 1 else None)
        self.model.fit(X_train, y_train)
        pred = self.model.predict(X_test)
        acc = float(accuracy_score(y_test, pred))
        f1 = float(f1_score(y_test, pred, average='weighted'))
        reliability = float(np.clip((acc + f1) / 2.0, 0.5, 0.99))
        feature_weights = self.model.feature_importances_
        return {
            'weights': feature_weights,
            'n_samples': len(X_train),
            'lambda_k': reliability,
            'accuracy': acc,
            'f1': f1,
            'trusted_rate': float(trusted['trusted_packet'].mean()),
            'mean_anomaly_score': float(trusted['anomaly_score'].mean()) if 'anomaly_score' in trusted else 0.0,
        }
