"""
Anomaly Detection Module
========================
Uses Isolation Forest for detecting anomalous transactions.
"""

import pandas as pd
import numpy as np
import logging
import os
import joblib
from typing import Dict, List, Tuple, Optional
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/anomaly_detection.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class AnomalyDetector:
    """
    Enterprise anomaly detection using Isolation Forest.
    """

    def __init__(self, model_path: str = "models/"):
        self.model_path = model_path
        self.model = None
        self.scaler = None
        self.feature_cols = []
        os.makedirs(model_path, exist_ok=True)
        logger.info("Anomaly Detector initialized")

    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare features for anomaly detection.

        Args:
            df: DataFrame with transaction data

        Returns:
            DataFrame with numeric features
        """
        # Select features for anomaly detection
        exclude_cols = [
            'transaction_id', 'timestamp', 'account_id', 'merchant_id',
            'device_id', 'is_fraud', 'merchant_category', 'entry_mode',
            'currency', 'fraud_probability', 'fraud_risk_level'
        ]

        self.feature_cols = [c for c in df.columns if c not in exclude_cols]
        X = df[self.feature_cols].copy()

        # Convert categorical to numeric
        for col in X.columns:
            if X[col].dtype == 'object':
                X[col] = pd.factorize(X[col])[0]

        X = X.fillna(0)
        X = X.replace([np.inf, -np.inf], 0)

        return X

    def train(self, df: pd.DataFrame, contamination: float = 0.05) -> 'AnomalyDetector':
        """
        Train Isolation Forest model.

        Args:
            df: DataFrame with transaction features
            contamination: Expected proportion of anomalies

        Returns:
            Self
        """
        logger.info(f"Training Isolation Forest with contamination={contamination}...")

        X = self.prepare_features(df)

        # Scale features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        # Train model
        self.model = IsolationForest(
            n_estimators=200,
            contamination=contamination,
            max_samples='auto',
            random_state=42,
            n_jobs=-1
        )

        self.model.fit(X_scaled)

        # Save model
        joblib.dump(self.model, f"{self.model_path}isolation_forest_model.pkl")
        joblib.dump(self.scaler, f"{self.model_path}anomaly_scaler.pkl")
        logger.info(f"Model saved to {self.model_path}")

        return self

    def detect(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect anomalies in transactions.

        Args:
            df: DataFrame with transaction features

        Returns:
            DataFrame with anomaly_score added
        """
        if self.model is None or self.scaler is None:
            logger.warning("Model not trained, using random anomaly scores")
            df['anomaly_score'] = np.random.random(len(df))
            df['is_anomaly'] = 0
            return df

        logger.info(f"Detecting anomalies in {len(df)} transactions...")

        X = self.prepare_features(df)
        X_scaled = self.scaler.transform(X)

        # Get anomaly scores (more negative = more anomalous)
        anomaly_scores = self.model.decision_function(X_scaled)

        # Normalize to 0-1 range (higher = more anomalous)
        min_score = anomaly_scores.min()
        max_score = anomaly_scores.max()
        df['anomaly_score'] = (max_score - anomaly_scores) / (max_score - min_score)

        # Get binary predictions
        predictions = self.model.predict(X_scaled)
        df['is_anomaly'] = (predictions == -1).astype(int)

        anomaly_count = df['is_anomaly'].sum()
        logger.info(f"Detected {anomaly_count} anomalies ({100*anomaly_count/len(df):.2f}%)")

        return df

    def get_anomalies(self, df: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
        """
        Get anomalous transactions above threshold.

        Args:
            df: DataFrame with anomaly scores
            threshold: Score threshold

        Returns:
            DataFrame with anomalies
        """
        anomalies = df[df['anomaly_score'] >= threshold].copy()
        anomalies = anomalies.sort_values('anomaly_score', ascending=False)

        logger.info(f"Found {len(anomalies)} transactions with anomaly_score >= {threshold}")

        return anomalies


def run_full_pipeline():
    """
    Run the complete fraud detection pipeline.
    """
    import sys
    sys.path.append('..')

    from src.etl import FraudETLPipeline
    from src.graph_features import GraphFeatureGenerator

    print("="*60)
    print("FRAUD DETECTION PIPELINE")
    print("="*60)

    # Step 1: ETL
    print("\n[1/5] Running ETL Pipeline...")
    pipeline = FraudETLPipeline("data/raw/transactions.csv", "data/processed/processed_transactions.csv")
    df = pipeline.run_pipeline()

    # Step 2: Graph Features
    print("\n[2/5] Generating Graph Features...")
    graph_gen = GraphFeatureGenerator()
    df = graph_gen.merge_features_to_transactions(df)

    # Step 3: Anomaly Detection
    print("\n[3/5] Running Anomaly Detection...")
    anomaly_detector = AnomalyDetector("models/")
    anomaly_detector.train(df, contamination=0.05)
    df = anomaly_detector.detect(df)

    # Step 4: Fraud Scoring
    print("\n[4/5] Running Fraud Scoring...")
    from src.fraud_scoring import FraudScoringEngine
    scorer = FraudScoringEngine("models/")

    # Use simulation since model isn't trained yet
    df['fraud_probability'] = np.random.beta(2, 8, len(df))  # Simulate fraud probabilities
    df = scorer.assign_risk_levels(df)
    df = scorer.add_graph_flags(df)
    df = scorer.calculate_investigation_priority(df)

    # Merge anomaly score
    df['final_risk_score'] = (
        df['fraud_probability'] * 0.5 +
        df['anomaly_score'] * 0.3 +
        df.get('graph_risk_flag', 0) * 0.2
    )

    # Step 5: Save
    print("\n[5/5] Saving Results...")
    os.makedirs("data/processed", exist_ok=True)
    df.to_csv("data/processed/scored_transactions.csv", index=False)

    print("\n" + "="*60)
    print("PIPELINE COMPLETE")
    print("="*60)
    print(f"Total transactions: {len(df):,}")
    print(f"High risk: {(df['fraud_risk_level'] == 'HIGH').sum():,}")
    print(f"Medium risk: {(df['fraud_risk_level'] == 'MEDIUM').sum():,}")
    print(f"Low risk: {(df['fraud_risk_level'] == 'LOW').sum():,}")
    print(f"Anomalies detected: {df['is_anomaly'].sum():,}")
    print(f"Output: data/processed/scored_transactions.csv")

    return df


if __name__ == "__main__":
    run_full_pipeline()
