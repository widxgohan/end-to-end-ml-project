"""
Live Fraud Simulation Module
==========================
Simulates real-time incoming transactions for monitoring.
"""

import pandas as pd
import numpy as np
import joblib
import os
import logging
import random
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class LiveFraudSimulator:
    """
    Real-time fraud simulation for monitoring dashboard.
    """

    def __init__(self, model_path: str = "models/"):
        self.model_path = model_path
        self.model = None
        self.base_dataset = None
        self.feature_cols = []
        logger.info("Live Fraud Simulator initialized")

    def load_model(self) -> bool:
        """
        Load the trained model.

        Returns:
            True if model loaded successfully
        """
        model_files = [
            "xgboost_fraud_model.pkl",
            "fraud_model.pkl",
            "models/xgboost_fraud_model.pkl",
            "models/fraud_model.pkl"
        ]

        for model_file in model_files:
            try:
                if os.path.exists(model_file):
                    self.model = joblib.load(model_file)
                    logger.info(f"Model loaded from {model_file}")
                    return True
            except Exception as e:
                continue

        logger.warning("No trained model found - using simulation mode")
        return False

    def load_base_dataset(self, data_path: str = "data/processed/scored_transactions.csv"):
        """
        Load base dataset for transaction sampling.

        Args:
            data_path: Path to scored transactions
        """
        try:
            if os.path.exists(data_path):
                self.base_dataset = pd.read_csv(data_path, low_memory=False)
                logger.info(f"Base dataset loaded: {len(self.base_dataset)} transactions")
            else:
                logger.warning(f"Dataset not found: {data_path}")
                self.base_dataset = None
        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
            self.base_dataset = None

    def _get_feature_columns(self) -> List[str]:
        """Get feature columns for prediction."""
        return [
            'amount', 'hour', 'day_of_week', 'is_weekend', 'amount_log',
            'is_night', 'high_risk_time', 'high_amount', 'amount_bin',
            'account_txn_count', 'account_amount_mean', 'account_amount_std',
            'account_amount_max', 'amount_vs_account_mean', 'time_since_last_txn',
            'merchant_txn_count', 'device_txn_count', 'unique_merchants_per_device',
            'accounts_per_device', 'shared_device_flag', 'devices_per_account',
            'fraud_count_per_account', 'fraud_rate_per_account',
            'merchant_category_encoded', 'entry_mode_encoded', 'currency_encoded',
            'node_degree', 'total_amount', 'avg_amount', 'std_amount',
            'max_amount', 'min_amount', 'unique_merchants', 'unique_devices',
            'connected_accounts', 'fraud_count', 'fraud_rate',
            'community_id', 'community_size', 'pagerank_score',
            'betweenness_centrality', 'fraud_ring_flag'
        ]

    def generate_random_transaction(self) -> pd.DataFrame:
        """
        Generate a random transaction by sampling and perturbing base data.

        Returns:
            DataFrame with single transaction
        """
        if self.base_dataset is None or len(self.base_dataset) == 0:
            return self._generate_synthetic_transaction()

        try:
            # Random sample from base dataset
            idx = random.randint(0, len(self.base_dataset) - 1)
            row = self.base_dataset.iloc[[idx]].copy()

            # Perturb amount (up to +/- 20%)
            if 'amount' in row.columns:
                perturbation = random.uniform(-0.2, 0.2)
                row['amount'] = row['amount'] * (1 + perturbation)
                row['amount'] = row['amount'].clip(lower=0.01)

                # Update amount_log
                if 'amount_log' in row.columns:
                    row['amount_log'] = np.log1p(row['amount'])

            # Perturb timestamp slightly
            if 'timestamp' in row.columns:
                try:
                    ts = pd.to_datetime(row['timestamp'].iloc[0])
                    # Add random offset up to 1 hour
                    offset = timedelta(minutes=random.randint(-60, 60))
                    row['timestamp'] = [ts + offset]
                    row['hour'] = [(ts + offset).hour]
                except:
                    pass

            # Remove fraud label to simulate real scenario
            if 'is_fraud' in row.columns:
                row = row.drop(columns=['is_fraud'])

            # Remove any existing scoring columns
            cols_to_remove = ['fraud_probability', 'fraud_risk_level',
                            'graph_risk_flag', 'investigation_priority', 'priority_level']
            for col in cols_to_remove:
                if col in row.columns:
                    row = row.drop(columns=[col])

            return row

        except Exception as e:
            logger.error(f"Error generating transaction: {e}")
            return self._generate_synthetic_transaction()

    def _generate_synthetic_transaction(self) -> pd.DataFrame:
        """
        Generate a completely synthetic transaction when no base data available.
        """
        np.random.seed(datetime.now().microsecond)

        # Generate random values
        amount = np.random.lognormal(4.5, 1.2)
        hour = random.randint(0, 23)
        day_of_week = random.randint(0, 6)

        data = {
            'transaction_id': [f"LIVE_{datetime.now().strftime('%Y%m%d%H%M%S')}_{random.randint(1000, 9999)}"],
            'timestamp': [datetime.now()],
            'amount': [round(amount, 2)],
            'account_id': [f"ACC_{random.randint(0, 100):06d}"],
            'merchant_id': [f"MER_{random.randint(0, 50):06d}"],
            'device_id': [f"DEV_{random.randint(0, 30):06d}"],
            'hour': [hour],
            'day_of_week': [day_of_week],
            'is_weekend': [1 if day_of_week >= 5 else 0],
            'amount_log': [np.log1p(amount)],
            'is_night': [1 if hour >= 22 or hour <= 5 else 0],
        }

        return pd.DataFrame(data)

    def _prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare features for prediction.
        """
        feature_cols = self._get_feature_columns()
        available_cols = [c for c in feature_cols if c in df.columns]

        X = df[available_cols].copy()

        for col in X.columns:
            if X[col].dtype == 'object' or X[col].dtype.name == 'category':
                X[col] = pd.factorize(X[col])[0]

        X = X.fillna(0)
        X = X.replace([np.inf, -np.inf], 0)

        return X

    def score_transaction(self, transaction: pd.DataFrame) -> Dict:
        """
        Score a transaction for fraud.

        Args:
            transaction: DataFrame with transaction data

        Returns:
            Dict with prediction results
        """
        try:
            X = self._prepare_features(transaction)

            if self.model is not None:
                # Use actual model
                prob = self.model.predict_proba(X)[0][1]
            else:
                # Simulate fraud probability
                amount = transaction['amount'].iloc[0] if 'amount' in transaction.columns else 100
                hour = transaction['hour'].iloc[0] if 'hour' in transaction.columns else 12

                # Base probability
                prob = np.random.beta(2, 8)

                # Adjust for risk factors
                if amount > 500:
                    prob += 0.15
                if hour >= 22 or hour <= 5:
                    prob += 0.1

                prob = min(prob, 0.95)

            # Determine risk level
            if prob > 0.8:
                risk_level = "HIGH"
            elif prob > 0.4:
                risk_level = "MEDIUM"
            else:
                risk_level = "LOW"

            return {
                'transaction_id': transaction['transaction_id'].iloc[0] if 'transaction_id' in transaction.columns else 'UNKNOWN',
                'timestamp': transaction['timestamp'].iloc[0] if 'timestamp' in transaction.columns else datetime.now(),
                'amount': float(transaction['amount'].iloc[0]) if 'amount' in transaction.columns else 0,
                'account_id': transaction['account_id'].iloc[0] if 'account_id' in transaction.columns else 'UNKNOWN',
                'fraud_probability': round(prob, 4),
                'risk_level': risk_level
            }

        except Exception as e:
            logger.error(f"Error scoring transaction: {e}")
            return {
                'transaction_id': 'ERROR',
                'timestamp': datetime.now(),
                'amount': 0,
                'account_id': 'UNKNOWN',
                'fraud_probability': 0.0,
                'risk_level': 'LOW',
                'error': str(e)
            }

    def get_next_transaction(self) -> Dict:
        """
        Generate and score the next transaction.

        Returns:
            Dict with transaction data and fraud score
        """
        transaction = self.generate_random_transaction()
        result = self.score_transaction(transaction)
        return result

    def get_live_stats(self, live_data: List[Dict]) -> Dict:
        """
        Calculate statistics from live transaction feed.

        Args:
            live_data: List of transaction dicts

        Returns:
            Dict with statistics
        """
        if not live_data:
            return {
                'total_transactions': 0,
                'fraud_rate': 0.0,
                'high_risk_count': 0,
                'avg_probability': 0.0
            }

        total = len(live_data)
        high_risk = sum(1 for t in live_data if t.get('risk_level') == 'HIGH')
        avg_prob = np.mean([t.get('fraud_probability', 0) for t in live_data])

        # Calculate detected frauds (simulated)
        detected_frauds = sum(1 for t in live_data if t.get('fraud_probability', 0) > 0.5)

        return {
            'total_transactions': total,
            'detected_frauds': detected_frauds,
            'fraud_rate': round(detected_frauds / total * 100, 2) if total > 0 else 0,
            'high_risk_count': high_risk,
            'avg_probability': round(avg_prob * 100, 2)
        }


def main():
    """Main execution for testing."""
    print("Live Fraud Simulator")
    print("=" * 50)

    simulator = LiveFraudSimulator("models/")
    simulator.load_base_dataset()

    # Generate sample transaction
    transaction = simulator.generate_random_transaction()
    print(f"Generated transaction: {transaction['transaction_id'].iloc[0]}")

    # Score it
    result = simulator.score_transaction(transaction)
    print(f"Fraud probability: {result['fraud_probability']:.4f}")
    print(f"Risk level: {result['risk_level']}")


if __name__ == "__main__":
    main()
