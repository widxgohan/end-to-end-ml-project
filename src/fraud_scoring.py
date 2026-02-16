"""
Fraud Scoring Engine
==================
Applies trained models to score transactions and assign risk levels.
"""

import pandas as pd
import numpy as np
import logging
import os
import joblib
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/fraud_scoring.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class FraudScoringEngine:
    """
    Enterprise fraud scoring engine.
    """

    def __init__(self, model_path: str = "models/"):
        self.model_path = model_path
        self.model = None
        self.feature_cols = []
        logger.info("Fraud Scoring Engine initialized")

    def load_model(self, model_name: str = "xgboost_fraud_model.pkl") -> bool:
        """
        Load a trained fraud detection model.

        Args:
            model_name: Name of the model file

        Returns:
            True if model loaded successfully
        """
        model_file = os.path.join(self.model_path, model_name)

        try:
            self.model = joblib.load(model_file)
            logger.info(f"Model loaded from {model_file}")
            return True
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False

    def _get_feature_cols(self, df: pd.DataFrame) -> List[str]:
        """Get feature columns for scoring."""
        exclude_cols = [
            'transaction_id', 'timestamp', 'account_id', 'merchant_id',
            'device_id', 'is_fraud', 'merchant_category', 'entry_mode',
            'currency', 'fraud_probability', 'fraud_risk_level'
        ]
        return [c for c in df.columns if c not in exclude_cols]

    def score_transactions(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Score all transactions with fraud probability.

        Args:
            df: DataFrame with transaction features

        Returns:
            DataFrame with fraud_probability added
        """
        if self.model is None:
            logger.warning("No model loaded, using random scores")
            df['fraud_probability'] = np.random.random(len(df))
            return df

        logger.info(f"Scoring {len(df)} transactions...")

        # Prepare features
        feature_cols = self._get_feature_cols(df)
        X = df[feature_cols].copy()

        # Handle non-numeric columns
        for col in X.columns:
            if X[col].dtype == 'object':
                X[col] = pd.factorize(X[col])[0]

        # Handle missing values
        X = X.fillna(0)
        X = X.replace([np.inf, -np.inf], 0)

        # Predict probabilities
        try:
            fraud_proba = self.model.predict_proba(X)[:, 1]
            df['fraud_probability'] = fraud_proba
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            df['fraud_probability'] = 0.0

        logger.info(f"Scoring complete. Avg fraud probability: {df['fraud_probability'].mean():.4f}")

        return df

    def assign_risk_levels(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Assign fraud risk levels based on probability thresholds.

        Args:
            df: DataFrame with fraud_probability

        Returns:
            DataFrame with fraud_risk_level added
        """
        logger.info("Assigning risk levels...")

        def get_risk_level(prob):
            if prob > 0.8:
                return 'HIGH'
            elif prob > 0.4:
                return 'MEDIUM'
            else:
                return 'LOW'

        df['fraud_risk_level'] = df['fraud_probability'].apply(get_risk_level)

        # Log risk distribution
        risk_dist = df['fraud_risk_level'].value_counts()
        logger.info(f"Risk distribution: {risk_dist.to_dict()}")

        return df

    def add_graph_flags(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add graph-based risk flags.

        Args:
            df: DataFrame with transaction data

        Returns:
            DataFrame with additional flags
        """
        logger.info("Adding graph risk flags...")

        # Check if graph features exist
        if 'fraud_ring_flag' not in df.columns:
            df['fraud_ring_flag'] = 0

        if 'shared_device_flag' not in df.columns:
            df['shared_device_flag'] = 0

        # Add graph risk flag based on multiple conditions
        df['graph_risk_flag'] = (
            (df['fraud_ring_flag'] == 1) |
            (df['shared_device_flag'] == 1) |
            (df.get('suspicious_flag', 0) == 1)
        ).astype(int)

        return df

    def calculate_investigation_priority(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate investigation priority score.

        Args:
            df: DataFrame with all features

        Returns:
            DataFrame with investigation_priority
        """
        logger.info("Calculating investigation priority...")

        # Priority formula: combines fraud probability with graph risk
        priority_score = (
            df['fraud_probability'] * 0.6 +
            df.get('graph_risk_flag', 0) * 0.2 +
            df.get('fraud_ring_flag', 0) * 0.2
        )

        df['investigation_priority'] = priority_score

        # Assign priority levels
        def get_priority_level(score):
            if score > 0.7:
                return 'CRITICAL'
            elif score > 0.5:
                return 'HIGH'
            elif score > 0.3:
                return 'MEDIUM'
            else:
                return 'LOW'

        df['priority_level'] = df['investigation_priority'].apply(get_priority_level)

        return df

    def run_scoring_pipeline(self, df: pd.DataFrame,
                            model_name: str = "xgboost_fraud_model.pkl") -> pd.DataFrame:
        """
        Run the complete scoring pipeline.

        Args:
            df: DataFrame with transaction data
            model_name: Model file to use

        Returns:
            Fully scored DataFrame
        """
        logger.info("="*60)
        logger.info("Running Fraud Scoring Pipeline")
        logger.info("="*60)

        # Load model
        self.load_model(model_name)

        # Score transactions
        df = self.score_transactions(df)

        # Assign risk levels
        df = self.assign_risk_levels(df)

        # Add graph flags
        df = self.add_graph_flags(df)

        # Calculate investigation priority
        df = self.calculate_investigation_priority(df)

        # Summary
        logger.info("\n" + "="*60)
        logger.info("Scoring Summary")
        logger.info("="*60)
        logger.info(f"Total transactions: {len(df)}")
        logger.info(f"High risk: {(df['fraud_risk_level'] == 'HIGH').sum()}")
        logger.info(f"Medium risk: {(df['fraud_risk_level'] == 'MEDIUM').sum()}")
        logger.info(f"Low risk: {(df['fraud_risk_level'] == 'LOW').sum()}")
        logger.info(f"Avg fraud probability: {df['fraud_probability'].mean():.4f}")

        return df

    def get_high_risk_transactions(self, df: pd.DataFrame,
                                   threshold: float = 0.5) -> pd.DataFrame:
        """
        Get high risk transactions above threshold.

        Args:
            df: Scored DataFrame
            threshold: Probability threshold

        Returns:
            DataFrame with high risk transactions
        """
        high_risk = df[df['fraud_probability'] >= threshold].copy()
        high_risk = high_risk.sort_values('fraud_probability', ascending=False)

        logger.info(f"Found {len(high_risk)} high risk transactions (threshold: {threshold})")

        return high_risk

    def get_suspicious_communities(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Get community-level fraud statistics.

        Args:
            df: Scored DataFrame with community_id

        Returns:
            DataFrame with community risk stats
        """
        if 'community_id' not in df.columns:
            logger.warning("No community_id column found")
            return pd.DataFrame()

        community_stats = df.groupby('community_id').agg({
            'transaction_id': 'count',
            'fraud_probability': 'mean',
            'is_fraud': 'sum',
            'fraud_ring_flag': 'sum'
        }).rename(columns={
            'transaction_id': 'transaction_count',
            'fraud_probability': 'avg_fraud_prob',
            'is_fraud': 'confirmed_fraud_count',
            'fraud_ring_flag': 'fraud_ring_count'
        })

        community_stats['community_risk_level'] = community_stats['avg_fraud_prob'].apply(
            lambda x: 'HIGH' if x > 0.5 else ('MEDIUM' if x > 0.2 else 'LOW')
        )

        community_stats = community_stats.sort_values('avg_fraud_prob', ascending=False)

        return community_stats.reset_index()


def main():
    """Main execution function."""
    import sys
    sys.path.append('..')

    from src.etl import FraudETLPipeline
    from src.graph_features import GraphFeatureGenerator

    # Run pipeline to get data
    pipeline = FraudETLPipeline("data/raw/transactions.csv", "data/processed/processed_transactions.csv")
    df = pipeline.run_pipeline()

    # Generate graph features
    graph_gen = GraphFeatureGenerator()
    df_with_graph = graph_gen.merge_features_to_transactions(df)

    # Run scoring
    scorer = FraudScoringEngine("models/")

    # If model exists, use it; otherwise simulate
    if os.path.exists("models/xgboost_fraud_model.pkl"):
        df_scored = scorer.run_scoring_pipeline(df_with_graph)
    else:
        logger.info("No trained model found, using simulation")
        df_scored = scorer.run_scoring_pipeline(df_with_graph, "nonexistent.pkl")

    # Get high risk transactions
    high_risk = scorer.get_high_risk_transactions(df_scored, threshold=0.5)
    print(f"\nHigh Risk Transactions: {len(high_risk)}")

    # Get suspicious communities
    communities = scorer.get_suspicious_communities(df_scored)
    print(f"\nTop 5 Risky Communities:")
    print(communities.head())


if __name__ == "__main__":
    main()
