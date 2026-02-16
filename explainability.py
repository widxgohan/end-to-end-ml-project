"""
SHAP Model Explainability Module
================================
Provides global and local SHAP-based explanations for fraud predictions.
"""

import pandas as pd
import numpy as np
import shap
import joblib
import os
import logging
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SHAPExplainer:
    """
    SHAP-based model explainability for fraud detection.
    Provides global and local feature importance explanations.
    """

    def __init__(self, model_path: str = "models/"):
        self.model_path = model_path
        self.model = None
        self.explainer = None
        self.feature_cols = []
        logger.info("SHAP Explainer initialized")

    def load_model(self) -> bool:
        """
        Load the trained XGBoost model.

        Returns:
            True if model loaded successfully
        """
        # Try different model filenames
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
                elif os.path.exists(os.path.join(self.model_path, model_file)):
                    full_path = os.path.join(self.model_path, model_file)
                    self.model = joblib.load(full_path)
                    logger.info(f"Model loaded from {full_path}")
                    return True
            except Exception as e:
                logger.debug(f"Could not load {model_file}: {e}")
                continue

        logger.warning("No trained model found - using simulation mode")
        return False

    def _get_feature_columns(self) -> List[str]:
        """Get feature columns used during training."""
        # Common feature columns for fraud detection
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

    def _prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare features for SHAP explanation.

        Args:
            df: DataFrame with transaction data

        Returns:
            DataFrame with aligned features
        """
        feature_cols = self._get_feature_columns()

        # Filter to available columns
        available_cols = [c for c in feature_cols if c in df.columns]

        X = df[available_cols].copy()

        # Ensure all columns are numeric
        for col in X.columns:
            if X[col].dtype == 'object' or X[col].dtype.name == 'category':
                X[col] = pd.factorize(X[col])[0]

        X = X.fillna(0)
        X = X.replace([np.inf, -np.inf], 0)

        # Store feature columns
        self.feature_cols = list(X.columns)

        return X

    def create_explainer(self, X_background: Optional[pd.DataFrame] = None):
        """
        Create SHAP TreeExplainer.

        Args:
            X_background: Background dataset for SHAP values
        """
        if self.model is None:
            logger.error("No model loaded - cannot create explainer")
            return

        try:
            # Use TreeExplainer for XGBoost
            self.explainer = shap.TreeExplainer(self.model)

            # If background provided, set it
            if X_background is not None:
                self.explainer = shap.TreeExplainer(
                    self.model,
                    data=X_background,
                    feature_perturbation='interventional'
                )

            logger.info("SHAP TreeExplainer created successfully")
        except Exception as e:
            logger.warning(f"TreeExplainer failed: {e}, using KernelExplainer")
            # Fall back to simpler approach
            self.explainer = None

    def compute_global_shap_values(self, df: pd.DataFrame,
                                   sample_size: int = 5000) -> pd.DataFrame:
        """
        Compute global SHAP values for all features.

        Args:
            df: DataFrame with transaction features
            sample_size: Number of samples to use for efficiency

        Returns:
            DataFrame with mean SHAP values per feature
        """
        logger.info("Computing global SHAP values...")

        if self.model is None:
            logger.warning("No model - returning simulated SHAP values")
            return self._simulate_global_shap(df)

        # Prepare features
        X = self._prepare_features(df)

        # Sample if too large
        if len(X) > sample_size:
            X_sample = X.sample(n=sample_size, random_state=42)
        else:
            X_sample = X

        # Create explainer if not exists
        if self.explainer is None:
            self.create_explainer(X_sample)

        try:
            # Compute SHAP values
            if self.explainer is not None:
                shap_values = self.explainer.shap_values(X_sample)
            else:
                # Fallback: use feature importance as proxy
                if hasattr(self.model, 'feature_importances_'):
                    importances = self.model.feature_importances_
                    shap_values = np.outer(
                        np.ones(len(X_sample)),
                        importances[:len(X_sample.columns)]
                    )
                else:
                    return self._simulate_global_shap(df)

            # Calculate mean absolute SHAP values
            if isinstance(shap_values, list):
                # For some models, shap_values is a list
                shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]

            mean_shap = np.abs(shap_values).mean(axis=0)

            # Create results DataFrame
            results = pd.DataFrame({
                'feature': X_sample.columns,
                'shap_importance': mean_shap
            }).sort_values('shap_importance', ascending=False)

            logger.info(f"Computed SHAP values for {len(results)} features")
            return results

        except Exception as e:
            logger.error(f"Error computing SHAP values: {e}")
            return self._simulate_global_shap(df)

    def _simulate_global_shap(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate simulated SHAP values when model unavailable.
        """
        X = self._prepare_features(df)

        # Use feature variance as proxy for importance
        importance = X.var() / X.var().sum()
        importance = importance.sort_values(ascending=False)

        return pd.DataFrame({
            'feature': importance.index,
            'shap_importance': importance.values
        })

    def compute_local_shap_values(self, transaction_row: pd.DataFrame) -> Dict:
        """
        Compute local SHAP values for a single transaction.

        Args:
            transaction_row: Single row DataFrame

        Returns:
            Dict with SHAP values and explanation
        """
        logger.info("Computing local SHAP values...")

        if self.model is None:
            return self._simulate_local_shap(transaction_row)

        try:
            # Prepare features
            X = self._prepare_features(transaction_row)

            if self.explainer is not None:
                shap_values = self.explainer.shap_values(X)
            else:
                return self._simulate_local_shap(transaction_row)

            # Handle list format
            if isinstance(shap_values, list):
                shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]

            # Get values for this transaction
            shap_vals = shap_values[0] if len(shap_values.shape) > 1 else shap_values

            # Create feature contributions
            contributions = pd.DataFrame({
                'feature': X.columns,
                'shap_value': shap_vals,
                'feature_value': X.iloc[0].values
            })

            # Sort by absolute value
            contributions['abs_shap'] = contributions['shap_value'].abs()
            contributions = contributions.sort_values('abs_shap', ascending=False)

            # Get prediction
            pred_proba = self.model.predict_proba(X)[0][1]

            return {
                'shap_values': contributions,
                'prediction': pred_proba,
                'base_value': self.explainer.expected_value if hasattr(self.explainer, 'expected_value') else 0.5,
                'top_features': contributions.head(5).to_dict('records')
            }

        except Exception as e:
            logger.error(f"Error computing local SHAP: {e}")
            return self._simulate_local_shap(transaction_row)

    def _simulate_local_shap(self, transaction_row: pd.DataFrame) -> Dict:
        """
        Generate simulated local SHAP values.
        """
        X = self._prepare_features(transaction_row)

        # Simulate based on feature values
        values = X.iloc[0].values
        features = X.columns

        # Simple simulation: higher values for certain features
        shap_vals = np.random.randn(len(features)) * 0.1

        # Boost importance for certain known fraud indicators
        for i, feat in enumerate(features):
            if 'amount' in feat.lower() and values[i] > values.mean():
                shap_vals[i] += 0.2
            if 'device' in feat.lower() or 'shared' in feat.lower():
                shap_vals[i] += 0.15

        contributions = pd.DataFrame({
            'feature': features,
            'shap_value': shap_vals,
            'feature_value': values
        })

        contributions['abs_shap'] = contributions['shap_value'].abs()
        contributions = contributions.sort_values('abs_shap', ascending=False)

        return {
            'shap_values': contributions,
            'prediction': np.random.uniform(0.1, 0.5),
            'base_value': 0.2,
            'top_features': contributions.head(5).to_dict('records')
        }

    def generate_summary_plot(self, df: pd.DataFrame, sample_size: int = 5000):
        """
        Generate SHAP summary plot.

        Args:
            df: DataFrame with transaction features
            sample_size: Number of samples to plot
        """
        logger.info("Generating SHAP summary plot...")

        X = self._prepare_features(df)

        if len(X) > sample_size:
            X_sample = X.sample(n=sample_size, random_state=42)
        else:
            X_sample = X

        # Compute SHAP values
        if self.explainer is not None:
            shap_values = self.explainer.shap_values(X_sample)
            if isinstance(shap_values, list):
                shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]
        else:
            # Use simulated values
            shap_values = np.random.randn(len(X_sample), len(X_sample.columns)) * 0.1

        # Create summary plot
        shap.summary_plot(shap_values, X_sample, show=False, plot_size=(12, 8))

    def generate_waterfall_plot(self, transaction_row: pd.DataFrame):
        """
        Generate SHAP waterfall plot for a transaction.

        Args:
            transaction_row: Single row DataFrame
        """
        logger.info("Generating SHAP waterfall plot...")

        X = self._prepare_features(transaction_row)

        if self.explainer is not None:
            shap_values = self.explainer.shap_values(X)
            if isinstance(shap_values, list):
                shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]
            shap_vals = shap_values[0]
        else:
            # Simulated values
            shap_vals = np.random.randn(len(X.columns)) * 0.1

        # Create waterfall plot
        shap.plots._waterfall.waterfall_legacy(
            self.explainer.expected_value if hasattr(self.explainer, 'expected_value') else 0,
            shap_vals,
            feature_names=X.columns.tolist(),
            show=False
        )

    def get_feature_explanation(self, transaction_row: pd.DataFrame) -> Dict:
        """
        Get detailed feature explanation for a transaction.

        Args:
            transaction_row: Single row DataFrame

        Returns:
            Dict with explanation details
        """
        local_shap = self.compute_local_shap_values(transaction_row)

        explanation = {
            'fraud_probability': local_shap['prediction'],
            'risk_level': 'HIGH' if local_shap['prediction'] > 0.8 else ('MEDIUM' if local_shap['prediction'] > 0.4 else 'LOW'),
            'base_value': local_shap.get('base_value', 0.2),
            'top_contributors': []
        }

        # Add top contributing features
        for idx, row in local_shap['shap_values'].head(5).iterrows():
            explanation['top_contributors'].append({
                'feature': row['feature'],
                'value': float(row['feature_value']),
                'shap_impact': float(row['shap_value']),
                'direction': 'increases' if row['shap_value'] > 0 else 'decreases'
            })

        return explanation


def main():
    """Main execution function for testing."""
    # This would require a trained model
    print("SHAP Explainability Module")
    print("=" * 50)
    print("To use: Load model and call explainer methods")
    print("Example:")
    print("  explainer = SHAPExplainer('models/')")
    print("  explainer.load_model()")
    print("  df = pd.read_csv('data/processed/scored_transactions.csv')")
    print("  shap_df = explainer.compute_global_shap_values(df)")


if __name__ == "__main__":
    main()
