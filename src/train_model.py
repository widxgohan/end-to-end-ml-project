"""
Fraud Detection ML Model Training
================================
Trains XGBoost and RandomForest models for fraud detection.
"""

import pandas as pd
import numpy as np
import logging
import os
import joblib
from typing import Tuple, Dict, List, Optional
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve
)
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/model_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class FraudModelTrainer:
    """
    Enterprise ML trainer for fraud detection models.
    """

    def __init__(self, model_path: str = "models/"):
        self.model_path = model_path
        self.xgb_model = None
        self.rf_model = None
        self.feature_cols = []
        self.scaler = None
        os.makedirs(model_path, exist_ok=True)
        logger.info("Fraud Model Trainer initialized")

    def prepare_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare features for model training.

        Args:
            df: DataFrame with transaction and graph features

        Returns:
            Tuple of (features, target)
        """
        logger.info("Preparing features for training...")

        # Exclude non-feature columns
        exclude_cols = [
            'transaction_id', 'timestamp', 'account_id', 'merchant_id',
            'device_id', 'is_fraud', 'merchant_category', 'entry_mode',
            'currency', 'fraud_probability', 'fraud_risk_level'
        ]

        # Get feature columns
        self.feature_cols = [c for c in df.columns if c not in exclude_cols]

        # Ensure all features are numeric
        X = df[self.feature_cols].copy()
        for col in X.columns:
            if X[col].dtype == 'object':
                X[col] = pd.factorize(X[col])[0]

        # Handle any remaining issues
        X = X.fillna(0)
        X = X.replace([np.inf, -np.inf], 0)

        y = df['is_fraud'].astype(int)

        logger.info(f"Prepared {len(self.feature_cols)} features for {len(X)} samples")
        logger.info(f"Fraud rate: {y.mean()*100:.2f}%")

        return X, y

    def handle_imbalance(self, X: pd.DataFrame, y: pd.Series,
                        method: str = 'smote') -> Tuple[pd.DataFrame, pd.Series]:
        """
        Handle class imbalance.

        Args:
            X: Features
            y: Target
            method: Method to use ('smote', 'class_weight')

        Returns:
            Balanced features and target
        """
        if method == 'smote':
            logger.info("Applying SMOTE for class imbalance...")
            smote = SMOTE(random_state=42)
            X_resampled, y_resampled = smote.fit_resample(X, y)
            logger.info(f"After SMOTE: {len(X_resampled)} samples (original: {len(X)})")
            return X_resampled, y_resampled
        else:
            return X, y

    def train_xgboost(self, X: pd.DataFrame, y: pd.Series,
                      test_size: float = 0.2) -> Dict:
        """
        Train XGBoost model.

        Args:
            X: Features
            y: Target
            test_size: Test set proportion

        Returns:
            Dictionary with model and metrics
        """
        logger.info("Training XGBoost model...")

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )

        # Calculate scale_pos_weight for imbalanced data
        scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

        # Train XGBoost
        self.xgb_model = XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            n_jobs=-1,
            eval_metric='auc',
            use_label_encoder=False
        )

        self.xgb_model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False
        )

        # Predictions
        y_pred = self.xgb_model.predict(X_test)
        y_pred_proba = self.xgb_model.predict_proba(X_test)[:, 1]

        # Metrics
        metrics = self._calculate_metrics(y_test, y_pred, y_pred_proba)

        # Save model
        joblib.dump(self.xgb_model, f"{self.model_path}xgboost_fraud_model.pkl")
        logger.info(f"XGBoost model saved to {self.model_path}xgboost_fraud_model.pkl")

        return {
            'model': self.xgb_model,
            'metrics': metrics,
            'X_test': X_test,
            'y_test': y_test,
            'y_pred_proba': y_pred_proba
        }

    def train_random_forest(self, X: pd.DataFrame, y: pd.Series,
                            test_size: float = 0.2) -> Dict:
        """
        Train Random Forest model.

        Args:
            X: Features
            y: Target
            test_size: Test set proportion

        Returns:
            Dictionary with model and metrics
        """
        logger.info("Training Random Forest model...")

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )

        # Train Random Forest
        self.rf_model = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )

        self.rf_model.fit(X_train, y_train)

        # Predictions
        y_pred = self.rf_model.predict(X_test)
        y_pred_proba = self.rf_model.predict_proba(X_test)[:, 1]

        # Metrics
        metrics = self._calculate_metrics(y_test, y_pred, y_pred_proba)

        # Save model
        joblib.dump(self.rf_model, f"{self.model_path}random_forest_fraud_model.pkl")
        logger.info(f"Random Forest model saved to {self.model_path}random_forest_fraud_model.pkl")

        return {
            'model': self.rf_model,
            'metrics': metrics,
            'X_test': X_test,
            'y_test': y_test,
            'y_pred_proba': y_pred_proba
        }

    def _calculate_metrics(self, y_true: pd.Series, y_pred: np.ndarray,
                           y_pred_proba: np.ndarray) -> Dict:
        """
        Calculate classification metrics.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Predicted probabilities

        Returns:
            Dictionary of metrics
        """
        metrics = {
            'roc_auc': roc_auc_score(y_true, y_pred_proba),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1': f1_score(y_true, y_pred, zero_division=0),
            'confusion_matrix': confusion_matrix(y_true, y_pred).tolist()
        }

        return metrics

    def cross_validate(self, X: pd.DataFrame, y: pd.Series,
                       model_type: str = 'xgboost', cv: int = 5) -> Dict:
        """
        Perform cross-validation.

        Args:
            X: Features
            y: Target
            model_type: 'xgboost' or 'random_forest'
            cv: Number of folds

        Returns:
            Cross-validation results
        """
        logger.info(f"Running {cv}-fold cross-validation for {model_type}...")

        if model_type == 'xgboost':
            model = XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                use_label_encoder=False,
                eval_metric='auc'
            )
        else:
            model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42
            )

        cv_scores = cross_val_score(model, X, y, cv=cv, scoring='roc_auc')

        results = {
            'cv_scores': cv_scores.tolist(),
            'mean_auc': cv_scores.mean(),
            'std_auc': cv_scores.std()
        }

        logger.info(f"Cross-validation AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

        return results

    def get_feature_importance(self, model_type: str = 'xgboost') -> pd.DataFrame:
        """
        Get feature importance from trained model.

        Args:
            model_type: 'xgboost' or 'random_forest'

        Returns:
            DataFrame with feature importance
        """
        if model_type == 'xgboost' and self.xgb_model:
            importances = self.xgb_model.feature_importances_
        elif model_type == 'random_forest' and self.rf_model:
            importances = self.rf_model.feature_importances_
        else:
            logger.warning("Model not trained yet")
            return pd.DataFrame()

        importance_df = pd.DataFrame({
            'feature': self.feature_cols,
            'importance': importances
        }).sort_values('importance', ascending=False)

        return importance_df

    def train_ensemble(self, X: pd.DataFrame, y: pd.Series,
                       test_size: float = 0.2) -> Dict:
        """
        Train ensemble of models.

        Args:
            X: Features
            y: Target
            test_size: Test set proportion

        Returns:
            Dictionary with all models and metrics
        """
        logger.info("Training ensemble models...")

        results = {}

        # Train XGBoost
        xgb_results = self.train_xgboost(X, y, test_size)
        results['xgboost'] = xgb_results

        # Train Random Forest
        rf_results = self.train_random_forest(X, y, test_size)
        results['random_forest'] = rf_results

        # Compare models
        logger.info("\n" + "="*60)
        logger.info("Model Comparison")
        logger.info("="*60)
        logger.info(f"XGBoost ROC-AUC: {xgb_results['metrics']['roc_auc']:.4f}")
        logger.info(f"Random Forest ROC-AUC: {rf_results['metrics']['roc_auc']:.4f}")

        # Select best model
        if xgb_results['metrics']['roc_auc'] >= rf_results['metrics']['roc_auc']:
            self.best_model = self.xgb_model
            self.best_model_type = 'xgboost'
            logger.info("Selected XGBoost as best model")
        else:
            self.best_model = self.rf_model
            self.best_model_type = 'random_forest'
            logger.info("Selected Random Forest as best model")

        results['best_model'] = self.best_model
        results['best_model_type'] = self.best_model_type

        return results


def main():
    """Main execution function."""
    import sys
    sys.path.append('..')

    from src.etl import FraudETLPipeline
    from src.graph_features import GraphFeatureGenerator

    # Run ETL pipeline
    logger.info("="*60)
    logger.info("Running ETL Pipeline")
    logger.info("="*60)
    pipeline = FraudETLPipeline("data/raw/transactions.csv", "data/processed/processed_transactions.csv")
    df = pipeline.run_pipeline()

    # Generate graph features
    logger.info("="*60)
    logger.info("Generating Graph Features")
    logger.info("="*60)
    graph_gen = GraphFeatureGenerator()
    df_with_graph = graph_gen.merge_features_to_transactions(df)

    # Train models
    logger.info("="*60)
    logger.info("Training Models")
    logger.info("="*60)

    trainer = FraudModelTrainer("models/")
    X, y = trainer.prepare_features(df_with_graph)

    # Handle imbalance with SMOTE
    X_balanced, y_balanced = trainer.handle_imbalance(X, y, method='smote')

    # Train ensemble
    results = trainer.train_ensemble(X_balanced, y_balanced)

    # Feature importance
    importance = trainer.get_feature_importance('xgboost')
    print("\nTop 15 Important Features:")
    print(importance.head(15).to_string(index=False))

    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)


if __name__ == "__main__":
    main()
