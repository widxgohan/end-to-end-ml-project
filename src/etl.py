"""
ETL Pipeline for Fraud Detection Platform
==========================================
Handles data ingestion, cleaning, feature engineering, and entity generation.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import os
import hashlib
from typing import Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/etl.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class FraudETLPipeline:
    """
    Enterprise ETL Pipeline for Fraud Detection Data Processing.
    """

    def __init__(self, raw_data_path: str, output_path: str):
        self.raw_data_path = raw_data_path
        self.output_path = output_path
        self.df = None
        logger.info(f"ETL Pipeline initialized with raw path: {raw_data_path}")

    def load_data(self) -> pd.DataFrame:
        """Load raw transaction data."""
        logger.info("Loading raw data...")

        try:
            # Try loading from CSV
            self.df = pd.read_csv(self.raw_data_path)
            logger.info(f"Loaded {len(self.df)} records from CSV")

            # Check if it's the Kaggle Credit Card dataset
            if 'V1' in self.df.columns and 'Class' in self.df.columns:
                logger.info("Detected Kaggle Credit Card Fraud dataset - transforming to standard format")
                self.df = self._transform_kaggle_creditcard(self.df)
        except Exception as e:
            logger.warning(f"CSV not found, generating synthetic data: {e}")
            self.df = self._generate_synthetic_data()

        return self.df

    def _transform_kaggle_creditcard(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform Kaggle Credit Card dataset to standard format.
        Adds synthetic entity IDs (account, merchant, device) for graph features.
        """
        logger.info("Transforming Kaggle Credit Card dataset...")

        n = len(df)

        # Convert Time to timestamp
        start_date = datetime.now() - timedelta(days=30)
        df['timestamp'] = [start_date + timedelta(seconds=int(t)) for t in df['Time']]

        # Rename Amount
        df['amount'] = df['Amount']

        # Rename Class to is_fraud
        df['is_fraud'] = df['Class']

        # Generate entity IDs based on V-features and Amount
        # This creates meaningful relationships for graph features
        np.random.seed(42)

        # Create account IDs based on V1 clustering (simulates real customers)
        account_clusters = pd.qcut(df['V1'], q=min(100, n//100), labels=False, duplicates='drop')
        df['account_id'] = [f"ACC_{int(a):06d}" for a in account_clusters]

        # Create merchant IDs based on V2-V3 clustering
        merchant_clusters = pd.qcut(df['V2'], q=min(50, n//200), labels=False, duplicates='drop')
        df['merchant_id'] = [f"MER_{int(m):06d}" for m in merchant_clusters]

        # Create device IDs based on V4 clustering
        device_clusters = pd.qcut(df['V4'], q=min(30, n//500), labels=False, duplicates='drop')
        df['device_id'] = [f"DEV_{int(d):06d}" for d in device_clusters]

        # Add transaction ID
        df['transaction_id'] = [f"TXN_{i:010d}" for i in range(n)]

        # Add merchant category based on V5
        df['merchant_category'] = pd.cut(df['V5'], bins=8,
                                        labels=['grocery', 'electronics', 'restaurant',
                                               'gas_station', 'online_shopping', 'travel',
                                               'entertainment', 'healthcare'])

        # Add entry mode based on V6
        df['entry_mode'] = pd.cut(df['V6'], bins=3,
                                 labels=['chip', 'swipe', 'online'])

        # Add card present (random with bias towards True)
        df['card_present'] = np.random.choice([0, 1], n, p=[0.25, 0.75])

        # Add currency (mostly USD)
        df['currency'] = np.random.choice(['USD', 'EUR', 'GBP'], n, p=[0.85, 0.1, 0.05])

        # Add hour and day_of_week from timestamp
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)

        logger.info(f"Transformed {n} records to standard format")
        logger.info(f"Unique accounts: {df['account_id'].nunique()}")
        logger.info(f"Unique merchants: {df['merchant_id'].nunique()}")
        logger.info(f"Unique devices: {df['device_id'].nunique()}")
        logger.info(f"Fraud rate: {df['is_fraud'].mean()*100:.2f}%")

        return df

    def _generate_synthetic_data(self, n_transactions: int = 50000) -> pd.DataFrame:
        """
        Generate realistic synthetic fraud data with entity relationships.
        This mimics the IEEE-CIS Fraud Detection dataset structure.
        """
        logger.info(f"Generating {n_transactions} synthetic transactions...")

        np.random.seed(42)

        # Generate dates for 6 months
        end_date = datetime.now()
        start_date = end_date - timedelta(days=180)
        timestamps = [
            start_date + timedelta(seconds=np.random.randint(0, int((end_date - start_date).total_seconds())))
            for _ in range(n_transactions)
        ]
        timestamps.sort()

        # Generate entities
        n_accounts = n_transactions // 20  # ~20 tx per account
        n_merchants = n_transactions // 50  # ~50 tx per merchant
        n_devices = n_transactions // 30  # ~30 tx per device

        # Create accounts
        accounts = [f"ACC_{i:06d}" for i in range(n_accounts)]

        # Create merchants with categories
        merchant_categories = ['grocery', 'electronics', 'restaurant', 'gas_station',
                               'online_shopping', 'travel', 'entertainment', 'healthcare']
        merchants = [f"MER_{i:06d}" for i in range(n_merchants)]

        # Create devices
        devices = [f"DEV_{i:06d}" for i in range(n_devices)]

        # Transaction amounts - log-normal distribution
        amounts = np.random.lognormal(mean=4.5, sigma=1.2, size=n_transactions)
        amounts = np.round(amounts, 2)

        # Assign entities randomly but with some patterns
        account_ids = np.random.choice(accounts, n_transactions)
        merchant_ids = np.random.choice(merchants, n_transactions)
        device_ids = np.random.choice(devices, n_transactions)

        # Generate fraud labels (approximately 3% fraud rate)
        fraud_probs = np.random.random(n_transactions)
        is_fraud = (fraud_probs < 0.03).astype(int)

        # Make some high-amount transactions more likely to be fraud
        high_amount_mask = amounts > amounts.mean() + 2 * amounts.std()
        is_fraud[high_amount_mask] = np.random.choice([0, 1], size=high_amount_mask.sum(),
                                                       p=[0.7, 0.3])

        # Create dataframe
        df = pd.DataFrame({
            'transaction_id': [f"TXN_{i:010d}" for i in range(n_transactions)],
            'timestamp': timestamps,
            'amount': amounts,
            'account_id': account_ids,
            'merchant_id': merchant_ids,
            'device_id': device_ids,
            'is_fraud': is_fraud,
            'merchant_category': np.random.choice(merchant_categories, n_transactions),
        })

        # Add hour of day (fraud patterns by time)
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)

        # Add card present flag
        df['card_present'] = np.random.choice([0, 1], n_transactions, p=[0.3, 0.7])

        # Add entry mode
        df['entry_mode'] = np.random.choice(['chip', 'swipe', 'online'], n_transactions,
                                             p=[0.5, 0.2, 0.3])

        # Add transaction currency (mostly USD)
        df['currency'] = np.random.choice(['USD', 'EUR', 'GBP'], n_transactions,
                                           p=[0.85, 0.1, 0.05])

        logger.info(f"Generated {len(df)} transactions with {is_fraud.sum()} frauds ({100*is_fraud.mean():.2f}%)")

        return df

    def clean_data(self) -> pd.DataFrame:
        """Clean and validate data."""
        logger.info("Cleaning data...")

        if self.df is None:
            raise ValueError("No data loaded. Call load_data() first.")

        initial_count = len(self.df)

        # Remove duplicates
        self.df = self.df.drop_duplicates(subset=['transaction_id'])

        # Handle missing values
        for col in self.df.columns:
            if self.df[col].isnull().any():
                if self.df[col].dtype in ['object']:
                    self.df[col] = self.df[col].fillna('UNKNOWN')
                else:
                    self.df[col] = self.df[col].fillna(self.df[col].median())

        # Validate amounts
        self.df = self.df[self.df['amount'] > 0]

        # Validate timestamps
        self.df = self.df[self.df['timestamp'] < datetime.now()]

        logger.info(f"Cleaned data: {initial_count} -> {len(self.df)} records")

        return self.df

    def engineer_features(self) -> pd.DataFrame:
        """Engineer behavioral and derived features."""
        logger.info("Engineering features...")

        if self.df is None:
            raise ValueError("No data loaded. Call load_data() first.")

        df = self.df.copy()

        # Time-based features
        df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
        df['day_of_week'] = pd.to_datetime(df['timestamp']).dt.dayofweek
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        df['is_night'] = ((df['hour'] >= 22) | (df['hour'] <= 5)).astype(int)
        df['month'] = pd.to_datetime(df['timestamp']).dt.month
        df['day_of_month'] = pd.to_datetime(df['timestamp']).dt.day

        # Amount-based features
        df['amount_log'] = np.log1p(df['amount'])
        df['amount_bin'] = pd.cut(df['amount'],
                                  bins=[0, 10, 50, 100, 500, 1000, float('inf')],
                                  labels=[0, 1, 2, 3, 4, 5]).astype(int)

        # Risk time flag
        df['high_risk_time'] = ((df['is_night'] == 1) | (df['is_weekend'] == 1)).astype(int)

        # Amount risk flag
        df['high_amount'] = (df['amount'] > df['amount'].quantile(0.95)).astype(int)

        # Velocity features (computed per account)
        logger.info("Computing velocity features...")

        df = df.sort_values(['account_id', 'timestamp'])

        # Transaction frequency per account
        account_counts = df.groupby('account_id').size().to_dict()
        df['account_txn_count'] = df['account_id'].map(account_counts)

        # Amount statistics per account
        account_amount_stats = df.groupby('account_id')['amount'].agg(['mean', 'std', 'max']).to_dict()
        df['account_amount_mean'] = df['account_id'].map(account_amount_stats['mean'])
        df['account_amount_std'] = df['account_id'].map(account_amount_stats['std']).fillna(0)
        df['account_amount_max'] = df['account_id'].map(account_amount_stats['max'])

        # Deviation from account mean
        df['amount_vs_account_mean'] = (df['amount'] - df['account_amount_mean']) / (df['account_amount_std'] + 1)

        # Time since last transaction (per account)
        df['time_since_last_txn'] = df.groupby('account_id')['timestamp'].diff().dt.total_seconds() / 3600

        # Merchant-based features
        merchant_counts = df.groupby('merchant_id').size().to_dict()
        df['merchant_txn_count'] = df['merchant_id'].map(merchant_counts)

        # Device-based features
        device_counts = df.groupby('device_id').size().to_dict()
        df['device_txn_count'] = df['device_id'].map(device_counts)

        # Cross-entity features
        # Unique merchants per device
        merchant_per_device = df.groupby('device_id')['merchant_id'].nunique().to_dict()
        df['unique_merchants_per_device'] = df['device_id'].map(merchant_per_device)

        # Unique accounts per device
        accounts_per_device = df.groupby('device_id')['account_id'].nunique().to_dict()
        df['accounts_per_device'] = df['device_id'].map(accounts_per_device)

        # Flag for shared device (potential fraud indicator)
        df['shared_device_flag'] = (df['accounts_per_device'] > 3).astype(int)

        # Unique devices per account
        devices_per_account = df.groupby('account_id')['device_id'].nunique().to_dict()
        df['devices_per_account'] = df['account_id'].map(devices_per_account)

        # Risk aggregations
        df['fraud_count_per_account'] = df.groupby('account_id')['is_fraud'].transform('sum')
        df['fraud_rate_per_account'] = df['fraud_count_per_account'] / df['account_txn_count']

        logger.info(f"Engineered {len([c for c in df.columns if c not in self.df.columns])} new features")

        return df

    def encode_categorical(self) -> pd.DataFrame:
        """Encode categorical features."""
        logger.info("Encoding categorical features...")

        if self.df is None:
            raise ValueError("No data loaded. Call load_data() first.")

        df = self.df.copy()

        # Encode merchant_category
        category_mapping = {cat: i for i, cat in enumerate(df['merchant_category'].unique())}
        df['merchant_category_encoded'] = df['merchant_category'].map(category_mapping)

        # Encode entry_mode
        entry_mapping = {mode: i for i, mode in enumerate(df['entry_mode'].unique())}
        df['entry_mode_encoded'] = df['entry_mode'].map(entry_mapping)

        # Encode currency
        currency_mapping = {curr: i for i, curr in enumerate(df['currency'].unique())}
        df['currency_encoded'] = df['currency'].map(currency_mapping)

        logger.info("Categorical encoding complete")

        return df

    def scale_features(self, feature_cols: list) -> pd.DataFrame:
        """Scale numerical features."""
        logger.info("Scaling features...")

        from sklearn.preprocessing import StandardScaler

        df = self.df.copy()

        scaler = StandardScaler()
        df[feature_cols] = scaler.fit_transform(df[feature_cols])

        return df

    def run_pipeline(self) -> pd.DataFrame:
        """Execute the complete ETL pipeline."""
        logger.info("=" * 60)
        logger.info("Starting ETL Pipeline")
        logger.info("=" * 60)

        # Step 1: Load data
        self.load_data()

        # Step 2: Clean data
        self.clean_data()

        # Step 3: Engineer features
        self.engineer_features()

        # Step 4: Encode categorical
        self.encode_categorical()

        # Step 5: Save processed data
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        self.df.to_csv(self.output_path, index=False)

        logger.info(f"Pipeline complete. Saved to: {self.output_path}")
        logger.info(f"Final dataset shape: {self.df.shape}")
        logger.info(f"Fraud rate: {self.df['is_fraud'].mean()*100:.2f}%")

        return self.df


def main():
    """Main execution function."""
    # Configuration
    RAW_DATA_PATH = "data/raw/transactions.csv"
    OUTPUT_PATH = "data/processed/processed_transactions.csv"

    # Create logs directory
    os.makedirs("logs", exist_ok=True)

    # Run pipeline
    pipeline = FraudETLPipeline(RAW_DATA_PATH, OUTPUT_PATH)
    df = pipeline.run_pipeline()

    print(f"\n{'='*60}")
    print("ETL Pipeline Summary")
    print(f"{'='*60}")
    print(f"Total transactions: {len(df):,}")
    print(f"Fraudulent transactions: {df['is_fraud'].sum():,}")
    print(f"Fraud rate: {df['is_fraud'].mean()*100:.2f}%")
    print(f"Features: {len(df.columns)}")
    print(f"Output: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
