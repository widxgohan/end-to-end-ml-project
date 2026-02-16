"""
FastAPI Fraud Detection API
===========================
Real-time fraud scoring API endpoints.
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Dict
import pandas as pd
import numpy as np
import joblib
import os
import logging
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI(
    title="FraudGuard AI API",
    description="Enterprise Fraud Detection & Graph Intelligence API",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Models
class TransactionRequest(BaseModel):
    """Transaction scoring request model."""
    transaction_id: str
    account_id: str
    merchant_id: str
    device_id: str
    amount: float = Field(gt=0)
    timestamp: str
    merchant_category: Optional[str] = "general"
    entry_mode: Optional[str] = "chip"
    hour: Optional[int] = Field(default=12, ge=0, le=23)
    day_of_week: Optional[int] = Field(default=0, ge=0, le=6)


class BatchTransactionRequest(BaseModel):
    """Batch transaction scoring request."""
    transactions: List[TransactionRequest]


class FraudScoreResponse(BaseModel):
    """Fraud score response model."""
    transaction_id: str
    fraud_probability: float
    fraud_risk_level: str
    graph_risk_flag: bool
    fraud_ring_flag: bool
    investigation_priority: float
    priority_level: str


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    timestamp: str
    model_loaded: bool
    data_available: bool


# Global variables
model = None
feature_cols = []


def load_model():
    """Load the trained model."""
    global model, feature_cols

    try:
        model_path = "models/xgboost_fraud_model.pkl"
        if os.path.exists(model_path):
            model = joblib.load(model_path)
            logger.info(f"Model loaded from {model_path}")
        else:
            logger.warning("No trained model found, using simulation")
            model = None
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        model = None


def prepare_features(tx: TransactionRequest) -> pd.DataFrame:
    """Prepare features for a transaction."""
    # Create feature dictionary
    features = {
        'amount': tx.amount,
        'hour': tx.hour,
        'day_of_week': tx.day_of_week,
        'amount_log': np.log1p(tx.amount),
        'is_night': 1 if tx.hour >= 22 or tx.hour <= 5 else 0,
        'is_weekend': 1 if tx.day_of_week >= 5 else 0,
    }

    # Add placeholder graph features
    for col in ['node_degree', 'pagerank_score', 'community_id',
                'fraud_ring_flag', 'shared_device_flag']:
        features[col] = 0

    df = pd.DataFrame([features])
    return df


def simulate_fraud_score(tx: TransactionRequest) -> Dict:
    """Simulate fraud score when no model is available."""
    # Base probability
    prob = 0.05

    # Adjust based on amount (higher amounts = higher risk)
    if tx.amount > 500:
        prob += 0.15
    elif tx.amount > 200:
        prob += 0.08

    # Adjust based on time (night = higher risk)
    if tx.hour >= 22 or tx.hour <= 5:
        prob += 0.1

    # Adjust based on day (weekend = higher risk)
    if tx.day_of_week >= 5:
        prob += 0.05

    # Cap at 0.95
    prob = min(prob, 0.95)

    # Determine risk level
    if prob > 0.8:
        risk_level = "HIGH"
    elif prob > 0.4:
        risk_level = "MEDIUM"
    else:
        risk_level = "LOW"

    # Investigation priority
    priority = prob

    if priority > 0.7:
        priority_level = "CRITICAL"
    elif priority > 0.5:
        priority_level = "HIGH"
    elif priority > 0.3:
        priority_level = "MEDIUM"
    else:
        priority_level = "LOW"

    return {
        'fraud_probability': round(prob, 4),
        'fraud_risk_level': risk_level,
        'graph_risk_flag': prob > 0.5,
        'fraud_ring_flag': False,
        'investigation_priority': round(priority, 4),
        'priority_level': priority_level
    }


@app.on_event("startup")
async def startup_event():
    """Initialize on startup."""
    load_model()
    logger.info("FraudGuard AI API started")


@app.get("/", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        model_loaded=model is not None,
        data_available=os.path.exists("data/processed/processed_transactions.csv")
    )


@app.post("/predict", response_model=FraudScoreResponse)
async def predict_fraud(transaction: TransactionRequest):
    """
    Predict fraud for a single transaction.

    Returns fraud probability and risk assessment.
    """
    try:
        if model is None:
            # Use simulation
            scores = simulate_fraud_score(transaction)
        else:
            # Prepare features
            features = prepare_features(transaction)

            # Get prediction
            prob = model.predict_proba(features)[0][1]

            # Determine risk level
            if prob > 0.8:
                risk_level = "HIGH"
            elif prob > 0.4:
                risk_level = "MEDIUM"
            else:
                risk_level = "LOW"

            scores = {
                'fraud_probability': round(prob, 4),
                'fraud_risk_level': risk_level,
                'graph_risk_flag': prob > 0.5,
                'fraud_ring_flag': False,
                'investigation_priority': round(prob, 4),
                'priority_level': 'HIGH' if prob > 0.5 else ('MEDIUM' if prob > 0.3 else 'LOW')
            }

        return FraudScoreResponse(
            transaction_id=transaction.transaction_id,
            **scores
        )

    except Exception as e:
        logger.error(f"Error predicting fraud: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch")
async def predict_batch(transactions: BatchTransactionRequest,
                        background_tasks: BackgroundTasks):
    """
    Predict fraud for multiple transactions.

    Returns fraud probabilities for all transactions.
    """
    results = []

    for tx in transactions.transactions:
        try:
            if model is None:
                scores = simulate_fraud_score(tx)
            else:
                features = prepare_features(tx)
                prob = model.predict_proba(features)[0][1]

                risk_level = "HIGH" if prob > 0.8 else ("MEDIUM" if prob > 0.4 else "LOW")

                scores = {
                    'fraud_probability': round(prob, 4),
                    'fraud_risk_level': risk_level,
                    'graph_risk_flag': prob > 0.5,
                    'fraud_ring_flag': False,
                    'investigation_priority': round(prob, 4),
                    'priority_level': 'HIGH' if prob > 0.5 else ('MEDIUM' if prob > 0.3 else 'LOW')
                }

            results.append({
                'transaction_id': tx.transaction_id,
                **scores,
                'status': 'success'
            })

        except Exception as e:
            results.append({
                'transaction_id': tx.transaction_id,
                'status': 'error',
                'error': str(e)
            })

    return {
        'total_transactions': len(transactions.transactions),
        'results': results,
        'timestamp': datetime.now().isoformat()
    }


@app.get("/stats/summary")
async def get_summary_stats():
    """Get summary statistics from processed data."""
    try:
        if os.path.exists("data/processed/scored_transactions.csv"):
            df = pd.read_csv("data/processed/scored_transactions.csv")
        elif os.path.exists("data/processed/processed_transactions.csv"):
            df = pd.read_csv("data/processed/processed_transactions.csv")
            # Add simulated scores
            df['fraud_probability'] = np.random.beta(2, 8, len(df))
        else:
            return {
                'error': 'No data available',
                'status': 'no_data'
            }

        # Calculate stats
        stats = {
            'total_transactions': len(df),
            'total_frauds': int(df['is_fraud'].sum()) if 'is_fraud' in df.columns else 0,
            'fraud_rate': round(df['is_fraud'].mean() * 100, 2) if 'is_fraud' in df.columns else 0,
            'avg_fraud_probability': round(df['fraud_probability'].mean() * 100, 2),
            'high_risk_count': int((df['fraud_probability'] > 0.8).sum()) if 'fraud_probability' in df.columns else 0,
            'unique_accounts': int(df['account_id'].nunique()),
            'unique_merchants': int(df['merchant_id'].nunique()),
            'unique_devices': int(df['device_id'].nunique()),
        }

        return stats

    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/transactions/high-risk")
async def get_high_risk_transactions(limit: int = 100):
    """Get high risk transactions."""
    try:
        if os.path.exists("data/processed/scored_transactions.csv"):
            df = pd.read_csv("data/processed/scored_transactions.csv")
        else:
            return {'error': 'No scored data available'}

        # Filter high risk
        high_risk = df[df['fraud_probability'] > 0.5].sort_values(
            'fraud_probability', ascending=False
        ).head(limit)

        # Select relevant columns
        cols = ['transaction_id', 'timestamp', 'amount', 'account_id',
                'merchant_id', 'fraud_probability']

        return {
            'count': len(high_risk),
            'transactions': high_risk[cols].to_dict(orient='records')
        }

    except Exception as e:
        logger.error(f"Error getting high risk transactions: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/retrain")
async def retrain_model():
    """
    Trigger model retraining.
    In production, this would run the full training pipeline.
    """
    try:
        # This would trigger the training pipeline
        # For now, return a placeholder response
        return {
            'status': 'success',
            'message': 'Retraining initiated',
            'timestamp': datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error retraining model: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
