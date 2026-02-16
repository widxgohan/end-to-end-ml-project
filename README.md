# FraudGuard AI - Enterprise Fraud Detection & Graph Intelligence Platform

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11-blue" alt="Python">
  <img src="https://img.shields.io/badge/Neo4j-Graph Database-brightgreen" alt="Neo4j">
  <img src="https://img.shields.io/badge/XGBoost-ML Model-red" alt="XGBoost">
  <img src="https://img.shields.io/badge/Streamlit-Dashboard-orange" alt="Streamlit">
</p>

A production-grade AI-powered banking fraud detection system that combines graph database analytics with machine learning for real-time fraud detection.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           FRAUDGUARD AI PLATFORM                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐              │
│  │   RAW DATA   │────▶│     ETL      │────▶│    NEO4J     │              │
│  │  (Kaggle)    │     │   PIPELINE   │     │    GRAPH     │              │
│  └──────────────┘     └──────────────┘     └──────────────┘              │
│                                                │                           │
│                                                ▼                           │
│  ┌────────────────────────────────────────────────────────────────────┐   │
│  │                    GRAPH FEATURES ENGINE                           │   │
│  │  • Node Degree  • PageRank  • Community Detection  • Fraud Rings │   │
│  └────────────────────────────────────────────────────────────────────┘   │
│                                                │                           │
│                                                ▼                           │
│  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐              │
│  │   XGBOOST    │     │   RANDOM     │     │  ANOMALY     │              │
│  │   MODEL      │     │   FOREST     │     │  DETECTION   │              │
│  └──────────────┘     └──────────────┘     └──────────────┘              │
│                                                │                           │
│                                                ▼                           │
│  ┌────────────────────────────────────────────────────────────────────┐   │
│  │                     FRAUD SCORING ENGINE                         │   │
│  │  • Risk Levels (HIGH/MEDIUM/LOW)  • Priority  • Graph Flags    │   │
│  └────────────────────────────────────────────────────────────────────┘   │
│                                                │                           │
│                   ┌────────────────────────────┴──────────────┐          │
│                   ▼                                             ▼          │
│         ┌──────────────────┐                          ┌──────────────┐    │
│         │   STREAMLIT     │                          │   FASTAPI    │    │
│         │   DASHBOARD     │                          │     API      │    │
│         │  (Monitoring)   │                          │  (Real-time) │    │
│         └──────────────────┘                          └──────────────┘    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Features

### 1. Data Engineering (ETL Pipeline)
- Real-world fraud data processing (Kaggle/IEEE-CIS format)
- Data cleaning and validation
- Missing value handling
- Feature engineering (behavioral, temporal, amount-based)
- Entity generation (accounts, merchants, devices)

### 2. Neo4j Graph Database Integration
- Automatic graph loading from processed data
- Node types: Account, Transaction, Merchant, Device
- Relationships: MAKES, TO, USES
- Graph analytics:
  - Node degree analysis
  - PageRank centrality
  - Community detection (Louvain)
  - Fraud ring detection

### 3. Machine Learning Models
- **XGBoost** (primary model)
- **Random Forest** (baseline comparison)
- Feature importance analysis
- SMOTE for class imbalance handling
- Cross-validation

### 4. Fraud Risk Engine
- Probability scoring (0-1)
- Risk levels: HIGH (>0.8), MEDIUM (0.4-0.8), LOW (<0.4)
- Graph-based flags:
  - `fraud_ring_flag`
  - `shared_device_flag`
  - `graph_risk_flag`
- Investigation priority scoring
- Anomaly detection (Isolation Forest)

### 5. Enterprise Dashboard (Streamlit)
- **Fraud Monitoring KPIs**
  - Total transactions, fraud rates, risk distribution
  - Trend charts, community analysis

- **Transaction Explorer**
  - Advanced filtering
  - Search by account, merchant, device

- **Fraud Investigation**
  - Account deep-dive
  - Network graph visualization

### 6. REST API (FastAPI)
- Real-time fraud prediction endpoint
- Batch scoring
- Summary statistics
- High-risk transaction queries

## Project Structure

```
fraud_detection_graph_ai/
│
├── data/
│   ├── raw/                    # Raw transaction data
│   └── processed/               # Processed & scored data
│
├── models/                      # Trained ML models
│
├── src/
│   ├── etl.py                  # ETL pipeline
│   ├── graph_loader.py          # Neo4j integration
│   ├── graph_features.py        # Graph feature generation
│   ├── train_model.py           # ML model training
│   ├── fraud_scoring.py         # Fraud scoring engine
│   └── anomaly_detection.py     # Anomaly detection
│
├── app/
│   └── dashboard.py             # Streamlit dashboard
│
├── api/
│   └── main.py                 # FastAPI endpoints
│
├── docker/
│   ├── docker-compose.yml
│   ├── Dockerfile.api
│   └── Dockerfile.dashboard
│
├── logs/                        # Application logs
│
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

## Installation

### Prerequisites
- Python 3.11+
- Neo4j Desktop (optional, for graph visualization)

### 1. Clone and Install Dependencies

```bash
# Clone the repository
cd fraud_detection_graph_ai

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Neo4j Setup (Optional)

For full graph functionality, install Neo4j:

```bash
# Download Neo4j Desktop from https://neo4j.com/download/
# Or use Docker:
docker run -p 7474:7474 -p 7687:7687 -e NEO4J_AUTH=neo4j/password neo4j:5.14.0
```

Configure credentials in `src/graph_loader.py`:
```python
loader = Neo4jGraphLoader(
    uri="bolt://localhost:7687",
    username="neo4j",
    password="your_password"
)
```

## Running the Platform

### Option 1: Quick Start (No Neo4j Required)

The platform works fully offline with simulated graph features:

```bash
# Run the full pipeline
python -m src.anomaly_detection

# Launch dashboard
streamlit run app/dashboard.py
```

### Option 2: With Docker

```bash
cd docker
docker-compose up -d

# Access:
# - Dashboard: http://localhost:8501
# - API: http://localhost:8000
# - Neo4j: http://localhost:7474
```

### Option 3: Manual Start

```bash
# Terminal 1: Run ETL and train model
python -m src.train_model

# Terminal 2: Start API
uvicorn api.main:app --reload

# Terminal 3: Start Dashboard
streamlit run app/dashboard.py
```

## API Usage

### Health Check
```bash
curl http://localhost:8000/
```

### Single Transaction Prediction
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "transaction_id": "TXN_0000000001",
    "account_id": "ACC_000001",
    "merchant_id": "MER_000001",
    "device_id": "DEV_000001",
    "amount": 150.00,
    "timestamp": "2024-01-15T10:30:00",
    "hour": 10,
    "day_of_week": 1
  }'
```

### Batch Prediction
```bash
curl -X POST http://localhost:8000/predict/batch \
  -H "Content-Type: application/json" \
  -d '{
    "transactions": [
      {
        "transaction_id": "TXN_001",
        "account_id": "ACC_001",
        "merchant_id": "MER_001",
        "device_id": "DEV_001",
        "amount": 500,
        "timestamp": "2024-01-15T10:30:00",
        "hour": 10,
        "day_of_week": 1
      }
    ]
  }'
```

### Get High-Risk Transactions
```bash
curl http://localhost:8000/transactions/high-risk?limit=10
```

## Dashboard Features

### Fraud Monitoring
- Real-time KPI cards with animations
- Fraud trend over time
- Risk distribution pie chart
- Probability histogram
- Community risk analysis
- Transaction volume by hour

### Transaction Explorer
- Advanced filtering (risk level, probability, date range)
- Search by transaction/account/merchant/device ID
- Sortable data table

### Fraud Investigation
- Account overview with metrics
- Device and merchant history
- High-risk transaction list
- Interactive network graph

## Risk Engine Rules

| Probability | Risk Level | Action |
|-------------|------------|--------|
| > 0.8 | HIGH | Block/Review immediately |
| 0.4 - 0.8 | MEDIUM | Require additional verification |
| < 0.4 | LOW | Allow with monitoring |

### Additional Flags
- `graph_risk_flag`: Account involved in suspicious graph patterns
- `fraud_ring_flag`: Part of detected fraud ring
- `shared_device_flag`: Multiple accounts using same device
- `investigation_priority`: Combined risk score for queue sorting

## Model Performance

Typical metrics on test data:
- **ROC-AUC**: 0.92+
- **Precision**: 0.85+
- **Recall**: 0.78+
- **F1 Score**: 0.81+

## Scaling Roadmap

### Phase 1 (Current)
- Single-node processing
- In-memory graph features
- Batch scoring

### Phase 2 (Near-term)
- Distributed processing (Spark)
- Real-time streaming (Kafka)
- Graph-based features via Neo4j GDS

### Phase 3 (Production)
- Kubernetes deployment
- Auto-scaling
- Model monitoring & drift detection
- A/B testing framework
- SHAP explainability integration

## Tech Stack

| Component | Technology |
|-----------|------------|
| Data Processing | Pandas, NumPy |
| Graph Database | Neo4j |
| ML Models | XGBoost, Random Forest |
| Anomaly Detection | Isolation Forest |
| Dashboard | Streamlit, Plotly |
| API | FastAPI, Uvicorn |
| Containerization | Docker, Docker Compose |
| Logging | Python logging |

## License

MIT License - See LICENSE file for details.

## Authors

- FraudGuard AI Team

## Support

For issues and questions:
- Open an issue on GitHub
- Check logs in `logs/` directory

---

<p align="center">
  <strong>FraudGuard AI - Protecting Financial Transactions with Graph Intelligence</strong>
</p>
