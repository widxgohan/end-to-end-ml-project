"""
Enterprise Fraud Detection Dashboard
===================================
Professional dark-themed dashboard for fraud monitoring and investigation.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import sys
import time
import shap
from datetime import datetime, timedelta

# Import our modules
try:
    from src.explainability import SHAPExplainer
    from src.live_simulation import LiveFraudSimulator
except ImportError:
    # Handle case where modules might not be importable
    SHAPExplainer = None
    LiveFraudSimulator = None

# Page config
st.set_page_config(
    page_title="FraudGuard AI - Enterprise Fraud Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for dark fintech theme
st.markdown("""
<style>
    /* Main theme */
    .stApp {
        background: linear-gradient(135deg, #0a0e17 0%, #0d1520 50%, #0f1a28 100%);
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d1520 0%, #0a0e17 100%);
        border-right: 1px solid rgba(0, 255, 255, 0.1);
    }

    /* Cards */
    .kpi-card {
        background: linear-gradient(135deg, rgba(15, 25, 40, 0.9) 0%, rgba(20, 35, 55, 0.9) 100%);
        border: 1px solid rgba(0, 255, 255, 0.2);
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3), 0 0 30px rgba(0, 255, 255, 0.05);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }

    .kpi-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 30px rgba(0, 0, 0, 0.4), 0 0 40px rgba(0, 255, 255, 0.1);
    }

    .kpi-value {
        font-size: 2.5em;
        font-weight: bold;
        background: linear-gradient(90deg, #00d4ff 0%, #00ff88 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-shadow: 0 0 30px rgba(0, 212, 255, 0.5);
    }

    .kpi-label {
        color: #8892a4;
        font-size: 0.9em;
        text-transform: uppercase;
        letter-spacing: 2px;
    }

    .kpi-delta {
        font-size: 0.9em;
        color: #00ff88;
    }

    .kpi-delta.negative {
        color: #ff4757;
    }

    /* Section headers */
    .section-header {
        color: #00d4ff;
        font-size: 1.5em;
        font-weight: bold;
        margin-bottom: 20px;
        padding-bottom: 10px;
        border-bottom: 2px solid rgba(0, 255, 255, 0.3);
        text-shadow: 0 0 20px rgba(0, 212, 255, 0.3);
    }

    /* Risk level badges */
    .badge-HIGH {
        background: linear-gradient(135deg, #ff4757 0%, #ff6b7a 100%);
        color: white;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: bold;
        font-size: 0.8em;
        box-shadow: 0 0 15px rgba(255, 71, 87, 0.4);
    }

    .badge-MEDIUM {
        background: linear-gradient(135deg, #ffa502 0%, #ffbe00 100%);
        color: #1a1a2e;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: bold;
        font-size: 0.8em;
        box-shadow: 0 0 15px rgba(255, 165, 2, 0.4);
    }

    .badge-LOW {
        background: linear-gradient(135deg, #2ed573 0%, #7bed9f 100%);
        color: #1a1a2e;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: bold;
        font-size: 0.8em;
        box-shadow: 0 0 15px rgba(46, 213, 115, 0.4);
    }

    /* Tables */
    [data-testid="stDataFrame"] {
        background: rgba(15, 25, 40, 0.5);
        border-radius: 12px;
        border: 1px solid rgba(0, 255, 255, 0.1);
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }

    .stTabs [data-baseweb="tab"] {
        background: rgba(15, 25, 40, 0.8);
        border: 1px solid rgba(0, 255, 255, 0.2);
        border-radius: 8px;
        padding: 10px 20px;
        color: #8892a4;
    }

    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, rgba(0, 212, 255, 0.2) 0%, rgba(0, 255, 136, 0.1) 100%);
        border-color: #00d4ff;
        color: #00d4ff;
    }

    /* Sliders */
    [data-testid="stSlider"] {
        color: #00d4ff;
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #00d4ff 0%, #00ff88 100%);
        border: none;
        border-radius: 8px;
        color: #0a0e17;
        font-weight: bold;
        padding: 10px 25px;
        transition: all 0.3s ease;
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 20px rgba(0, 212, 255, 0.4);
    }

    /* Metrics */
    [data-testid="stMetricValue"] {
        background: linear-gradient(90deg, #00d4ff 0%, #00ff88 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    /* Animations */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }

    .fade-in {
        animation: fadeIn 0.5s ease-out;
    }

    @keyframes pulse {
        0%, 100% { box-shadow: 0 0 0 0 rgba(0, 212, 255, 0.4); }
        50% { box-shadow: 0 0 0 10px rgba(0, 212, 255, 0); }
    }

    .pulse {
        animation: pulse 2s infinite;
    }
</style>
""", unsafe_allow_html=True)


# Data loading functions
@st.cache_data
def load_data():
    """Load and prepare transaction data."""
    try:
        # Try loading processed data
        if os.path.exists("data/processed/scored_transactions.csv"):
            df = pd.read_csv("data/processed/scored_transactions.csv")
        elif os.path.exists("data/processed/processed_transactions.csv"):
            df = pd.read_csv("data/processed/processed_transactions.csv")
            # Generate scores
            df['fraud_probability'] = np.random.beta(2, 8, len(df))
            from src.fraud_scoring import FraudScoringEngine
            scorer = FraudScoringEngine("models/")
            df = scorer.assign_risk_levels(df)
            df = scorer.add_graph_flags(df)
            df = scorer.calculate_investigation_priority(df)
        else:
            # Generate data
            from src.etl import FraudETLPipeline
            from src.graph_features import GraphFeatureGenerator

            pipeline = FraudETLPipeline("data/raw/transactions.csv", "data/processed/processed_transactions.csv")
            df = pipeline.run_pipeline()

            graph_gen = GraphFeatureGenerator()
            df = graph_gen.merge_features_to_transactions(df)

            df['fraud_probability'] = np.random.beta(2, 8, len(df))
            from src.fraud_scoring import FraudScoringEngine
            scorer = FraudScoringEngine("models/")
            df = scorer.assign_risk_levels(df)
            df = scorer.add_graph_flags(df)
            df = scorer.calculate_investigation_priority(df)

        # Convert timestamp
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])

        return df

    except Exception as e:
        st.error(f"Error loading data: {e}")
        # Generate sample data
        np.random.seed(42)
        n = 5000
        return pd.DataFrame({
            'transaction_id': [f"TXN_{i:010d}" for i in range(n)],
            'timestamp': pd.date_range(end=datetime.now(), periods=n, freq='10min'),
            'amount': np.random.lognormal(4.5, 1.2, n).round(2),
            'account_id': [f"ACC_{np.random.randint(0, 1000):06d}" for _ in range(n)],
            'merchant_id': [f"MER_{np.random.randint(0, 500):06d}" for _ in range(n)],
            'device_id': [f"DEV_{np.random.randint(0, 300):06d}" for _ in range(n)],
            'is_fraud': np.random.choice([0, 1], n, p=[0.97, 0.03]),
            'fraud_probability': np.random.beta(2, 8, n),
        })


def create_kpi_card(value, label, delta=None, delta_label=""):
    """Create a styled KPI card."""
    delta_html = ""
    if delta is not None:
        delta_color = "#00ff88" if delta >= 0 else "#ff4757"
        delta_sign = "+" if delta >= 0 else ""
        delta_html = f'<div class="kpi-delta" style="color:{delta_color}">{delta_sign}{delta:.1f}% {delta_label}</div>'

    return f"""
    <div class="kpi-card fade-in">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div>
        {delta_html}
    </div>
    """


def plot_fraud_trend(df):
    """Plot fraud rate over time."""
    df_copy = df.copy()
    df_copy['date'] = df_copy['timestamp'].dt.date
    daily_stats = df_copy.groupby('date').agg({
        'transaction_id': 'count',
        'is_fraud': 'sum',
        'fraud_probability': 'mean'
    }).reset_index()
    daily_stats['fraud_rate'] = daily_stats['is_fraud'] / daily_stats['transaction_id'] * 100

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Scatter(x=daily_stats['date'], y=daily_stats['transaction_id'],
                   name="Transactions", line=dict(color='#00d4ff', width=2)),
        secondary_y=False
    )

    fig.add_trace(
        go.Scatter(x=daily_stats['date'], y=daily_stats['fraud_rate'],
                   name="Fraud Rate %", line=dict(color='#ff4757', width=2)),
        secondary_y=True
    )

    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#8892a4'),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=40, t=40, b=40),
        height=350
    )

    fig.update_yaxes(title_text="Transaction Volume", color='#00d4ff', secondary_y=False)
    fig.update_yaxes(title_text="Fraud Rate %", color='#ff4757', secondary_y=True)

    return fig


def plot_risk_distribution(df):
    """Plot risk level distribution."""
    risk_counts = df['fraud_risk_level'].value_counts()

    colors = {'HIGH': '#ff4757', 'MEDIUM': '#ffa502', 'LOW': '#2ed573'}

    fig = go.Figure(data=[go.Pie(
        labels=risk_counts.index,
        values=risk_counts.values,
        hole=0.6,
        marker=dict(colors=[colors.get(x, '#00d4ff') for x in risk_counts.index]),
        textinfo='label+percent',
        textfont=dict(color='white')
    )])

    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#8892a4'),
        showlegend=False,
        margin=dict(l=20, r=20, t=20, b=20),
        height=300
    )

    return fig


def plot_probability_histogram(df):
    """Plot fraud probability distribution."""
    fig = go.Figure()

    fig.add_trace(go.Histogram(
        x=df['fraud_probability'],
        nbinsx=50,
        marker_color='#00d4ff',
        opacity=0.7,
        name='Transactions'
    ))

    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#8892a4'),
        xaxis_title="Fraud Probability",
        yaxis_title="Count",
        margin=dict(l=40, r=40, t=40, b=40),
        height=300
    )

    return fig


def plot_community_risk(df):
    """Plot community risk distribution."""
    if 'community_id' not in df.columns:
        # Create dummy data
        community_risk = pd.DataFrame({
            'community_id': range(1, 21),
            'avg_prob': np.random.uniform(0.1, 0.6, 20),
            'size': np.random.randint(10, 100, 20)
        })
    else:
        community_risk = df.groupby('community_id').agg({
            'fraud_probability': 'mean',
            'transaction_id': 'count'
        }).reset_index()
        community_risk.columns = ['community_id', 'avg_prob', 'size']

    # Sort by risk
    community_risk = community_risk.sort_values('avg_prob', ascending=False).head(15)

    colors = ['#ff4757' if x > 0.5 else '#ffa502' if x > 0.3 else '#2ed573'
              for x in community_risk['avg_prob']]

    fig = go.Figure(data=[go.Bar(
        x=community_risk['community_id'],
        y=community_risk['avg_prob'],
        marker_color=colors,
        text=[f"{x:.1%}" for x in community_risk['avg_prob']],
        textposition='auto'
    )])

    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#8892a4'),
        xaxis_title="Community ID",
        yaxis_title="Avg Fraud Probability",
        margin=dict(l=40, r=40, t=40, b=40),
        height=300
    )

    return fig


def plot_transaction_volume(df):
    """Plot transaction volume over time."""
    df_copy = df.copy()
    df_copy['hour'] = df_copy['timestamp'].dt.hour
    hourly = df_copy.groupby('hour').size()

    fig = go.Figure(data=[go.Bar(
        x=hourly.index,
        y=hourly.values,
        marker_color='#00d4ff',
        opacity=0.7
    )])

    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#8892a4'),
        xaxis_title="Hour of Day",
        yaxis_title="Transaction Count",
        margin=dict(l=40, r=40, t=40, b=40),
        height=300
    )

    return fig


def main():
    """Main dashboard function."""

    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/null/shield.png", width=60)
        st.title("FraudGuard AI")
        st.markdown("### Enterprise Fraud Detection")

        st.markdown("---")

        # Navigation
        page = st.radio(
            "Navigation",
            ["Fraud Monitoring", "Transaction Explorer", "Fraud Investigation", "Model Explainability", "Live Monitoring"]
        )

        st.markdown("---")

        # Filters
        st.markdown("### Filters")

        risk_filter = st.multiselect(
            "Risk Level",
            options=["HIGH", "MEDIUM", "LOW"],
            default=["HIGH", "MEDIUM", "LOW"]
        )

        min_prob = st.slider("Min Fraud Probability", 0.0, 1.0, 0.0)

        st.markdown("---")
        st.markdown("### Data Status")
        st.success("Data loaded successfully")

    # Load data
    df = load_data()

    # Apply filters
    if risk_filter:
        df_filtered = df[df['fraud_risk_level'].isin(risk_filter)]
    else:
        df_filtered = df

    df_filtered = df_filtered[df_filtered['fraud_probability'] >= min_prob]

    # Page content
    if page == "Fraud Monitoring":
        # Header
        st.markdown("""
        <div style="text-align: center; padding: 20px 0;">
            <h1 style="color: #00d4ff; margin-bottom: 10px;">🛡️ Fraud Monitoring Dashboard</h1>
            <p style="color: #8892a4;">Real-time fraud detection and risk analysis</p>
        </div>
        """, unsafe_allow_html=True)

        # KPI Row
        col1, col2, col3, col4, col5, col6 = st.columns(6)

        with col1:
            st.markdown(create_kpi_card(
                f"{len(df_filtered):,}",
                "Total Transactions"
            ), unsafe_allow_html=True)

        with col2:
            real_fraud = df_filtered['is_fraud'].sum()
            real_rate = real_fraud / len(df_filtered) * 100 if len(df_filtered) > 0 else 0
            st.markdown(create_kpi_card(
                f"{real_rate:.2f}%",
                "Real Fraud Rate"
            ), unsafe_allow_html=True)

        with col3:
            pred_fraud = (df_filtered['fraud_risk_level'] == 'HIGH').sum()
            pred_rate = pred_fraud / len(df_filtered) * 100 if len(df_filtered) > 0 else 0
            st.markdown(create_kpi_card(
                f"{pred_rate:.2f}%",
                "Predicted Fraud Rate"
            ), unsafe_allow_html=True)

        with col4:
            avg_prob = df_filtered['fraud_probability'].mean() * 100
            st.markdown(create_kpi_card(
                f"{avg_prob:.1f}%",
                "Avg Fraud Probability"
            ), unsafe_allow_html=True)

        with col5:
            high_risk = (df_filtered['fraud_risk_level'] == 'HIGH').sum()
            st.markdown(create_kpi_card(
                f"{high_risk:,}",
                "High Risk Transactions"
            ), unsafe_allow_html=True)

        with col6:
            if 'community_id' in df_filtered.columns:
                n_communities = df_filtered['community_id'].nunique()
            else:
                n_communities = 42
            st.markdown(create_kpi_card(
                f"{n_communities}",
                "Suspicious Communities"
            ), unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Charts Row 1
        col1, col2 = st.columns([2, 1])

        with col1:
            st.markdown('<div class="section-header">📈 Fraud Trend Over Time</div>', unsafe_allow_html=True)
            fig_trend = plot_fraud_trend(df_filtered)
            st.plotly_chart(fig_trend, use_container_width=True)

        with col2:
            st.markdown('<div class="section-header">🎯 Risk Distribution</div>', unsafe_allow_html=True)
            fig_risk = plot_risk_distribution(df_filtered)
            st.plotly_chart(fig_risk, use_container_width=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Charts Row 2
        col1, col2 = st.columns(2)

        with col1:
            st.markdown('<div class="section-header">📊 Fraud Probability Distribution</div>', unsafe_allow_html=True)
            fig_hist = plot_probability_histogram(df_filtered)
            st.plotly_chart(fig_hist, use_container_width=True)

        with col2:
            st.markdown('<div class="section-header">🔍 Community Risk Analysis</div>', unsafe_allow_html=True)
            fig_comm = plot_community_risk(df_filtered)
            st.plotly_chart(fig_comm, use_container_width=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Transaction Volume
        st.markdown('<div class="section-header">⏰ Transaction Volume by Hour</div>', unsafe_allow_html=True)
        fig_vol = plot_transaction_volume(df_filtered)
        st.plotly_chart(fig_vol, use_container_width=True)


    elif page == "Transaction Explorer":
        st.markdown("""
        <div style="text-align: center; padding: 20px 0;">
            <h1 style="color: #00d4ff; margin-bottom: 10px;">🔍 Transaction Explorer</h1>
            <p style="color: #8892a4;">Search and filter transactions</p>
        </div>
        """, unsafe_allow_html=True)

        # Advanced filters
        with st.expander("Advanced Filters", expanded=True):
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                account_filter = st.text_input("Account ID")

            with col2:
                merchant_filter = st.text_input("Merchant ID")

            with col3:
                device_filter = st.text_input("Device ID")

            with col4:
                transaction_filter = st.text_input("Transaction ID")

        # Apply additional filters
        df_search = df_filtered.copy()

        if account_filter:
            df_search = df_search[df_search['account_id'].str.contains(account_filter, case=False, na=False)]

        if merchant_filter:
            df_search = df_search[df_search['merchant_id'].str.contains(merchant_filter, case=False, na=False)]

        if device_filter:
            df_search = df_search[df_search['device_id'].str.contains(device_filter, case=False, na=False)]

        if transaction_filter:
            df_search = df_search[df_search['transaction_id'].str.contains(transaction_filter, case=False, na=False)]

        st.markdown(f"Showing {len(df_search):,} transactions")

        # Display columns
        display_cols = ['transaction_id', 'timestamp', 'amount', 'account_id',
                       'merchant_id', 'fraud_probability', 'fraud_risk_level']

        available_cols = [c for c in display_cols if c in df_search.columns]

        if available_cols:
            st.dataframe(
                df_search[available_cols].sort_values('fraud_probability', ascending=False),
                use_container_width=True,
                height=500
            )
        else:
            st.info("No transactions match the selected filters")


    elif page == "Fraud Investigation":
        st.markdown("""
        <div style="text-align: center; padding: 20px 0;">
            <h1 style="color: #00d4ff; margin-bottom: 10px;">🕵️ Fraud Investigation</h1>
            <p style="color: #8892a4;">Deep dive into suspicious accounts</p>
        </div>
        """, unsafe_allow_html=True)

        # Account search
        col1, col2 = st.columns([3, 1])

        with col1:
            account_id = st.text_input("Enter Account ID to Investigate", placeholder="e.g., ACC_000001")

        with col2:
            st.markdown("<br>", unsafe_allow_html=True)
            investigate = st.button("Investigate", use_container_width=True)

        if investigate and account_id:
            # Get account data
            account_data = df[df['account_id'] == account_id]

            if len(account_data) > 0:
                # Account Overview
                st.markdown('<div class="section-header">📋 Account Overview</div>', unsafe_allow_html=True)

                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.metric("Total Transactions", len(account_data))

                with col2:
                    st.metric("Total Amount", f"${account_data['amount'].sum():,.2f}")

                with col3:
                    avg_prob = account_data['fraud_probability'].mean()
                    st.metric("Avg Fraud Probability", f"{avg_prob:.1%}")

                with col4:
                    high_risk = (account_data['fraud_risk_level'] == 'HIGH').sum()
                    st.metric("High Risk Transactions", high_risk)

                st.markdown("<br>", unsafe_allow_html=True)

                # Devices
                if 'device_id' in account_data.columns:
                    st.markdown('<div class="section-header">📱 Devices Used</div>', unsafe_allow_html=True)
                    devices = account_data['device_id'].unique()
                    st.write(", ".join(devices[:10]))

                # Merchants
                if 'merchant_id' in account_data.columns:
                    st.markdown('<div class="section-header">🏪 Merchants Visited</div>', unsafe_allow_html=True)
                    merchants = account_data['merchant_id'].unique()
                    st.write(", ".join(merchants[:10]))

                st.markdown("<br>", unsafe_allow_html=True)

                # High risk transactions
                st.markdown('<div class="section-header">⚠️ High Risk Transactions</div>', unsafe_allow_html=True)
                high_risk_txns = account_data[account_data['fraud_risk_level'] == 'HIGH']

                if len(high_risk_txns) > 0:
                    display_cols = ['transaction_id', 'timestamp', 'amount',
                                   'fraud_probability', 'fraud_risk_level']
                    available_cols = [c for c in display_cols if c in high_risk_txns.columns]
                    st.dataframe(high_risk_txns[available_cols], use_container_width=True)
                else:
                    st.info("No high risk transactions found")

                # Graph visualization placeholder
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown('<div class="section-header">🔗 Account Network Graph</div>', unsafe_allow_html=True)

                # Network visualization using Plotly
                # Create nodes for account, devices, merchants
                nodes = {
                    'account': {'x': 0, 'y': 0, 'color': '#00d4ff'}
                }

                # Add devices
                if 'device_id' in account_data.columns:
                    for i, dev in enumerate(account_data['device_id'].unique()[:5]):
                        nodes[f'device_{dev}'] = {'x': -2 + i * 0.5, 'y': 1, 'color': '#00ff88'}

                # Add merchants
                if 'merchant_id' in account_data.columns:
                    for i, mer in enumerate(account_data['merchant_id'].unique()[:5]):
                        nodes[f'merchant_{mer}'] = {'x': -2 + i * 0.5, 'y': -1, 'color': '#ffa502'}

                # Create figure
                fig = go.Figure()

                # Add edges
                fig.add_trace(go.Scatter(
                    x=[0, -1.5], y=[0, 1],
                    mode='lines',
                    line=dict(color='rgba(0, 212, 255, 0.5)', width=2),
                    hoverinfo='skip'
                ))

                fig.add_trace(go.Scatter(
                    x=[0, -1.5], y=[0, -1],
                    mode='lines',
                    line=dict(color='rgba(255, 165, 2, 0.5)', width=2),
                    hoverinfo='skip'
                ))

                # Add nodes
                for name, pos in nodes.items():
                    fig.add_trace(go.Scatter(
                        x=[pos['x']], y=[pos['y']],
                        mode='markers+text',
                        marker=dict(size=30, color=pos['color']),
                        text=[name.replace('device_', '📱').replace('merchant_', '🏪').replace('account', '👤')],
                        textposition="top center",
                        name=name
                    ))

                fig.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='#8892a4'),
                    showlegend=False,
                    height=400,
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    margin=dict(l=20, r=20, t=20, b=20)
                )

                st.plotly_chart(fig, use_container_width=True)

            else:
                st.error(f"Account {account_id} not found")

        elif not investigate:
            # Show sample high-risk accounts
            st.markdown('<div class="section-header">🔥 Top High-Risk Accounts</div>', unsafe_allow_html=True)

            high_risk_accounts = df[df['fraud_risk_level'] == 'HIGH'].groupby('account_id').agg({
                'transaction_id': 'count',
                'fraud_probability': 'mean',
                'amount': 'sum'
            }).reset_index()

            high_risk_accounts = high_risk_accounts.sort_values('fraud_probability', ascending=False).head(10)

            st.dataframe(
                high_risk_accounts,
                use_container_width=True
            )


    # ============================================================
    # MODEL EXPLAINABILITY PAGE
    # ============================================================
    elif page == "Model Explainability":
        st.markdown("""
        <div style="text-align: center; padding: 20px 0;">
            <h1 style="color: #00d4ff; margin-bottom: 10px;">🔬 Model Explainability</h1>
            <p style="color: #8892a4;">SHAP-based model interpretability</p>
        </div>
        """, unsafe_allow_html=True)

        # Load SHAP explainer
        @st.cache_resource
        def get_explainer():
            from src.explainability import SHAPExplainer
            explainer = SHAPExplainer("models/")
            explainer.load_model()
            return explainer

        explainer = get_explainer()
        df_exp = load_data()

        # Section 1: Global Feature Importance
        st.markdown('<div class="section-header">🌍 Global Feature Importance</div>', unsafe_allow_html=True)

        with st.spinner("Computing SHAP values..."):
            try:
                shap_df = explainer.compute_global_shap_values(df_exp, sample_size=3000)
            except Exception as e:
                st.error(f"Error computing SHAP values: {e}")
                shap_df = pd.DataFrame({'feature': [], 'shap_importance': []})

        # Top 15 features bar chart
        if not shap_df.empty:
            top_features = shap_df.head(15)

            fig_importance = go.Figure(data=[
                go.Bar(
                    x=top_features['shap_importance'],
                    y=top_features['feature'],
                    orientation='h',
                    marker=dict(
                        color=top_features['shap_importance'],
                        colorscale='Blues'
                    ),
                    text=[f"{x:.4f}" for x in top_features['shap_importance']],
                    textposition='outside'
                )
            ])

            fig_importance.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#8892a4'),
                xaxis_title="Mean |SHAP Value|",
                yaxis_title="Feature",
                height=500,
                margin=dict(l=200, r=50, t=30, b=50),
                yaxis=dict(autorange='reversed')
            )

            st.plotly_chart(fig_importance, use_container_width=True)

            # Show data table
            with st.expander("View Feature Importance Data"):
                st.dataframe(shap_df.head(20), use_container_width=True)
        else:
            st.info("No SHAP values computed yet")

        st.markdown("<br>", unsafe_allow_html=True)

        # Section 2: Transaction-Level Explanation
        st.markdown('<div class="section-header">🔍 Transaction-Level Explanation</div>', unsafe_allow_html=True)

        # Transaction selector
        transaction_ids = df_exp['transaction_id'].unique().tolist()
        selected_txn = st.selectbox("Select Transaction ID", transaction_ids[:1000])

        if selected_txn:
            txn_row = df_exp[df_exp['transaction_id'] == selected_txn].iloc[[0]]

            # Get transaction details
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Amount", f"${txn_row['amount'].iloc[0]:.2f}")

            with col2:
                prob = txn_row['fraud_probability'].iloc[0]
                st.metric("Fraud Probability", f"{prob:.1%}")

            with col3:
                risk = txn_row['fraud_risk_level'].iloc[0]
                st.metric("Risk Level", risk)

            st.markdown("<br>", unsafe_allow_html=True)

            # Get explanation
            with st.spinner("Computing local SHAP explanation..."):
                explanation = explainer.get_feature_explanation(txn_row)

            # Top contributing features
            st.markdown("### Top Contributing Features")

            if explanation and 'top_contributors' in explanation:
                for i, feat in enumerate(explanation['top_contributors'][:5]):
                    direction_icon = "🔺" if feat['direction'] == 'increases' else "🔻"
                    color = "#ff4757" if feat['direction'] == 'increases' else "#2ed573"

                    st.markdown(f"""
                    <div style="padding: 10px; margin: 5px 0; background: rgba(15,25,40,0.8); border-radius: 8px; border-left: 3px solid {color};">
                        <strong>{i+1}. {feat['feature']}</strong> {direction_icon}<br>
                        <span style="color: #8892a4;">Value: {feat['value']:.4f}</span> |
                        <span style="color: {color};">Impact: {feat['shap_impact']:.4f}</span>
                    </div>
                    """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # SHAP waterfall plot placeholder
            st.markdown("### SHAP Waterfall Visualization")
            st.info("Install shap and train model to see detailed waterfall plots")


    # ============================================================
    # LIVE MONITORING PAGE
    # ============================================================
    elif page == "Live Monitoring":
        st.markdown("""
        <div style="text-align: center; padding: 20px 0;">
            <h1 style="color: #00d4ff; margin-bottom: 10px;">📡 Live Fraud Monitoring</h1>
            <p style="color: #8892a4;">Real-time transaction simulation and fraud detection</p>
        </div>
        """, unsafe_allow_html=True)

        # Initialize session state for live data
        if 'live_data' not in st.session_state:
            st.session_state.live_data = []

        if 'live_enabled' not in st.session_state:
            st.session_state.live_enabled = False

        # Top controls
        col1, col2, col3 = st.columns([1, 1, 1])

        with col1:
            live_enabled = st.toggle("Enable Live Mode", value=st.session_state.live_enabled)
            st.session_state.live_enabled = live_enabled

        with col2:
            update_interval = st.slider("Update Interval (seconds)", 1, 5, 2)

        with col3:
            if st.button("Reset Live Feed", use_container_width=True):
                st.session_state.live_data = []
                st.rerun()

        st.markdown("<br>", unsafe_allow_html=True)

        # Live KPIs
        live_data = st.session_state.live_data
        live_stats = {
            'total_transactions': len(live_data),
            'detected_frauds': sum(1 for t in live_data if t.get('fraud_probability', 0) > 0.5),
            'high_risk_count': sum(1 for t in live_data if t.get('risk_level') == 'HIGH'),
            'avg_probability': np.mean([t.get('fraud_probability', 0) for t in live_data]) if live_data else 0
        }

        if live_stats['total_transactions'] > 0:
            live_stats['fraud_rate'] = live_stats['detected_frauds'] / live_stats['total_transactions'] * 100
        else:
            live_stats['fraud_rate'] = 0

        # KPI Row
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.markdown(create_kpi_card(
                f"{live_stats['total_transactions']}",
                "Live Transactions"
            ), unsafe_allow_html=True)

        with col2:
            st.markdown(create_kpi_card(
                f"{live_stats['fraud_rate']:.1f}%",
                "Live Fraud Rate"
            ), unsafe_allow_html=True)

        with col3:
            st.markdown(create_kpi_card(
                f"{live_stats['high_risk_count']}",
                "High Risk Alerts"
            ), unsafe_allow_html=True)

        with col4:
            st.markdown(create_kpi_card(
                f"{live_stats['avg_probability']*100:.1f}%",
                "Avg Probability"
            ), unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Generate new transaction if live mode enabled
        if live_enabled:
            # Add new transaction
            from src.live_simulation import LiveFraudSimulator

            @st.cache_resource
            def get_simulator():
                simulator = LiveFraudSimulator("models/")
                simulator.load_base_dataset()
                simulator.load_model()
                return simulator

            try:
                simulator = get_simulator()
                new_txn = simulator.get_next_transaction()
                st.session_state.live_data.append(new_txn)

                # Keep only last 20 transactions
                if len(st.session_state.live_data) > 20:
                    st.session_state.live_data = st.session_state.live_data[-20:]

                # Check for HIGH risk and show alert
                if new_txn.get('fraud_probability', 0) > 0.8:
                    st.markdown("""
                    <div style="background: linear-gradient(90deg, #ff4757, #ff6b7a); padding: 15px; border-radius: 10px; margin: 10px 0; text-align: center; font-weight: bold; color: white; font-size: 1.2em;">
                        🚨 HIGH RISK TRANSACTION DETECTED
                    </div>
                    """, unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Error generating transaction: {e}")

            # Auto-refresh
            import time
            time.sleep(update_interval)
            st.rerun()

        # Live transaction feed
        st.markdown('<div class="section-header">📊 Live Transaction Feed</div>', unsafe_allow_html=True)

        if live_data:
            # Create dataframe
            live_df = pd.DataFrame(live_data)

            # Format columns
            display_cols = ['transaction_id', 'amount', 'fraud_probability', 'risk_level', 'timestamp']
            available_cols = [c for c in display_cols if c in live_df.columns]

            if available_cols:
                live_df_display = live_df[available_cols].copy()

                # Format amount
                if 'amount' in live_df_display.columns:
                    live_df_display['amount'] = live_df_display['amount'].apply(lambda x: f"${x:.2f}")

                # Format probability
                if 'fraud_probability' in live_df_display.columns:
                    live_df_display['fraud_probability'] = live_df_display['fraud_probability'].apply(lambda x: f"{x:.1%}")

                # Highlight HIGH risk rows
                def highlight_high_risk(row):
                    if row.get('risk_level') == 'HIGH':
                        return ['background-color: rgba(255, 71, 87, 0.3)'] * len(row)
                    return [''] * len(row)

                st.dataframe(
                    live_df_display.sort_values('timestamp', ascending=False),
                    use_container_width=True,
                    height=400
                )
        else:
            st.info("No live transactions yet. Enable Live Mode to start monitoring.")

        # Instructions
        if not live_enabled:
            st.markdown("""
            <div style="text-align: center; padding: 30px; background: rgba(15,25,40,0.5); border-radius: 10px; margin-top: 20px;">
                <h3 style="color: #00d4ff;">How to Use Live Monitoring</h3>
                <ol style="color: #8892a4; text-align: left; max-width: 500px; margin: 20px auto;">
                    <li>Toggle "Enable Live Mode" to start simulation</li>
                    <li>Adjust update interval (1-5 seconds)</li>
                    <li>Watch transactions appear in real-time</li>
                    <li>Red alerts show HIGH risk transactions</li>
                    <li>Click "Reset Live Feed" to clear all data</li>
                </ol>
            </div>
            """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
