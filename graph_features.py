"""
Graph Features Generator
========================
Generates graph-based features for ML model from transaction data.
"""

import pandas as pd
import numpy as np
import logging
import os
from typing import Dict, List, Optional
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/graph_features.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class GraphFeatureGenerator:
    """
    Generates graph-based features from transaction data.
    Works offline (no Neo4j required) using in-memory graph construction.
    """

    def __init__(self):
        self.graph = defaultdict(lambda: {
            'transactions': [],
            'merchants': set(),
            'devices': set(),
            'connected_accounts': set()
        })
        logger.info("Graph Feature Generator initialized")

    def build_graph(self, df: pd.DataFrame):
        """
        Build in-memory graph from transaction DataFrame.

        Args:
            df: DataFrame with transaction data
        """
        logger.info(f"Building graph from {len(df)} transactions...")

        # Build account-level graph
        for _, row in df.iterrows():
            account_id = row['account_id']
            merchant_id = row['merchant_id']
            device_id = row['device_id']
            transaction_id = row['transaction_id']

            self.graph[account_id]['transactions'].append({
                'transaction_id': transaction_id,
                'amount': row['amount'],
                'timestamp': row['timestamp'],
                'is_fraud': row.get('is_fraud', 0)
            })
            self.graph[account_id]['merchants'].add(merchant_id)
            self.graph[account_id]['devices'].add(device_id)

        # Build cross-account connections (shared devices)
        logger.info("Computing cross-account connections...")
        device_to_accounts = defaultdict(set)
        for account_id, data in self.graph.items():
            for device_id in data['devices']:
                device_to_accounts[device_id].add(account_id)

        # Connect accounts via shared devices
        for device_id, accounts in device_to_accounts.items():
            accounts_list = list(accounts)
            for i, acc1 in enumerate(accounts_list):
                for acc2 in accounts_list[i+1:]:
                    self.graph[acc1]['connected_accounts'].add(acc2)
                    self.graph[acc2]['connected_accounts'].add(acc1)

        logger.info(f"Built graph with {len(self.graph)} accounts")

    def compute_node_features(self) -> pd.DataFrame:
        """
        Compute node-level features for each account.

        Returns:
            DataFrame with graph features
        """
        logger.info("Computing node features...")

        features = []

        for account_id, data in self.graph.items():
            txns = data['transactions']
            txn_count = len(txns)

            # Basic metrics
            amounts = [t['amount'] for t in txns]
            fraud_count = sum(t['is_fraud'] for t in txns)

            features.append({
                'account_id': account_id,
                'node_degree': txn_count,
                'total_amount': sum(amounts),
                'avg_amount': np.mean(amounts) if amounts else 0,
                'std_amount': np.std(amounts) if len(amounts) > 1 else 0,
                'max_amount': max(amounts) if amounts else 0,
                'min_amount': min(amounts) if amounts else 0,
                'unique_merchants': len(data['merchants']),
                'unique_devices': len(data['devices']),
                'connected_accounts': len(data['connected_accounts']),
                'fraud_count': fraud_count,
                'fraud_rate': fraud_count / txn_count if txn_count > 0 else 0
            })

        df = pd.DataFrame(features)
        logger.info(f"Computed features for {len(df)} accounts")

        return df

    def compute_community_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Perform community detection using connected components.

        Args:
            df: DataFrame with account data

        Returns:
            DataFrame with community assignments
        """
        logger.info("Computing community features...")

        # Build adjacency for community detection
        adjacency = defaultdict(set)
        for account_id, data in self.graph.items():
            for connected in data['connected_accounts']:
                adjacency[account_id].add(connected)

        # Union-Find for connected components
        parent = {}

        def find(x):
            if x not in parent:
                parent[x] = x
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]

        def union(x, y):
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py

        # Union connected accounts
        for account_id, neighbors in adjacency.items():
            for neighbor in neighbors:
                union(account_id, neighbor)

        # Assign community IDs
        communities = {}
        for account_id in self.graph.keys():
            comm_id = find(account_id)
            if comm_id not in communities:
                communities[comm_id] = len(communities)
            communities[account_id] = communities[comm_id]

        # Convert to DataFrame
        community_df = pd.DataFrame([
            {'account_id': k, 'community_id': v}
            for k, v in communities.items()
        ])

        # Compute community-level stats
        community_stats = community_df.groupby('community_id').size().to_dict()
        community_df['community_size'] = community_df['community_id'].map(community_stats)

        logger.info(f"Found {len(set(communities.values()))} communities")

        return community_df

    def detect_fraud_rings(self) -> pd.DataFrame:
        """
        Detect potential fraud rings (accounts connected via shared devices
        with similar transaction patterns).

        Returns:
            DataFrame with fraud ring information
        """
        logger.info("Detecting fraud rings...")

        fraud_rings = []

        # Find accounts that share devices and have similar merchant/amount patterns
        for account_id, data in self.graph.items():
            if len(data['connected_accounts']) == 0:
                continue

            connected = list(data['connected_accounts'])[:10]  # Limit checks

            for connected_id in connected:
                if connected_id not in self.graph:
                    continue

                other_data = self.graph[connected_id]

                # Check for shared merchants
                shared_merchants = data['merchants'] & other_data['merchants']

                # Check for amount similarity
                amounts1 = [t['amount'] for t in data['transactions']]
                amounts2 = [t['amount'] for t in other_data['transactions']]

                if len(amounts1) > 0 and len(amounts2) > 0:
                    avg1, avg2 = np.mean(amounts1), np.mean(amounts2)
                    amount_similarity = 1 - min(abs(avg1 - avg2) / max(avg1, avg2, 1), 1)

                    # Flag as potential fraud ring
                    if len(shared_merchants) >= 2 and amount_similarity > 0.7:
                        fraud_rings.append({
                            'account1': account_id,
                            'account2': connected_id,
                            'shared_merchants': len(shared_merchants),
                            'amount_similarity': amount_similarity,
                            'ring_size': len(data['connected_accounts']) + len(other_data['connected_accounts'])
                        })

        df = pd.DataFrame(fraud_rings)
        logger.info(f"Detected {len(df)} potential fraud rings")

        return df

    def compute_pagerank(self, alpha: float = 0.85, max_iter: int = 100) -> pd.DataFrame:
        """
        Compute PageRank scores for accounts.

        Args:
            alpha: Damping factor
            max_iter: Maximum iterations

        Returns:
            DataFrame with PageRank scores
        """
        logger.info("Computing PageRank scores...")

        accounts = list(self.graph.keys())
        n = len(accounts)
        account_idx = {acc: i for i, acc in enumerate(accounts)}

        # Build transition matrix
        pagerank = np.ones(n) / n
        matrix = np.zeros((n, n))

        for account_id, data in self.graph.items():
            i = account_idx[account_id]
            neighbors = list(data['connected_accounts'])

            if len(neighbors) > 0:
                for neighbor in neighbors:
                    if neighbor in account_idx:
                        j = account_idx[neighbor]
                        matrix[j, i] = 1 / len(neighbors)

        # Power iteration
        for _ in range(max_iter):
            new_pagerank = (1 - alpha) / n + alpha * matrix @ pagerank
            if np.allclose(new_pagerank, pagerank, atol=1e-6):
                break
            pagerank = new_pagerank

        # Convert to DataFrame
        df = pd.DataFrame([
            {'account_id': acc, 'pagerank_score': pagerank[account_idx[acc]]}
            for acc in accounts
        ])

        logger.info("PageRank computation complete")

        return df

    def compute_betweenness(self) -> pd.DataFrame:
        """
        Compute betweenness centrality for accounts.

        Returns:
            DataFrame with betweenness scores
        """
        logger.info("Computing betweenness centrality...")

        accounts = list(self.graph.keys())
        betweenness = {acc: 0.0 for acc in accounts}

        # Simplified betweenness (number of shortest paths through each node)
        for source in accounts:
            for target in accounts:
                if source == target:
                    continue

                # BFS to find shortest paths
                queue = [source]
                visited = {source: [source]}
                levels = {source: 0}

                while queue:
                    current = queue.pop(0)
                    if current == target:
                        break

                    current_level = levels[current]
                    neighbors = list(self.graph[current]['connected_accounts'])

                    for neighbor in neighbors:
                        if neighbor not in levels:
                            levels[neighbor] = current_level + 1
                            visited[neighbor] = visited[current] + [neighbor]
                            queue.append(neighbor)
                        elif levels[neighbor] == current_level + 1:
                            visited[neighbor] = visited[current] + [neighbor]

                # Count intermediate nodes
                if target in visited:
                    path = visited[target]
                    for node in path[1:-1]:
                        betweenness[node] += 1

        # Normalize
        max_betweenness = max(betweenness.values()) if max(betweenness.values()) > 0 else 1
        for node in betweenness:
            betweenness[node] /= max_betweenness

        df = pd.DataFrame([
            {'account_id': k, 'betweenness_centrality': v}
            for k, v in betweenness.items()
        ])

        logger.info("Betweenness centrality computation complete")

        return df

    def merge_features_to_transactions(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Merge all graph features to transaction DataFrame.

        Args:
            df: Original transaction DataFrame

        Returns:
            DataFrame with graph features added
        """
        logger.info("Merging graph features to transactions...")

        # Build graph from transactions
        self.build_graph(df)

        # Compute features
        node_features = self.compute_node_features()
        community_features = self.compute_community_features(df)
        pagerank_features = self.compute_pagerank()
        betweenness_features = self.compute_betweenness()
        fraud_rings = self.detect_fraud_rings()

        # Merge all features
        result = df.merge(node_features, on='account_id', how='left')
        result = result.merge(community_features[['account_id', 'community_id', 'community_size']],
                              on='account_id', how='left')
        result = result.merge(pagerank_features, on='account_id', how='left')
        result = result.merge(betweenness_features, on='account_id', how='left')

        # Add fraud ring flag
        fraud_ring_accounts = set(fraud_rings['account1'].unique()) | set(fraud_rings['account2'].unique())
        result['fraud_ring_flag'] = result['account_id'].isin(fraud_ring_accounts).astype(int)

        # Fill missing values - only for numeric columns
        numeric_cols = result.select_dtypes(include=[np.number]).columns
        result[numeric_cols] = result[numeric_cols].fillna(0)

        # Fill categorical columns with appropriate values
        for col in result.columns:
            if col not in numeric_cols and result[col].isnull().any():
                if result[col].dtype.name == 'category':
                    result[col] = result[col].cat.add_categories([0]).fillna(0)
                else:
                    result[col] = result[col].fillna('UNKNOWN')

        logger.info(f"Added {len([c for c in result.columns if c not in df.columns])} graph features")

        return result


def main():
    """Main execution function."""
    import sys
    sys.path.append('..')

    # Create sample data
    from src.etl import FraudETLPipeline

    # Run ETL
    pipeline = FraudETLPipeline("data/raw/transactions.csv", "data/processed/test.csv")
    df = pipeline.run_pipeline()

    # Generate graph features
    generator = GraphFeatureGenerator()
    df_with_features = generator.merge_features_to_transactions(df)

    print(f"\n{'='*60}")
    print("Graph Features Summary")
    print(f"{'='*60}")
    print(f"Original columns: {len(df.columns)}")
    print(f"After features: {len(df_with_features.columns)}")
    print(f"New features: {set(df_with_features.columns) - set(df.columns)}")


if __name__ == "__main__":
    main()
