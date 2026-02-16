"""
Neo4j Graph Database Integration
================================
Handles graph database operations, entity creation, and relationships.
"""

import pandas as pd
import numpy as np
from neo4j import GraphDatabase
import logging
import os
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/graph.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class Neo4jGraphLoader:
    """
    Enterprise Neo4j Graph Database Integration for Fraud Detection.
    """

    def __init__(self, uri: str = "bolt://localhost:7687",
                 username: str = "neo4j",
                 password: str = "password"):
        """
        Initialize Neo4j connection.

        Args:
            uri: Neo4j bolt URI
            username: Neo4j username
            password: Neo4j password
        """
        self.uri = uri
        self.username = username
        self.password = password
        self.driver = None
        self._connect()
        logger.info(f"Neo4j Graph Loader initialized with URI: {uri}")

    def _connect(self):
        """Establish connection to Neo4j."""
        try:
            self.driver = GraphDatabase.driver(
                self.uri,
                auth=(self.username, self.password)
            )
            # Test connection
            with self.driver.session() as session:
                result = session.run("RETURN 1 AS test")
                result.single()
            logger.info("Successfully connected to Neo4j")
        except Exception as e:
            logger.warning(f"Could not connect to Neo4j: {e}")
            logger.info("Running in offline mode with simulated graph features")
            self.driver = None

    def close(self):
        """Close the Neo4j connection."""
        if self.driver:
            self.driver.close()
            logger.info("Neo4j connection closed")

    def clear_database(self):
        """Clear all nodes and relationships from the database."""
        if not self.driver:
            logger.warning("No Neo4j connection - skipping clear_database")
            return

        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
        logger.info("Database cleared")

    def create_constraints(self):
        """Create uniqueness constraints for node IDs."""
        if not self.driver:
            logger.warning("No Neo4j connection - skipping constraints")
            return

        constraints = [
            "CREATE CONSTRAINT IF NOT EXISTS FOR (a:Account) REQUIRE a.account_id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (t:Transaction) REQUIRE t.transaction_id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (m:Merchant) REQUIRE m.merchant_id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (d:Device) REQUIRE d.device_id IS UNIQUE",
        ]

        with self.driver.session() as session:
            for constraint in constraints:
                try:
                    session.run(constraint)
                except Exception as e:
                    logger.debug(f"Constraint creation: {e}")

        logger.info("Constraints created")

    def load_transactions_to_graph(self, df: pd.DataFrame, batch_size: int = 1000):
        """
        Load transaction data into Neo4j graph.

        Args:
            df: Processed DataFrame with transaction data
            batch_size: Number of records to process per batch
        """
        if not self.driver:
            logger.warning("No Neo4j connection - skipping graph load")
            return

        logger.info(f"Loading {len(df)} transactions to Neo4j...")

        # Clear existing data
        self.clear_database()
        self.create_constraints()

        # Create nodes and relationships in batches
        total_batches = (len(df) + batch_size - 1) // batch_size

        with self.driver.session() as session:
            for i in range(total_batches):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, len(df))
                batch = df.iloc[start_idx:end_idx]

                # Create Account nodes
                accounts = batch[['account_id']].drop_duplicates()
                for _, row in accounts.iterrows():
                    session.run("""
                        MERGE (a:Account {account_id: $account_id})
                    """, account_id=row['account_id'])

                # Create Merchant nodes
                merchants = batch[['merchant_id', 'merchant_category']].drop_duplicates()
                for _, row in merchants.iterrows():
                    session.run("""
                        MERGE (m:Merchant {merchant_id: $merchant_id})
                        SET m.category = $category
                    """, merchant_id=row['merchant_id'], category=row.get('merchant_category', 'unknown'))

                # Create Device nodes
                devices = batch[['device_id']].drop_duplicates()
                for _, row in devices.iterrows():
                    session.run("""
                        MERGE (d:Device {device_id: $device_id})
                    """, device_id=row['device_id'])

                # Create Transaction nodes and relationships
                for _, row in batch.iterrows():
                    # Parse timestamp
                    ts = pd.to_datetime(row['timestamp'])
                    timestamp_str = ts.isoformat()

                    session.run("""
                        MERGE (t:Transaction {transaction_id: $transaction_id})
                        SET t.amount = $amount,
                            t.timestamp = $timestamp,
                            t.is_fraud = $is_fraud,
                            t.hour = $hour,
                            t.day_of_week = $day_of_week
                        WITH t
                        MATCH (a:Account {account_id: $account_id})
                        MERGE (a)-[:MAKES]->(t)
                    """,
                        transaction_id=row['transaction_id'],
                        amount=float(row['amount']),
                        timestamp=timestamp_str,
                        is_fraud=int(row['is_fraud']),
                        hour=int(row.get('hour', ts.hour)),
                        day_of_week=int(row.get('day_of_week', ts.dayofweek)),
                        account_id=row['account_id']
                    )

                    session.run("""
                        MATCH (t:Transaction {transaction_id: $transaction_id})
                        MATCH (m:Merchant {merchant_id: $merchant_id})
                        MERGE (t)-[:TO]->(m)
                    """,
                        transaction_id=row['transaction_id'],
                        merchant_id=row['merchant_id']
                    )

                    session.run("""
                        MATCH (a:Account {account_id: $account_id})
                        MATCH (d:Device {device_id: $device_id})
                        MERGE (a)-[:USES]->(d)
                    """,
                        account_id=row['account_id'],
                        device_id=row['device_id']
                    )

                logger.info(f"Batch {i+1}/{total_batches} loaded")

        logger.info(f"Successfully loaded {len(df)} transactions to Neo4j")

    def get_graph_stats(self) -> Dict:
        """Get graph statistics."""
        if not self.driver:
            return self._get_simulated_stats()

        stats = {}

        with self.driver.session() as session:
            # Count nodes
            result = session.run("MATCH (n) RETURN labels(n)[0] AS label, count(*) AS count")
            for record in result:
                stats[f"{record['label']}_count"] = record['count']

            # Count relationships
            result = session.run("MATCH ()-[r]->() RETURN type(r) AS type, count(*) AS count")
            for record in result:
                stats[f"{record['type']}_count"] = record['count']

        return stats

    def _get_simulated_stats(self) -> Dict:
        """Return simulated stats when Neo4j is not available."""
        return {
            'Account_count': 0,
            'Transaction_count': 0,
            'Merchant_count': 0,
            'Device_count': 0,
            'MAKES_count': 0,
            'TO_count': 0,
            'USES_count': 0
        }

    def run_graph_analytics(self) -> pd.DataFrame:
        """
        Run graph analytics using Neo4j Cypher queries.
        Returns DataFrame with graph features for each transaction.
        """
        if not self.driver:
            logger.warning("No Neo4j connection - returning empty graph features")
            return pd.DataFrame()

        logger.info("Running graph analytics...")

        features = []

        with self.driver.session() as session:
            # 1. Account degree (number of transactions)
            result = session.run("""
                MATCH (a:Account)-[:MAKES]->(t:Transaction)
                RETURN a.account_id AS account_id, count(t) AS node_degree
            """)
            degree_map = {record['account_id']: record['node_degree'] for record in result}

            # 2. Device shared count (accounts using same device)
            result = session.run("""
                MATCH (a1:Account)-[:USES]->(d:Device)<-[:USES]-(a2:Account)
                WHERE a1.account_id <> a2.account_id
                RETURN a1.account_id AS account_id, count(DISTINCT a2) AS shared_device_count
            """)
            shared_device_map = {record['account_id']: record['shared_device_count'] for record in result}

            # 3. Merchant transaction count
            result = session.run("""
                MATCH (t:Transaction)-[:TO]->(m:Merchant)
                RETURN m.merchant_id AS merchant_id, count(t) AS merchant_txn_count
            """)
            merchant_count_map = {record['merchant_id']: record['merchant_txn_count'] for record in result}

            # 4. Device transaction count
            result = session.run("""
                MATCH (a:Account)-[:USES]->(d:Device)<-[:USES]-(a2:Account)
                RETURN d.device_id AS device_id, count(DISTINCT a2) AS device_account_count
            """)
            device_account_map = {record['device_id']: record['device_account_count'] for record in result}

            # 5. Find suspicious patterns (accounts with multiple devices making high-value txns)
            result = session.run("""
                MATCH (a:Account)-[:USES]->(d:Device)
                MATCH (a)-[:MAKES]->(t:Transaction)
                WHERE t.amount > 500
                WITH a, count(DISTINCT d) AS device_count, count(t) AS high_value_txns
                WHERE device_count > 2 OR high_value_txns > 5
                RETURN a.account_id AS account_id, 1 AS suspicious_flag
            """)
            suspicious_accounts = set(record['account_id'] for record in result)

            # Get all accounts for features
            result = session.run("MATCH (a:Account) RETURN a.account_id AS account_id")
            all_accounts = [record['account_id'] for record in result]

            # Build features for each account
            for account_id in all_accounts:
                features.append({
                    'account_id': account_id,
                    'node_degree': degree_map.get(account_id, 0),
                    'shared_device_count': shared_device_map.get(account_id, 0),
                    'suspicious_flag': 1 if account_id in suspicious_accounts else 0
                })

        df = pd.DataFrame(features)
        logger.info(f"Generated graph features for {len(df)} accounts")

        return df

    def detect_fraud_rings(self) -> List[Dict]:
        """
        Detect potential fraud rings using cycle detection.
        """
        if not self.driver:
            logger.warning("No Neo4j connection - skipping fraud ring detection")
            return []

        logger.info("Detecting fraud rings...")

        fraud_rings = []

        with self.driver.session() as session:
            # Find cycles in the graph (accounts connected via devices making similar transactions)
            result = session.run("""
                MATCH (a1:Account)-[:USES]->(d:Device)<-[:USES]-(a2:Account)-[:MAKES]->(t1:Transaction)
                MATCH (a1)-[:MAKES]->(t2:Transaction)
                WHERE t1.merchant_id = t2.merchant_id
                AND t1.amount > t2.amount * 0.8
                AND t1.amount < t2.amount * 1.2
                WITH a1, a2, collect(DISTINCT d.device_id) AS shared_devices
                WHERE size(shared_devices) > 1
                RETURN a1.account_id AS account1, a2.account_id AS account2,
                       shared_devices, size(shared_devices) AS ring_size
                LIMIT 50
            """)

            for record in result:
                fraud_rings.append({
                    'account1': record['account1'],
                    'account2': record['account2'],
                    'shared_devices': record['shared_devices'],
                    'ring_size': record['ring_size']
                })

        logger.info(f"Detected {len(fraud_rings)} potential fraud rings")
        return fraud_rings

    def get_account_neighbors(self, account_id: str, depth: int = 2) -> Dict:
        """
        Get neighboring accounts connected to a given account.

        Args:
            account_id: Account to find neighbors for
            depth: Maximum hop depth

        Returns:
            Dict with neighbor information
        """
        if not self.driver:
            return {'accounts': [], 'merchants': [], 'devices': []}

        with self.driver.session() as session:
            # Find connected accounts via shared devices
            result = session.run(f"""
                MATCH path = (a:Account {{account_id: $account_id}})-[:USES]->(d:Device)<-[:USES]-(a2:Account)
                RETURN DISTINCT a2.account_id AS connected_account
                LIMIT 20
            """, account_id=account_id)
            connected_accounts = [record['connected_account'] for record in result]

            # Find merchants visited
            result = session.run("""
                MATCH (a:Account {account_id: $account_id})-[:MAKES]->(t:Transaction)-[:TO]->(m:Merchant)
                RETURN DISTINCT m.merchant_id AS merchant_id
                LIMIT 20
            """, account_id=account_id)
            merchants = [record['merchant_id'] for record in result]

            # Find devices used
            result = session.run("""
                MATCH (a:Account {account_id: $account_id})-[:USES]->(d:Device)
                RETURN d.device_id AS device_id
                LIMIT 20
            """, account_id=account_id)
            devices = [record['device_id'] for record in result]

        return {
            'accounts': connected_accounts,
            'merchants': merchants,
            'devices': devices
        }

    def community_detection(self) -> pd.DataFrame:
        """
        Perform community detection using Louvain algorithm via Neo4j GDS.
        Returns DataFrame with community assignments.
        """
        if not self.driver:
            logger.warning("No Neo4j connection - skipping community detection")
            return pd.DataFrame()

        logger.info("Running community detection...")

        # Check if GDS is available
        try:
            with self.driver.session() as session:
                # Try to use GDS
                result = session.run("""
                    CALL gds.graph.exists('fraudGraph') YIELD exists
                    RETURN exists
                """)
                if not result.single()['exists']:
                    # Create graph projection
                    session.run("""
                        CALL gds.graph.project('fraudGraph',
                            ['Account', 'Device'],
                            ['USES']
                        )
                    """)

                # Run Louvain
                result = session.run("""
                    CALL gds.louvain.write('fraudGraph', {
                        writeProperty: 'communityId'
                    })
                    YIELD communityCount
                    RETURN communityCount
                """)
                community_count = result.single()['communityCount']

                # Get community assignments
                result = session.run("""
                    MATCH (a:Account)
                    RETURN a.account_id AS account_id, a.communityId AS community_id
                """)

                communities = []
                for record in result:
                    communities.append({
                        'account_id': record['account_id'],
                        'community_id': record['community_id']
                    })

                logger.info(f"Found {community_count} communities")
                return pd.DataFrame(communities)

        except Exception as e:
            logger.warning(f"GDS not available, using simplified community detection: {e}")
            return self._simple_community_detection()

    def _simple_community_detection(self) -> pd.DataFrame:
        """Simple community detection using connected components."""
        if not self.driver:
            return pd.DataFrame()

        with self.driver.session() as session:
            # Find connected accounts via devices
            result = session.run("""
                MATCH (a1:Account)-[:USES]->(d:Device)<-[:USES]-(a2:Account)
                RETURN a1.account_id AS account_id, collect(DISTINCT a2.account_id) AS neighbors
            """)

            communities = {}
            community_id = 0
            visited = set()

            for record in result:
                account = record['account_id']
                if account not in visited:
                    # BFS to find all connected accounts
                    queue = [account]
                    while queue:
                        curr = queue.pop(0)
                        if curr not in visited:
                            visited.add(curr)
                            communities[curr] = community_id
                            for neighbor in record['neighbors']:
                                if neighbor not in visited:
                                    queue.append(neighbor)
                    community_id += 1

            # Add isolated accounts
            result = session.run("MATCH (a:Account) RETURN a.account_id AS account_id")
            for record in result:
                if record['account_id'] not in communities:
                    communities[record['account_id']] = community_id
                    community_id += 1

            df = pd.DataFrame([
                {'account_id': k, 'community_id': v}
                for k, v in communities.items()
            ])

            logger.info(f"Simple community detection found {community_id} communities")
            return df


def main():
    """Main execution function for testing."""
    import sys
    sys.path.append('..')

    # Test connection
    loader = Neo4jGraphLoader()

    stats = loader.get_graph_stats()
    print(f"\n{'='*60}")
    print("Graph Statistics")
    print(f"{'='*60}")
    for key, value in stats.items():
        print(f"{key}: {value:,}")

    loader.close()


if __name__ == "__main__":
    main()
