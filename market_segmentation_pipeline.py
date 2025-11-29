"""
Market Segmentation Data Mining Pipeline
Uses LSTM to generate time-series embeddings then clusters markets based on these embeddings for dynamic segmentation.
"""

import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
from typing import Tuple, Dict, List
import warnings
warnings.filterwarnings('ignore')

# Deep Learning
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Clustering
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# Snowflake
import snowflake.connector

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns


class TimeSeriesDataset(Dataset):
    """Dataset for time-series sequences"""
    def __init__(self, sequences, labels=None):
        self.sequences = torch.FloatTensor(sequences)
        self.labels = labels
        
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        if self.labels is not None:
            return self.sequences[idx], self.labels[idx]
        return self.sequences[idx]


class LSTMEncoder(nn.Module):
    """LSTM-based encoder for time-series embeddings"""
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, embedding_dim=32, dropout=0.2):
        super(LSTMEncoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_dim, 
            hidden_dim, 
            num_layers, 
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Projection to embedding space
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, embedding_dim)
        )
        
    def forward(self, x):
        # x shape: (batch, seq_len, input_dim)
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Use the last hidden state
        last_hidden = hidden[-1]  # (batch, hidden_dim)
        
        # Project to embedding
        embedding = self.fc(last_hidden)  # (batch, embedding_dim)
        
        return embedding


# GRU and Transformer removed - using LSTM only for simplicity and reliability


class MarketSegmentationPipeline:
    """Main pipeline for market segmentation using deep learning"""
    
    def __init__(self, snowflake_config: Dict, device='cpu'):
        """
        Args:
            snowflake_config: Dictionary with Snowflake connection parameters
            device: 'cpu' or 'cuda'
        """
        self.snowflake_config = snowflake_config
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        
    def get_snowflake_connection(self):
        """Create Snowflake connection"""
        return snowflake.connector.connect(**self.snowflake_config)
    
    def extract_time_series_data(self, lookback_months=12) -> pd.DataFrame:
        """
        Extract time-series data from Snowflake DWH
        Aggregates sales data by market (CountryRegion) and time period (monthly)
        """
        print(f"📊 Extracting time-series data (lookback: {lookback_months} months)...")
        
        conn = self.get_snowflake_connection()
        cur = conn.cursor()
        
        query = f"""
        WITH MonthlyMarketData AS (
            SELECT
                t.CountryRegion AS Market,
                dt.Year,
                dt.Month,
                dt.DateKey,
                SUM(f.Revenue) AS Revenue,
                SUM(f.ProductQuantity) AS Quantity,
                COUNT(DISTINCT f.SalesOrderID) AS OrderCount,
                COUNT(DISTINCT f.DimCustomerKey) AS CustomerCount,
                AVG(f.Revenue) AS AvgOrderValue,
                -- Product diversity metrics
                COUNT(DISTINCT p.Subcategory) AS UniqueSubcategories,
                COUNT(DISTINCT p.ProductLine) AS UniqueProductLines
            FROM FACTSALE f
            JOIN BRIDGEPRODUCTSPECIALOFFER b 
                ON b.BrdgProductSpecialOfferKey = f.BrdgProductSpecialOfferKey
                AND b.IsActive = TRUE
            JOIN DIMPRODUCT p 
                ON p.ProductSuggorateKey = b.ProductSuggorateKey
                AND p.IsActive = TRUE
            JOIN DIMTERRITORY t 
                ON t.TerritorySuggorateKey = f.DimTerritoryKey
            LEFT JOIN DIMTIME dt 
                ON dt.DateKey = f.DimTimeKey
            GROUP BY t.CountryRegion, dt.Year, dt.Month, dt.DateKey
        ),
        RecentMonths AS (
            -- Get the most recent N months from the available data
            SELECT DISTINCT Year, Month, DateKey
            FROM MonthlyMarketData
            ORDER BY Year DESC, Month DESC
            LIMIT {lookback_months}
        )
        SELECT 
            m.Market,
            m.Year,
            m.Month,
            m.DateKey,
            m.Revenue,
            m.Quantity,
            m.OrderCount,
            m.CustomerCount,
            m.AvgOrderValue,
            m.UniqueSubcategories,
            m.UniqueProductLines,
            -- Calculate growth rates (compared to previous month)
            LAG(m.Revenue) OVER (PARTITION BY m.Market ORDER BY m.Year, m.Month) AS PrevRevenue,
            LAG(m.Quantity) OVER (PARTITION BY m.Market ORDER BY m.Year, m.Month) AS PrevQuantity
        FROM MonthlyMarketData m
        INNER JOIN RecentMonths r 
            ON m.Year = r.Year AND m.Month = r.Month
        ORDER BY m.Market, m.Year, m.Month
        """
        
        cur.execute(query)
        df = cur.fetch_pandas_all()
        
        # Snowflake returns uppercase column names - normalize to lowercase
        df.columns = df.columns.str.lower()
        
        # Calculate growth rates
        df['revenuegrowth'] = (df['revenue'] - df['prevrevenue']) / (df['prevrevenue'] + 1e-6)
        df['quantitygrowth'] = (df['quantity'] - df['prevquantity']) / (df['prevquantity'] + 1e-6)
        
        # Fill NaN values for first month
        df['revenuegrowth'].fillna(0, inplace=True)
        df['quantitygrowth'].fillna(0, inplace=True)
        
        conn.close()
        
        if len(df) == 0:
            raise ValueError("No data extracted from Snowflake! Check your FactSale table has data.")
        
        print(f"✅ Extracted {len(df)} records for {df['market'].nunique()} markets")
        print(f"   Date range: {df['datekey'].min()} to {df['datekey'].max()}")
        
        # Show data per market
        market_counts = df.groupby('market').size().sort_values(ascending=False)
        print(f"   Top markets by record count:")
        for market, count in market_counts.head(5).items():
            print(f"     - {market}: {count} months")
        
        return df
    
    def extract_static_features(self) -> pd.DataFrame:
        """
        Extract static/aggregated features for each market
        These features don't change over time (or change slowly)
        """
        print(f"📊 Extracting static features per market...")
        
        conn = self.get_snowflake_connection()
        cur = conn.cursor()
        
        query = """
        SELECT
            t.CountryRegion AS Market,
            -- Average discount percentage per market
            AVG(so.DiscountPct) AS AvgDiscount,
            -- Customer loyalty distribution
            SUM(CASE WHEN c.LoyatyStatus = 'Platinum' THEN 1 ELSE 0 END) * 1.0 / NULLIF(COUNT(DISTINCT c.CustomerSuggorateKey), 0) AS PlatinumShare,
            SUM(CASE WHEN c.LoyatyStatus = 'Gold' THEN 1 ELSE 0 END) * 1.0 / NULLIF(COUNT(DISTINCT c.CustomerSuggorateKey), 0) AS GoldShare,
            SUM(CASE WHEN c.LoyatyStatus = 'Silver' THEN 1 ELSE 0 END) * 1.0 / NULLIF(COUNT(DISTINCT c.CustomerSuggorateKey), 0) AS SilverShare,
            -- Average customer lifetime value
            AVG(c.LifeTimeValue) AS AvgLTV,
            -- Product mix
            COUNT(DISTINCT p.ProductLine) AS TotalProductLines,
            COUNT(DISTINCT p.Subcategory) AS TotalSubcategories
        FROM FACTSALE f
        JOIN DIMTERRITORY t 
            ON t.TerritorySuggorateKey = f.DimTerritoryKey
        JOIN DIMCUSTOMER c 
            ON c.CustomerSuggorateKey = f.DimCustomerKey
            AND c.IsActive = TRUE
        JOIN BRIDGEPRODUCTSPECIALOFFER b 
            ON b.BrdgProductSpecialOfferKey = f.BrdgProductSpecialOfferKey
            AND b.IsActive = TRUE
        JOIN DIMSPECIALOFFER so 
            ON so.SpecialOfferSuggorateKey = b.SpecialOfferSuggorateKey
            AND so.IsActive = TRUE
        JOIN DIMPRODUCT p 
            ON p.ProductSuggorateKey = b.ProductSuggorateKey
            AND p.IsActive = TRUE
        GROUP BY t.CountryRegion
        ORDER BY t.CountryRegion
        """
        
        cur.execute(query)
        df = cur.fetch_pandas_all()
        
        # Normalize column names
        df.columns = df.columns.str.lower()
        
        # Fill NaN values with 0
        df = df.fillna(0)
        
        conn.close()
        
        print(f"✅ Extracted static features for {len(df)} markets")
        print(f"   Features: {list(df.columns[1:])}")
        
        return df
    
    def prepare_sequences(self, df: pd.DataFrame, static_features_df: pd.DataFrame, sequence_length=6) -> Tuple[np.ndarray, np.ndarray, List[str], pd.DataFrame]:
        """
        Prepare time-series sequences and static features for each market
        
        Args:
            df: DataFrame with time-series data
            static_features_df: DataFrame with static features per market
            sequence_length: Number of time steps in each sequence
            
        Returns:
            sequences: numpy array of shape (n_markets, sequence_length, n_time_features)
            static_features: numpy array of shape (n_markets, n_static_features)
            market_names: list of market names
            metadata: DataFrame with market metadata for the latest period
        """
        print(f"🔧 Preparing sequences (length: {sequence_length})...")
        
        # Features to use for time-series (lowercase to match Snowflake)
        time_feature_cols = [
            'revenue', 'quantity', 'ordercount', 'customercount', 
            'avgordervalue', 'uniquesubcategories', 'uniqueproductlines',
            'revenuegrowth', 'quantitygrowth'
        ]
        
        # Static features (from static_features_df)
        static_feature_cols = [
            'avgdiscount', 'platinumshare', 'goldshare', 'silvershare',
            'avgltv', 'totalproductlines', 'totalsubcategories'
        ]
        
        self.feature_names = time_feature_cols
        self.static_feature_names = static_feature_cols
        
        sequences = []
        static_features = []
        market_names = []
        metadata_list = []
        
        for market in df['market'].unique():
            market_data = df[df['market'] == market].sort_values(['year', 'month'])
            
            if len(market_data) < sequence_length:
                print(f"⚠️  Skipping {market}: insufficient data ({len(market_data)} < {sequence_length})")
                continue
            
            # Get the most recent sequence for LSTM
            recent_data = market_data.tail(sequence_length)
            sequence = recent_data[time_feature_cols].values
            
            # Get static features for this market
            static_row = static_features_df[static_features_df['market'] == market]
            if len(static_row) == 0:
                print(f"⚠️  Skipping {market}: no static features found")
                continue
            
            static_feat = static_row[static_feature_cols].values.flatten()
            
            sequences.append(sequence)
            static_features.append(static_feat)
            market_names.append(market)
            
            # Store metadata for the latest period
            latest = market_data.iloc[-1]
            metadata_list.append({
                'Market': market,
                'DateKey': latest['datekey'],
                'Revenue': latest['revenue'],
                'Quantity': latest['quantity'],
                'ProductRange': self._infer_product_range(latest['revenue'])
            })
        
        sequences = np.array(sequences)  # Shape: (n_markets, sequence_length, n_time_features)
        static_features = np.array(static_features)  # Shape: (n_markets, n_static_features)
        metadata = pd.DataFrame(metadata_list)
        
        if len(sequences) == 0:
            raise ValueError(
                f"No markets have sufficient data! "
                f"Need at least {sequence_length} months of data per market. "
                f"Try reducing sequence_length or check your data."
            )
        
        print(f"✅ Prepared {len(sequences)} sequences with {sequences.shape[2]} time-series features")
        print(f"✅ Prepared {len(static_features)} static feature vectors with {static_features.shape[1]} features")
        return sequences, static_features, market_names, metadata
    
    def _infer_product_range(self, revenue):
        """Infer product range category based on revenue"""
        if revenue < 10000:
            return 'Low'
        elif revenue < 50000:
            return 'Medium'
        else:
            return 'High'
    
    def normalize_sequences(self, sequences: np.ndarray) -> np.ndarray:
        """Normalize sequences using StandardScaler"""
        print("📏 Normalizing sequences...")
        
        n_markets, seq_len, n_features = sequences.shape
        
        # Reshape to (n_markets * seq_len, n_features)
        sequences_reshaped = sequences.reshape(-1, n_features)
        
        # Fit and transform
        sequences_normalized = self.scaler.fit_transform(sequences_reshaped)
        
        # Reshape back
        sequences_normalized = sequences_normalized.reshape(n_markets, seq_len, n_features)
        
        return sequences_normalized
    
    def build_model(self, input_dim, hidden_dim=64, embedding_dim=32):
        """Build the LSTM encoder model"""
        print(f"🏗️  Building LSTM encoder...")
        
        model = LSTMEncoder(input_dim, hidden_dim, embedding_dim=embedding_dim)
        model = model.to(self.device)
        self.model = model
        
        print(f"✅ Model built with {sum(p.numel() for p in model.parameters())} parameters")
        return model
    
    def train_autoencoder(self, sequences: np.ndarray, epochs=50, batch_size=16, lr=0.001):
        """
        Train the encoder using reconstruction loss (autoencoder approach)
        This is unsupervised learning to learn meaningful embeddings
        """
        print(f"🎓 Training autoencoder for {epochs} epochs...")
        
        # Build decoder
        embedding_dim = 32
        input_dim = sequences.shape[2]
        
        decoder = nn.Sequential(
            nn.Linear(embedding_dim, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim * sequences.shape[1])
        ).to(self.device)
        
        # Create dataset and dataloader
        dataset = TimeSeriesDataset(sequences)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Optimizer
        params = list(self.model.parameters()) + list(decoder.parameters())
        optimizer = torch.optim.Adam(params, lr=lr)
        criterion = nn.MSELoss()
        
        # Training loop
        self.model.train()
        decoder.train()
        
        losses = []
        for epoch in range(epochs):
            epoch_loss = 0
            for batch in dataloader:
                batch = batch.to(self.device)
                
                # Forward pass
                embeddings = self.model(batch)
                reconstructed = decoder(embeddings)
                reconstructed = reconstructed.view(batch.shape)
                
                # Compute loss
                loss = criterion(reconstructed, batch)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / len(dataloader)
            losses.append(avg_loss)
            
            if (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")
        
        print("✅ Training completed!")
        return losses
    
    def generate_embeddings(self, sequences: np.ndarray) -> np.ndarray:
        """Generate embeddings for sequences"""
        print("🧬 Generating embeddings...")
        
        self.model.eval()
        
        dataset = TimeSeriesDataset(sequences)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
        
        embeddings = []
        with torch.no_grad():
            for batch in dataloader:
                batch = batch.to(self.device)
                embedding = self.model(batch)
                embeddings.append(embedding.cpu().numpy())
        
        embeddings = np.vstack(embeddings)
        print(f"✅ Generated embeddings with shape {embeddings.shape}")
        
        return embeddings
    
    def cluster_embeddings(self, embeddings: np.ndarray, static_features: np.ndarray, n_clusters=5, method='kmeans') -> Tuple[np.ndarray, float]:
        """
        Cluster using hybrid approach: LSTM embeddings + Static features
        
        Args:
            embeddings: numpy array of LSTM time-series embeddings
            static_features: numpy array of static market features
            n_clusters: number of clusters
            method: 'kmeans' or 'dbscan'
            
        Returns:
            cluster_labels: array of cluster assignments
            silhouette: silhouette score
        """
        print(f"🎯 Clustering with HYBRID approach (LSTM + Static) using {method.upper()}...")
        
        # Step 1: Normalize static features (they have different scales than embeddings)
        static_scaler = StandardScaler()
        static_normalized = static_scaler.fit_transform(static_features)
        
        # Step 2: Concatenate LSTM embeddings with normalized static features
        combined_features = np.hstack([embeddings, static_normalized])
        
        print(f"   LSTM embedding shape: {embeddings.shape}")
        print(f"   Static features shape: {static_features.shape}")
        print(f"   Combined feature shape: {combined_features.shape}")
        
        # Step 3: Optional final normalization to ensure equal weighting
        # This ensures neither LSTM nor static features dominate
        final_scaler = StandardScaler()
        combined_features = final_scaler.fit_transform(combined_features)
        
        # Step 4: Perform clustering on combined features
        if method == 'kmeans':
            clusterer = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = clusterer.fit_predict(combined_features)
        elif method == 'dbscan':
            clusterer = DBSCAN(eps=0.5, min_samples=2)
            cluster_labels = clusterer.fit_predict(combined_features)
            n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
            print(f"  DBSCAN found {n_clusters} clusters")
        else:
            raise ValueError(f"Unknown clustering method: {method}")
        
        # Calculate silhouette score on combined features
        if len(set(cluster_labels)) > 1:
            silhouette = silhouette_score(combined_features, cluster_labels)
        else:
            silhouette = 0.0
        
        print(f"✅ Clustering completed. Silhouette score: {silhouette:.4f}")
        
        return cluster_labels, silhouette
    
    def save_to_snowflake(self, metadata: pd.DataFrame, cluster_labels: np.ndarray, 
                         embeddings: np.ndarray, silhouette: float):
        """Save segmentation results to FactMarketSegmentation table"""
        print("💾 Saving results to Snowflake...")
        
        conn = self.get_snowflake_connection()
        cur = conn.cursor()
        
        model_version = f"lstm_hybrid_v2.0_{datetime.now().strftime('%Y%m%d')}"
        
        # Prepare data for insertion
        for idx, row in metadata.iterrows():
            market = row['Market']
            date_key = int(row['DateKey'])
            segment_id = int(cluster_labels[idx])
            revenue = float(row['Revenue'])
            quantity = int(row['Quantity'])
            product_range = row['ProductRange']
            embedding_json = json.dumps(embeddings[idx].tolist())
            
            # Insert or update
            query = f"""
            MERGE INTO FACTMARKETSEGMENTATION AS target
            USING (
                SELECT 
                    {date_key} AS DateKey,
                    '{market}' AS MarketNameKey
            ) AS source
            ON target.DateKey = source.DateKey 
                AND target.MarketNameKey = source.MarketNameKey
            WHEN MATCHED THEN
                UPDATE SET
                    SegmentID = {segment_id},
                    Revenue = {revenue},
                    Quantity = {quantity},
                    ProductRange = '{product_range}',
                    ModelVersion = '{model_version}',
                    EmbeddingJSON = '{embedding_json}',
                    ConfidenceScore = {silhouette},
                    CreatedAt = CURRENT_TIMESTAMP()
            WHEN NOT MATCHED THEN
                INSERT (DateKey, MarketNameKey, SegmentID, Revenue, Quantity, 
                       ProductRange, ModelVersion, EmbeddingJSON, ConfidenceScore)
                VALUES ({date_key}, '{market}', {segment_id}, {revenue}, {quantity},
                       '{product_range}', '{model_version}', '{embedding_json}', {silhouette})
            """
            
            cur.execute(query)
        
        conn.commit()
        conn.close()
        
        print(f"✅ Saved {len(metadata)} records to FACTMARKETSEGMENTATION")
    
    def run_pipeline(self, sequence_length=6, n_clusters=5, epochs=50):
        """Run the complete hybrid pipeline (LSTM + Static Features)"""
        print("=" * 80)
        print("🚀 MARKET SEGMENTATION PIPELINE - HYBRID APPROACH")
        print("   LSTM (Time-Series) + Static Features (Territory, Loyalty, etc.)")
        print("=" * 80)
        
        # Step 1: Extract time-series data
        df = self.extract_time_series_data(lookback_months=12)
        
        # Step 2: Extract static features
        static_features_df = self.extract_static_features()
        
        # Step 3: Prepare sequences and static features
        sequences, static_features, market_names, metadata = self.prepare_sequences(
            df, static_features_df, sequence_length
        )
        
        # Step 4: Normalize time-series sequences
        sequences_normalized = self.normalize_sequences(sequences)
        
        # Step 5: Build LSTM model
        input_dim = sequences.shape[2]
        self.build_model(input_dim)
        
        # Step 6: Train autoencoder on time-series
        losses = self.train_autoencoder(sequences_normalized, epochs=epochs)
        
        # Step 7: Generate LSTM embeddings from time-series
        embeddings = self.generate_embeddings(sequences_normalized)
        
        # Step 8: Cluster using HYBRID approach (LSTM embeddings + Static features)
        cluster_labels, silhouette = self.cluster_embeddings(
            embeddings, static_features, n_clusters=n_clusters
        )
        
        # Step 9: Save to Snowflake
        self.save_to_snowflake(metadata, cluster_labels, embeddings, silhouette)
        
        print("=" * 80)
        print("✅ PIPELINE COMPLETED SUCCESSFULLY")
        print("=" * 80)
        
        # Return results for analysis
        return {
            'market_names': market_names,
            'embeddings': embeddings,
            'static_features': static_features,
            'cluster_labels': cluster_labels,
            'silhouette_score': silhouette,
            'metadata': metadata,
            'training_losses': losses
        }


def main():
    """Main execution function"""
    
    # Snowflake configuration (update with your credentials)
    snowflake_config = {
        'user': 'YOUR_USER',
        'password': 'YOUR_PASSWORD',
        'account': 'YOUR_ACCOUNT',
        'warehouse': 'YOUR_WAREHOUSE',
        'database': 'YOUR_DATABASE',
        'schema': 'DWH'
    }
    
    # Initialize pipeline
    pipeline = MarketSegmentationPipeline(
        snowflake_config=snowflake_config,
        device='cpu'  # Use 'cuda' if GPU available
    )
    
    # Run pipeline
    results = pipeline.run_pipeline(
        sequence_length=6,  # 6 months of history
        n_clusters=5,       # Number of market segments
        epochs=50           # Training epochs
    )
    
    print("\n📊 Results Summary:")
    print(f"  - Markets segmented: {len(results['market_names'])}")
    print(f"  - Embedding dimension: {results['embeddings'].shape[1]}")
    print(f"  - Silhouette score: {results['silhouette_score']:.4f}")
    print(f"  - Segments found: {len(set(results['cluster_labels']))}")


if __name__ == "__main__":
    main()
