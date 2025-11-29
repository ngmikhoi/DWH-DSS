"""
Streamlit App for Running and Monitoring Market Segmentation Pipeline
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
from datetime import datetime
import sys

from market_segmentation_pipeline import MarketSegmentationPipeline

# Page configuration
st.set_page_config(
    page_title="Market Segmentation Pipeline",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: bold;
        padding: 0.75rem;
        border: none;
    }
    .stButton>button:hover {
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    .metric-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .success-card {
        background: linear-gradient(135deg, #d4fc79 0%, #96e6a1 100%);
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
    }
    .warning-card {
        background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
    }
    h1 {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3rem;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.title("🧬 Market Segmentation Pipeline")
st.markdown("### Deep Learning-Based Dynamic Market Segmentation")

# Sidebar configuration
st.sidebar.header("⚙️ Pipeline Configuration")

# Snowflake credentials
st.sidebar.subheader("🔐 Snowflake Connection")
use_secrets = st.sidebar.checkbox("Use secrets.toml", value=True)

if use_secrets:
    try:
        snowflake_config = {
            'user': st.secrets["snowflake_user"],
            'password': st.secrets["snowflake_password"],
            'account': st.secrets["snowflake_account"],
            'warehouse': st.secrets["snowflake_warehouse"],
            'database': st.secrets["snowflake_database"],
            'schema': st.secrets["snowflake_schema"]
        }
        st.sidebar.success("✅ Loaded from secrets.toml")
    except Exception as e:
        st.sidebar.error(f"❌ Error loading secrets: {e}")
        st.stop()
else:
    snowflake_config = {
        'user': st.sidebar.text_input("User"),
        'password': st.sidebar.text_input("Password", type="password"),
        'account': st.sidebar.text_input("Account"),
        'warehouse': st.sidebar.text_input("Warehouse"),
        'database': st.sidebar.text_input("Database"),
        'schema': st.sidebar.text_input("Schema", value="DWH")
    }

# Model configuration
st.sidebar.subheader("🤖 Model Settings")
st.sidebar.info("Using LSTM (Long Short-Term Memory) for time-series embedding")

sequence_length = st.sidebar.slider(
    "Sequence Length (months)",
    min_value=3,
    max_value=12,
    value=6,
    help="Number of historical months to use for each sequence"
)

n_clusters = st.sidebar.slider(
    "Number of Segments",
    min_value=2,
    max_value=10,
    value=5,
    help="Target number of market segments"
)

clustering_method = st.sidebar.selectbox(
    "Clustering Method",
    ["K-Means", "DBSCAN"],
    help="Algorithm for clustering embeddings"
)

# Training parameters
st.sidebar.subheader("🎓 Training Parameters")
epochs = st.sidebar.slider(
    "Training Epochs",
    min_value=10,
    max_value=200,
    value=50,
    step=10,
    help="Number of training iterations"
)

batch_size = st.sidebar.slider(
    "Batch Size",
    min_value=8,
    max_value=64,
    value=16,
    step=8
)

learning_rate = st.sidebar.select_slider(
    "Learning Rate",
    options=[0.0001, 0.0005, 0.001, 0.005, 0.01],
    value=0.001
)

# Device selection
device = st.sidebar.radio(
    "Compute Device",
    ["CPU", "GPU (CUDA)"],
    help="GPU recommended for faster training"
)

# Main content
tab1, tab2, tab3, tab4 = st.tabs(["🚀 Run Pipeline", "📊 Results", "📈 Analysis", "ℹ️ About"])

with tab1:
    st.header("Run Segmentation Pipeline")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        <div class='metric-card'>
            <h3>🎯 Pipeline Overview</h3>
            <p>This pipeline performs the following steps:</p>
            <ol>
                <li><strong>Data Extraction:</strong> Retrieve time-series sales data from Snowflake DWH</li>
                <li><strong>Sequence Preparation:</strong> Create sequences for each market</li>
                <li><strong>Model Training:</strong> Train autoencoder to learn embeddings</li>
                <li><strong>Embedding Generation:</strong> Generate market embeddings</li>
                <li><strong>Clustering:</strong> Group markets into segments</li>
                <li><strong>Storage:</strong> Save results to FactMarketSegmentation</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class='metric-card'>
            <h3>⚙️ Current Settings</h3>
            <ul>
                <li><strong>Model:</strong> LSTM</li>
                <li><strong>Sequence:</strong> {sequence_length} months</li>
                <li><strong>Segments:</strong> {n_clusters}</li>
                <li><strong>Epochs:</strong> {epochs}</li>
                <li><strong>Device:</strong> {device}</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Run button
    if st.button("🚀 Run Pipeline", key="run_pipeline"):
        
        # Initialize pipeline
        with st.spinner("Initializing pipeline..."):
            try:
                pipeline = MarketSegmentationPipeline(
                    snowflake_config=snowflake_config,
                    device='cuda' if device == "GPU (CUDA)" else 'cpu'
                )
                st.success("✅ Pipeline initialized!")
            except Exception as e:
                st.error(f"❌ Error initializing pipeline: {e}")
                st.stop()
        
        # Create progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Run pipeline with progress updates
        try:
            # Step 1: Extract data
            status_text.text("📊 Extracting time-series data...")
            progress_bar.progress(10)
            df = pipeline.extract_time_series_data(lookback_months=12)
            st.session_state['raw_data'] = df
            
            # Step 2: Extract static features
            status_text.text("📊 Extracting static features...")
            progress_bar.progress(15)
            static_features_df = pipeline.extract_static_features()
            
            # Step 3: Prepare sequences
            status_text.text("🔧 Preparing sequences...")
            progress_bar.progress(25)
            sequences, static_features, market_names, metadata = pipeline.prepare_sequences(
                df, static_features_df, sequence_length
            )
            st.session_state['sequences'] = sequences
            st.session_state['static_features'] = static_features
            st.session_state['market_names'] = market_names
            st.session_state['metadata'] = metadata
            
            # Step 4: Normalize
            status_text.text("📏 Normalizing data...")
            progress_bar.progress(35)
            sequences_normalized = pipeline.normalize_sequences(sequences)
            
            # Step 5: Build model
            status_text.text(f"🏭️ Building LSTM model...")
            progress_bar.progress(45)
            input_dim = sequences.shape[2]
            pipeline.build_model(input_dim)
            
            # Step 6: Train
            status_text.text(f"🎓 Training for {epochs} epochs...")
            progress_bar.progress(55)
            losses = pipeline.train_autoencoder(
                sequences_normalized, 
                epochs=epochs,
                batch_size=batch_size,
                lr=learning_rate
            )
            st.session_state['losses'] = losses
            
            # Step 7: Generate embeddings
            status_text.text("🧬 Generating embeddings...")
            progress_bar.progress(75)
            embeddings = pipeline.generate_embeddings(sequences_normalized)
            st.session_state['embeddings'] = embeddings
            
            # Step 8: Cluster (Hybrid)
            status_text.text(f"🎯 Clustering with {clustering_method} (HYBRID)...")
            progress_bar.progress(90)
            cluster_labels, silhouette = pipeline.cluster_embeddings(
                embeddings,
                st.session_state['static_features'],
                n_clusters=n_clusters,
                method=clustering_method.lower().replace('-', '')
            )
            st.session_state['cluster_labels'] = cluster_labels
            st.session_state['silhouette'] = silhouette
            
            # Step 9: Save
            status_text.text("💾 Saving to Snowflake...")
            progress_bar.progress(98)
            pipeline.save_to_snowflake(metadata, cluster_labels, embeddings, silhouette)
            
            progress_bar.progress(100)
            status_text.text("✅ Pipeline completed successfully!")
            
            # Show success message
            st.markdown(f"""
            <div class='success-card'>
                <h2>🎉 Pipeline Completed Successfully!</h2>
                <p><strong>Markets Segmented:</strong> {len(market_names)}</p>
                <p><strong>Segments Created:</strong> {len(set(cluster_labels))}</p>
                <p><strong>Silhouette Score:</strong> {silhouette:.4f}</p>
                <p><strong>Embedding Dimension:</strong> {embeddings.shape[1]}</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.balloons()
            
        except Exception as e:
            st.error(f"❌ Pipeline failed: {e}")
            import traceback
            st.code(traceback.format_exc())

with tab2:
    st.header("📊 Segmentation Results")
    
    if 'cluster_labels' in st.session_state:
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Markets", len(st.session_state['market_names']))
        with col2:
            st.metric("Segments", len(set(st.session_state['cluster_labels'])))
        with col3:
            st.metric("Silhouette Score", f"{st.session_state['silhouette']:.4f}")
        with col4:
            st.metric("Embedding Dim", st.session_state['embeddings'].shape[1])
        
        st.markdown("---")
        
        # Segment distribution
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Segment Distribution")
            segment_counts = pd.Series(st.session_state['cluster_labels']).value_counts().sort_index()
            fig = px.bar(
                x=segment_counts.index,
                y=segment_counts.values,
                labels={'x': 'Segment ID', 'y': 'Number of Markets'},
                title="Markets per Segment",
                color=segment_counts.values,
                color_continuous_scale='Viridis'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Training Loss")
            if 'losses' in st.session_state:
                fig = px.line(
                    y=st.session_state['losses'],
                    labels={'index': 'Epoch', 'y': 'Loss'},
                    title="Training Loss Over Time"
                )
                fig.update_traces(line_color='#667eea', line_width=2)
                st.plotly_chart(fig, use_container_width=True)
        
        # Market-Segment mapping
        st.subheader("Market Segmentation Details")
        
        results_df = pd.DataFrame({
            'Market': st.session_state['market_names'],
            'Segment': st.session_state['cluster_labels'],
            'Revenue': st.session_state['metadata']['Revenue'].values,
            'Quantity': st.session_state['metadata']['Quantity'].values,
            'Product Range': st.session_state['metadata']['ProductRange'].values
        })
        
        # Add segment statistics
        results_df = results_df.sort_values('Segment')
        
        st.dataframe(
            results_df.style.background_gradient(subset=['Revenue', 'Quantity'], cmap='YlGnBu'),
            use_container_width=True
        )
        
        # Download results
        csv = results_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Results as CSV",
            data=csv,
            file_name=f"market_segmentation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
        
    else:
        st.info("👈 Run the pipeline first to see results")

with tab3:
    st.header("📈 Advanced Analysis")
    
    if 'embeddings' in st.session_state:
        
        # Embedding visualization (2D projection using PCA)
        st.subheader("Hybrid Feature Space Visualization")
        st.markdown("*Visualizing combined LSTM embeddings + Static features (same as used for clustering)*")
        
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler
        
        embeddings = st.session_state['embeddings']
        static_features = st.session_state['static_features']
        
        # Combine features the same way as clustering
        static_scaler = StandardScaler()
        static_normalized = static_scaler.fit_transform(static_features)
        combined_features = np.hstack([embeddings, static_normalized])
        
        # Apply PCA on combined features
        pca = PCA(n_components=2)
        combined_2d = pca.fit_transform(combined_features)
        
        viz_df = pd.DataFrame({
            'PC1': combined_2d[:, 0],
            'PC2': combined_2d[:, 1],
            'Market': st.session_state['market_names'],
            'Segment': st.session_state['cluster_labels'].astype(str),
            'Revenue': st.session_state['metadata']['Revenue'].values
        })
        
        fig = px.scatter(
            viz_df,
            x='PC1',
            y='PC2',
            color='Segment',
            size='Revenue',
            hover_data=['Market'],
            title="Market Segments (PCA on LSTM + Static Features)",
            color_discrete_sequence=px.colors.qualitative.Bold
        )
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown(f"**Explained Variance:** PC1: {pca.explained_variance_ratio_[0]:.2%}, PC2: {pca.explained_variance_ratio_[1]:.2%}")
        
        # Segment characteristics
        st.subheader("Segment Characteristics")
        
        results_df = pd.DataFrame({
            'Market': st.session_state['market_names'],
            'Segment': st.session_state['cluster_labels'],
            'Revenue': st.session_state['metadata']['Revenue'].values,
            'Quantity': st.session_state['metadata']['Quantity'].values
        })
        
        segment_stats = results_df.groupby('Segment').agg({
            'Market': 'count',
            'Revenue': ['mean', 'sum', 'std'],
            'Quantity': ['mean', 'sum', 'std']
        }).round(2)
        
        st.dataframe(segment_stats, use_container_width=True)
        
    else:
        st.info("👈 Run the pipeline first to see analysis")

with tab4:
    st.header("ℹ️ About This Pipeline")
    
    st.markdown("""
    ### 🧬 Hybrid Market Segmentation (LSTM + Static Features)
    
    This pipeline uses a **hybrid approach** combining LSTM deep learning for time-series patterns with static market features for comprehensive segmentation.
    
    #### 🎯 Key Features:
    
    1. **Hybrid Architecture**: Combines temporal patterns (LSTM) with structural attributes (static features)
    2. **Time-Series Embeddings**: LSTM learns from revenue trends, growth rates, and seasonality
    3. **Static Features**: Territory, customer loyalty distribution, average LTV, product mix
    4. **Interpretable Segments**: Can explain segments by both behavior AND characteristics
    5. **Dynamic Segmentation**: Segments evolve as market behavior changes over time
    
    #### 🏗️ Why Hybrid Approach?
    
    **The hybrid approach is superior because:**
    - ✅ **Right tool for each job**: LSTM for temporal patterns, clustering for static attributes
    - ✅ **Interpretability**: Can explain "Markets in Europe with declining sales" vs black-box embeddings
    - ✅ **Respects schema design**: Separates time-variant (FactSale) from static (DimTerritory, DimCustomer)
    - ✅ **Better performance**: Combining both types of information improves segmentation quality
    
    #### 📊 How It Works:
    
    **Phase 1: Time-Series Learning (LSTM)**
    1. Extract monthly time-series data per market (revenue, quantity, growth rates, etc.)
    2. Create sequences of N months for each market
    3. Train LSTM autoencoder to compress sequences into embeddings
    4. Embeddings capture: trends, seasonality, volatility, growth patterns
    
    **Phase 2: Static Feature Extraction**
    5. Extract aggregated market attributes (territory, loyalty distribution, avg LTV, product mix)
    6. These features represent structural/contextual market characteristics
    
    **Phase 3: Hybrid Clustering**
    7. Normalize static features to match embedding scale
    8. Concatenate: `[LSTM Embedding] + [Static Features]`
    9. Apply K-Means/DBSCAN on combined feature vector
    10. Store results in FactMarketSegmentation table
    
    #### 🎓 Model Training:
    
    The LSTM autoencoder is trained to reconstruct input sequences, forcing the model to learn
    meaningful temporal representations. Markets with similar time-series patterns get similar embeddings.
    These embeddings are then combined with static features for final clustering.
    
    #### 📈 Benefits:
    
    - **Comprehensive**: Captures both "how markets behave" (LSTM) and "what markets are" (static)
    - **Interpretable**: Segments can be described in business terms
    - **Scalable**: Efficient processing of many markets
    - **Flexible**: Can adjust weighting between temporal and static features
    
    #### 🔄 Recommended Usage:
    
    - Run monthly to update segmentation
    - Use 6-12 months of historical data
    - Start with 5-7 segments
    - Monitor silhouette score (>0.3 is good)
    - Analyze segment characteristics to inform strategy
    
    ---
    
    **Version:** 2.0 (Hybrid)  
    **Last Updated:** 2025-11-29  
    **Model:** LSTM (Time-Series) + Static Features (Hybrid Approach)  
    **Clustering:** K-Means, DBSCAN on Combined Features
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>🧬 Market Segmentation Pipeline | Powered by Deep Learning & Snowflake DWH</p>
</div>
""", unsafe_allow_html=True)
