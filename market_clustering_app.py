import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import io
import zipfile
import matplotlib.pyplot as plt
from io import BytesIO
from datetime import datetime
import re
import json
import snowflake.connector
import sys
import subprocess

# Import the pipeline class
from market_segmentation_pipeline import MarketSegmentationPipeline

# Page configuration
st.set_page_config(
    page_title="Unified Market Segmentation DSS",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main { padding: 1rem; }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        font-weight: bold;
    }
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .success-card {
        background-color: #d4fc79;
        background-image: linear-gradient(315deg, #d4fc79 0%, #96e6a1 74%);
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
        color: #1f2937;
    }
    h1, h2, h3 {
        color: #1f2937;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# HELPER FUNCTIONS
# -----------------------------------------------------------------------------

def get_snowflake_connection():
    """Get Snowflake connection using secrets"""
    try:
        return snowflake.connector.connect(
            user=st.secrets["snowflake_user"],
            password=st.secrets["snowflake_password"],
            account=st.secrets["snowflake_account"],
            warehouse=st.secrets["snowflake_warehouse"],
            database=st.secrets["snowflake_database"],
            schema=st.secrets["snowflake_schema"],
        )
    except Exception as e:
        st.error(f"❌ Error connecting to Snowflake: {e}")
        return None



@st.cache_data(ttl=600)
def load_official_segments(include_historical=False):
    """Load pre-computed segments from FACTMARKETSEGMENTATION
    
    Args:
        include_historical: If True, load all historical data. If False, only latest version.
    """
    conn = get_snowflake_connection()
    if not conn: return None
    
    try:
        cur = conn.cursor()
        
        if include_historical:
            # Load all historical segmentation data
            cur.execute("""
                SELECT 
                    f.DateKey,
                    f.MarketNameKey,
                    f.SegmentID,
                    f.Revenue,
                    f.Quantity,
                    f.ProductRange,
                    f.ModelVersion,
                    f.EmbeddingJSON,
                    f.ConfidenceScore,
                    f.CreatedAt
                FROM FACTMARKETSEGMENTATION f
                ORDER BY f.DateKey, f.MarketNameKey
            """)
        else:
            # Get the latest segmentation run (by model version)
            cur.execute("""
                WITH LatestVersion AS (
                    SELECT ModelVersion, MAX(CreatedAt) AS MaxCreatedAt
                    FROM FACTMARKETSEGMENTATION
                    GROUP BY ModelVersion
                    ORDER BY MaxCreatedAt DESC
                    LIMIT 1
                )
                SELECT 
                    f.DateKey,
                    f.MarketNameKey,
                    f.SegmentID,
                    f.Revenue,
                    f.Quantity,
                    f.ProductRange,
                    f.ModelVersion,
                    f.EmbeddingJSON,
                    f.ConfidenceScore,
                    f.CreatedAt
                FROM FACTMARKETSEGMENTATION f
                INNER JOIN LatestVersion lv ON f.ModelVersion = lv.ModelVersion
                ORDER BY f.MarketNameKey
            """)
        
        df = cur.fetch_pandas_all()
        # Normalize columns to uppercase for consistency
        df.columns = df.columns.str.upper()
        return df
    finally:
        conn.close()

@st.cache_data(ttl=600)
def load_historical_trends():
    """Load time-series data for segment trend analysis"""
    conn = get_snowflake_connection()
    if not conn: return None
    
    try:
        cur = conn.cursor()
        cur.execute("""
            SELECT 
                f.DateKey,
                f.SegmentID,
                SUM(f.Revenue) AS TotalRevenue,
                SUM(f.Quantity) AS TotalQuantity,
                COUNT(DISTINCT f.MarketNameKey) AS MarketCount,
                AVG(f.ConfidenceScore) AS AvgConfidence
            FROM FACTMARKETSEGMENTATION f
            GROUP BY f.DateKey, f.SegmentID
            ORDER BY f.DateKey, f.SegmentID
        """)
        df = cur.fetch_pandas_all()
        df.columns = df.columns.str.upper()
        return df
    finally:
        conn.close()

def find_column_case_insensitive(df, name):
    if df is None: return None
    for c in df.columns:
        if str(c).strip().lower() == str(name).strip().lower():
            return c
    return None

# -----------------------------------------------------------------------------
# MODE 1: OFFICIAL DASHBOARD
# -----------------------------------------------------------------------------

def render_official_dashboard():
    st.header("📊 Official Market Segmentation Dashboard")
    st.markdown("Insights from the latest **LSTM Deep Learning Model** run.")
    
    # Add refresh button
    if st.button("🔄 Refresh Data", help="Clear cache and reload latest segmentation"):
        st.cache_data.clear()
        st.rerun()
    
    df = load_official_segments()
    
    if df is None or len(df) == 0:
        st.warning("⚠️ No segmentation data found in FACTMARKETSEGMENTATION. Run the pipeline first!")
        return

    # ========================================================================
    # FILTERS
    # ========================================================================
    st.sidebar.subheader("🔍 Filters")
    
    # Create MonthKey for consistent filtering (YYYYMM)
    df['MONTHKEY'] = df['DATEKEY'].apply(lambda x: int(str(x)[:6]))
    
    # Date Filter (New)
    all_months = sorted(df['MONTHKEY'].unique(), reverse=True)
    formatted_months = [f"{str(m)[:4]}-{str(m)[4:]}" for m in all_months]
    
    selected_month_str = st.sidebar.selectbox(
        "Select Month (Snapshot)",
        options=formatted_months,
        index=0,
        help="Select the month to view segmentation results for"
    )
    
    # Convert selected "YYYY-MM" back to YYYYMM integer
    selected_month_key = int(selected_month_str.replace("-", ""))
    
    # Filter by MonthKey FIRST
    date_filtered_df = df[df['MONTHKEY'] == selected_month_key]
    
    # Segment Filter
    all_segments = sorted(date_filtered_df['SEGMENTID'].unique())
    selected_segments = st.sidebar.multiselect(
        "Select Segments",
        options=all_segments,
        default=all_segments,
        help="Filter by segment ID"
    )
    
    # Territory Filter
    all_territories = sorted(date_filtered_df['MARKETNAMEKEY'].unique())
    selected_territories = st.sidebar.multiselect(
        "Select Territories",
        options=all_territories,
        default=all_territories,
        help="Filter by country/region"
    )
    
    # Apply filters
    filtered_df = date_filtered_df[
        (date_filtered_df['SEGMENTID'].isin(selected_segments)) &
        (date_filtered_df['MARKETNAMEKEY'].isin(selected_territories))
    ]
    
    if len(filtered_df) == 0:
        st.warning("⚠️ No data matches the selected filters.")
        return
    
    # ========================================================================
    # A. SEGMENT OVERVIEW
    # ========================================================================
    st.subheader("📈 A. Segment Overview")
    
    # Key Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    num_segments = filtered_df['SEGMENTID'].nunique()
    total_markets = len(filtered_df)
    total_revenue = filtered_df['REVENUE'].sum()
    total_quantity = filtered_df['QUANTITY'].sum()
    
    with col1:
        st.metric("Total Segments", num_segments)
    with col2:
        st.metric("Total Markets", total_markets)
    with col3:
        st.metric("Total Revenue", f"${total_revenue:,.0f}")
    with col4:
        st.metric("Total Quantity", f"{total_quantity:,.0f}")
    
    # Segment-level metrics table
    st.markdown("#### Segment Metrics")
    segment_metrics = filtered_df.groupby('SEGMENTID').agg({
        'MARKETNAMEKEY': 'count',  # Number of markets
        'REVENUE': ['sum', 'mean'],
        'QUANTITY': ['sum', 'mean']
    }).round(2)
    
    segment_metrics.columns = ['Markets Count', 'Total Revenue', 'Avg Revenue', 
                               'Total Quantity', 'Avg Quantity']
    segment_metrics = segment_metrics.reset_index()
    segment_metrics = segment_metrics.sort_values('Total Revenue', ascending=False)
    
    st.dataframe(segment_metrics, use_container_width=True)

    st.markdown("---")

    # ========================================================================
    # B. SEGMENT COMPARISON
    # ========================================================================
    st.subheader("📊 B. Segment Comparison")
    
    # ROW 1: SHARES (Donut Charts)
    col_share1, col_share2 = st.columns(2)
    
    with col_share1:
        st.markdown("#### Revenue Share")
        rev_by_seg = filtered_df.groupby('SEGMENTID')['REVENUE'].sum().reset_index()
        rev_by_seg['SEGMENTID'] = rev_by_seg['SEGMENTID'].astype(str)
        
        fig_rev_share = px.pie(
            rev_by_seg,
            values='REVENUE',
            names='SEGMENTID',
            hole=0.4,
            title="Revenue Distribution (%)",
            color='SEGMENTID',
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        fig_rev_share.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig_rev_share, use_container_width=True)

    with col_share2:
        st.markdown("#### Market Count Distribution")
        market_counts = filtered_df['SEGMENTID'].value_counts().reset_index()
        market_counts.columns = ['SEGMENTID', 'Count']
        market_counts['SEGMENTID'] = market_counts['SEGMENTID'].astype(str)
        
        fig_mkt_share = px.pie(
            market_counts,
            values='Count',
            names='SEGMENTID',
            hole=0.4,
            title="Proportion of Markets per Segment",
            color='SEGMENTID',
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        fig_mkt_share.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig_mkt_share, use_container_width=True)
        st.caption("💡 **Insight:** Compare this with Revenue Share. A segment with low Market Count but high Revenue Share indicates high-performing markets.")
    
    st.markdown("---")
    
    # ROW 2: PERFORMANCE (Bar Charts)
    col_perf1, col_perf2 = st.columns(2)
    
    with col_perf1:
        st.markdown("#### Efficiency (Avg Revenue/Market)")
        avg_rev_by_seg = filtered_df.groupby('SEGMENTID')['REVENUE'].mean().reset_index()
        avg_rev_by_seg.columns = ['SEGMENTID', 'AVG_REVENUE']
        avg_rev_by_seg = avg_rev_by_seg.sort_values('AVG_REVENUE', ascending=False)
        avg_rev_by_seg['SEGMENTID'] = avg_rev_by_seg['SEGMENTID'].astype(str)
        
        fig_avg = px.bar(
            avg_rev_by_seg,
            x='SEGMENTID',
            y='AVG_REVENUE',
            labels={'SEGMENTID': 'Segment', 'AVG_REVENUE': 'Avg Revenue ($)'},
            color='AVG_REVENUE',
            color_continuous_scale='Greens',
            text='AVG_REVENUE'
        )
        fig_avg.update_traces(texttemplate='$%{text:,.0f}', textposition='outside')
        st.plotly_chart(fig_avg, use_container_width=True)
        
    with col_perf2:
        st.markdown("#### Total Quantity Sold")
        qty_by_seg = filtered_df.groupby('SEGMENTID')['QUANTITY'].sum().reset_index()
        qty_by_seg = qty_by_seg.sort_values('QUANTITY', ascending=False)
        qty_by_seg['SEGMENTID'] = qty_by_seg['SEGMENTID'].astype(str)
        
        fig_qty = px.bar(
            qty_by_seg,
            x='SEGMENTID',
            y='QUANTITY',
            labels={'SEGMENTID': 'Segment', 'QUANTITY': 'Total Quantity'},
            color='QUANTITY',
            color_continuous_scale='Blues',
            text='QUANTITY'
        )
        fig_qty.update_traces(texttemplate='%{text:,.0f}', textposition='outside')
        st.plotly_chart(fig_qty, use_container_width=True)
    
    # ========================================================================
    # TEMPORAL TRENDS (if historical data available)
    # ========================================================================
    st.markdown("#### 📈 Temporal Trends Over Time")
    
    trends_df = load_historical_trends()
    
    if trends_df is not None and len(trends_df) > 0:
        # Convert DateKey to Datetime for better plotting
        trends_df['DATE'] = pd.to_datetime(trends_df['DATEKEY'].astype(str), format='%Y%m%d')
        
        # Check if we have multiple time periods
        unique_dates = trends_df['DATEKEY'].nunique()
        
        if unique_dates > 1:
            # Filter trends by selected segments
            trends_filtered = trends_df[trends_df['SEGMENTID'].isin(selected_segments)]
            
            if len(trends_filtered) > 0:
                col_trend1, col_trend2 = st.columns(2)
                
                with col_trend1:
                    # Revenue trend over time
                    fig_trend_rev = px.line(
                        trends_filtered,
                        x='DATE',
                        y='TOTALREVENUE',
                        color='SEGMENTID',
                        markers=True,
                        title="Revenue Trends by Segment Over Time",
                        labels={'DATE': 'Date', 'TOTALREVENUE': 'Total Revenue', 'SEGMENTID': 'Segment'}
                    )
                    st.plotly_chart(fig_trend_rev, use_container_width=True)
                
                with col_trend2:
                    # Quantity trend over time
                    fig_trend_qty = px.line(
                        trends_filtered,
                        x='DATE',
                        y='TOTALQUANTITY',
                        color='SEGMENTID',
                        markers=True,
                        title="Quantity Trends by Segment Over Time",
                        labels={'DATE': 'Date', 'TOTALQUANTITY': 'Total Quantity', 'SEGMENTID': 'Segment'}
                    )
                    st.plotly_chart(fig_trend_qty, use_container_width=True)
                
                # Market count evolution
                fig_trend_markets = px.line(
                    trends_filtered,
                    x='DATE',
                    y='MARKETCOUNT',
                    color='SEGMENTID',
                    markers=True,
                    title="Number of Markets per Segment Over Time",
                    labels={'DATE': 'Date', 'MARKETCOUNT': 'Market Count', 'SEGMENTID': 'Segment'}
                )
                st.plotly_chart(fig_trend_markets, use_container_width=True)
            else:
                st.info("No trend data available for selected segments.")
        else:
            st.info("📊 Temporal trends require multiple time periods. Run the pipeline on different dates to see trends.")
    else:
        st.info("📊 No historical trend data available yet. Run the pipeline multiple times to track trends over time.")

    st.markdown("---")

    # ========================================================================
    # C. MARKET DETAILS BY SEGMENT (Drill-down)
    # ========================================================================
    st.subheader("🔍 C. Market Details by Segment")
    
    # Segment selector for drill-down
    selected_segment_drill = st.selectbox(
        "Select a Segment to Drill Down",
        options=sorted(filtered_df['SEGMENTID'].unique()),
        help="View detailed market information for a specific segment"
    )
    
    segment_data = filtered_df[filtered_df['SEGMENTID'] == selected_segment_drill]
    
    # Segment summary
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Markets in Segment", len(segment_data))
    with col2:
        st.metric("Segment Revenue", f"${segment_data['REVENUE'].sum():,.0f}")
    with col3:
        st.metric("Segment Quantity", f"{segment_data['QUANTITY'].sum():,.0f}")
    
    # Geographic distribution
    st.markdown(f"#### Geographic Distribution (Segment {selected_segment_drill})")
    
    geo_data = segment_data.groupby('MARKETNAMEKEY').agg({
        'REVENUE': 'sum',
        'QUANTITY': 'sum'
    }).reset_index()
    geo_data = geo_data.sort_values('REVENUE', ascending=False)
    
    fig_geo = px.bar(
        geo_data,
        x='MARKETNAMEKEY',
        y='REVENUE',
        labels={'MARKETNAMEKEY': 'Territory', 'REVENUE': 'Revenue'},
        color='REVENUE',
        color_continuous_scale='Plasma',
        title=f"Revenue by Territory in Segment {selected_segment_drill}"
    )
    fig_geo.update_xaxes(tickangle=-45)
    st.plotly_chart(fig_geo, use_container_width=True)
    
    # Market performance table
    st.markdown(f"#### Market Performance Details (Segment {selected_segment_drill})")
    
    market_details = segment_data[[
        'MARKETNAMEKEY', 'REVENUE', 'QUANTITY', 
        'PRODUCTRANGE', 'DATEKEY'
    ]].sort_values('REVENUE', ascending=False)
    
    market_details.columns = ['Territory', 'Revenue', 'Quantity', 
                              'Product Range', 'Date Key']
    
    st.dataframe(market_details, use_container_width=True)

    st.markdown("---")


    st.markdown("---")
    
    # ========================================================================
    # ADDITIONAL INFO
    # ========================================================================
    st.subheader("ℹ️ Model Information")
    col1, col2 = st.columns(2)
    with col1:
        st.info(f"**Model Version:** {filtered_df['MODELVERSION'].iloc[0]}")
    with col2:
        st.info(f"**Last Updated:** {filtered_df['CREATEDAT'].max()}")

# -----------------------------------------------------------------------------
# MODE 2: PIPELINE MANAGEMENT
# -----------------------------------------------------------------------------



def render_pipeline_management():
    st.header("⚙️ Pipeline Management")
    st.markdown("Trigger and monitor the **LSTM Market Segmentation Pipeline**.")
    
    # Configuration
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Model Settings")
        st.info("📅 **Window Size:** 5 months (fixed)")
        # st.caption("Each month will be segmented using a 5-month rolling window")
        n_clusters = st.slider("Target Segments", 2, 10, 5)
        device = st.radio("Device", ["CPU", "GPU (CUDA)"])
        
    with col2:
        st.subheader("Training Settings")
        epochs = st.slider("Epochs", 10, 200, 50)
        batch_size = st.select_slider("Batch Size", options=[8, 16, 32, 64], value=16)
    
    st.markdown("---")
    
    st.markdown("### 📊 Dynamic Monthly Segmentation")
    st.markdown("""
    This pipeline will:
    - Process **each month independently** with a 5-month rolling window
    - Track **segment movement** over time
    - Save results for each month to `FACTMARKETSEGMENTATION`
    """)
    
    if st.button("🚀 Run Monthly Segmentation", type="primary"):
        run_monthly_pipeline_logic(n_clusters, epochs, batch_size, device)

def run_monthly_pipeline_logic(n_clusters, epochs, batch_size, device):
    status_text = st.empty()
    progress_bar = st.progress(0)
    
    try:
        # Config
        snowflake_config = {
            'user': st.secrets["snowflake_user"],
            'password': st.secrets["snowflake_password"],
            'account': st.secrets["snowflake_account"],
            'warehouse': st.secrets["snowflake_warehouse"],
            'database': st.secrets["snowflake_database"],
            'schema': st.secrets["snowflake_schema"]
        }
        
        # Initialize
        status_text.text("🚀 Initializing pipeline...")
        pipeline = MarketSegmentationPipeline(
            snowflake_config=snowflake_config,
            device='cuda' if device == "GPU (CUDA)" else 'cpu'
        )
        
        # Run monthly segmentation
        status_text.text("🚀 Running monthly segmentation...")
        progress_bar.progress(10)
        
        results = pipeline.run_monthly_segmentation(
            window_size=3,
            n_clusters=n_clusters,
            epochs=epochs
        )
        
        progress_bar.progress(100)
        status_text.text("✅ Pipeline Completed!")
        
        # Show results
        st.success(f"✅ Processed {len(results)} months successfully!")
        st.markdown("### 📊 Results Summary")
        
        results_df = pd.DataFrame(results)
        results_df['Month'] = results_df.apply(lambda x: f"{int(x['year'])}-{int(x['month']):02d}", axis=1)
        results_df = results_df[['Month', 'num_markets', 'silhouette']]
        results_df.columns = ['Month', 'Markets', 'Silhouette Score']
        st.dataframe(results_df, use_container_width=True)
        
        st.balloons()
        
    except Exception as e:
        st.error(f"Pipeline failed: {e}")

# -----------------------------------------------------------------------------
# MODE 3: PLAYGROUND
# -----------------------------------------------------------------------------

@st.cache_data(ttl=3600)
def load_playground_data():
    """Load raw aggregated data for the playground"""
    conn = get_snowflake_connection()
    if not conn: return None
    
    try:
        cur = conn.cursor()
        query = """
        SELECT
            CONCAT(t.Territory_ID, ' - ', t.CountryRegion) AS Market,
            dt.Year,
            dt.Month,
            MAX(dt.DateKey) AS DateKey,
            SUM(f.Revenue) AS Revenue,
            SUM(f.ProductQuantity) AS Quantity,
            COUNT(DISTINCT f.SalesOrderID) AS OrderCount,
            COUNT(DISTINCT f.DimCustomerKey) AS CustomerCount,
            AVG(f.Revenue) AS AvgOrderValue,
            COUNT(DISTINCT p.Subcategory) AS UniqueSubcategories,
            COUNT(DISTINCT p.ProductLine) AS UniqueProductLines,
            AVG(so.DiscountPct) AS AvgDiscount
        FROM FACTSALE f
        JOIN BRIDGEPRODUCTSPECIALOFFER b 
            ON b.BrdgProductSpecialOfferKey = f.BrdgProductSpecialOfferKey
        JOIN DIMPRODUCT p 
            ON p.ProductSuggorateKey = b.ProductSuggorateKey
        JOIN DIMSPECIALOFFER so
            ON so.SpecialOfferSuggorateKey = b.SpecialOfferSuggorateKey
        JOIN DIMTERRITORY t 
            ON t.TerritorySuggorateKey = f.DimTerritoryKey
        LEFT JOIN DIMTIME dt 
            ON dt.DateKey = f.DimTimeKey
        GROUP BY t.Territory_ID, t.CountryRegion, dt.Year, dt.Month
        ORDER BY dt.Year DESC, dt.Month DESC, Market
        """
        cur.execute(query)
        df = cur.fetch_pandas_all()
        df.columns = df.columns.str.upper()
        return df
    finally:
        conn.close()

def render_playground():
    st.header("🧪 Analyst Playground")
    st.markdown("Experiment with **K-Means Clustering** on raw monthly data.")
    
    # Load Data
    with st.spinner("Loading raw data..."):
        df = load_playground_data()
        
    if df is None:
        st.error("Failed to load data.")
        return

    # Sidebar Controls
    st.sidebar.subheader("🧪 Playground Settings")
    
    # 1. Select Month
    df['MONTH_STR'] = df.apply(lambda x: f"{int(x['YEAR'])}-{int(x['MONTH']):02d}", axis=1)
    available_months = sorted(df['MONTH_STR'].unique(), reverse=True)
    
    selected_month = st.sidebar.selectbox(
        "1. Select Month",
        options=available_months,
        index=0
    )
    
    # Filter data
    month_data = df[df['MONTH_STR'] == selected_month].copy()
    st.info(f"Analyzing **{len(month_data)} markets** for {selected_month}")
    
    # 2. Select Features
    numeric_cols = ['REVENUE', 'QUANTITY', 'ORDERCOUNT', 'CUSTOMERCOUNT', 
                   'AVGORDERVALUE', 'UNIQUESUBCATEGORIES', 'UNIQUEPRODUCTLINES', 'AVGDISCOUNT']
    
    selected_features = st.sidebar.multiselect(
        "2. Select Features",
        options=numeric_cols,
        default=['REVENUE', 'QUANTITY', 'ORDERCOUNT']
    )
    
    # 3. Select K
    k = st.sidebar.slider("3. Number of Clusters (K)", 2, 10, 4)
    
    if not selected_features:
        st.warning("Please select at least one feature.")
        return
        
    # Run Clustering (Auto-run)
    try:
        # Prepare data
        X = month_data[selected_features]
        
        # Handle NaNs if any (fill with 0 for playground)
        X = X.fillna(0)
        
        # Standardize
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Cluster
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(X_scaled)
        
        month_data['CLUSTER'] = clusters
        month_data['CLUSTER'] = month_data['CLUSTER'].astype(str)
        
        # Calculate Silhouette
        if len(month_data) > k:
            score = silhouette_score(X_scaled, clusters)
            st.sidebar.metric("Silhouette Score", f"{score:.3f}")
        
        # Visuals
        st.subheader("📊 Clustering Results")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Scatter Plot
            x_axis = st.selectbox("X Axis", options=selected_features, index=0)
            y_axis = st.selectbox("Y Axis", options=selected_features, index=1 if len(selected_features) > 1 else 0)
            
            fig = px.scatter(
                month_data,
                x=x_axis,
                y=y_axis,
                color='CLUSTER',
                hover_data=['MARKET'] + selected_features,
                title=f"Clusters: {x_axis} vs {y_axis}",
                color_discrete_sequence=px.colors.qualitative.Bold
            )
            st.plotly_chart(fig, use_container_width=True)
            
        with col2:
            # Cluster Stats
            st.markdown("#### Cluster Profiles")
            stats = month_data.groupby('CLUSTER')[selected_features].mean().reset_index()
            format_dict = {col: "{:.1f}" for col in selected_features}
            st.dataframe(stats.style.format(format_dict), use_container_width=True)
            
        # Detailed Data
        st.markdown("### 📋 Detailed Market Data")
        st.dataframe(month_data[['MARKET', 'CLUSTER'] + selected_features], use_container_width=True)
        
    except Exception as e:
        st.error(f"Clustering failed: {e}")

def main():
    st.sidebar.title("🧬 Unified DSS")
    
    mode = st.sidebar.radio(
        "Select Mode:",
        ["📊 Official Dashboard", "⚙️ Segmentation Management", "🧪 Playground"]
    )
    
    st.sidebar.markdown("---")
    
    if mode == "📊 Official Dashboard":
        render_official_dashboard()
    elif mode == "⚙️ Segmentation Management":
        render_pipeline_management()
    elif mode == "🧪 Playground":
        render_playground()

if __name__ == "__main__":
    main()
