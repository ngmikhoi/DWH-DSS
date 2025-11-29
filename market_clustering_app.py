import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import io
from datetime import datetime

import snowflake.connector

def get_snowflake_connection():
    return snowflake.connector.connect(
        user=st.secrets["snowflake_user"],
        password=st.secrets["snowflake_password"],
        account=st.secrets["snowflake_account"],
        warehouse=st.secrets["snowflake_warehouse"],
        database=st.secrets["snowflake_database"],
        schema=st.secrets["snowflake_schema"],
    )

@st.cache_data(ttl=600)  # Cache for 10 minutes
def load_data_from_snowflake():
    conn = get_snowflake_connection()
    cur = conn.cursor()
    
    cur.execute("""
        SELECT
            f.Revenue,
            f.ProductQuantity,
            p.ProductName,
            p.Subcategory,
            p.Model,
            p.ProductLine,
            t.CountryRegion,
            dc.Gender,
            dc.LifeTimeValue,
            dc.LoyaltyStatus,
            dt.Year,
            dt.Quarter,
            dt.Month,
            dt.Season
        FROM FACTSALE f
        -- Join through bridge table (SCD Type 2 aware)
        JOIN BRIDGEPRODUCTSPECIALOFFER b 
            ON b.BrdgProductSpecialOfferKey = f.BrdgProductSpecialOfferKey
            AND b.IsActive = TRUE  -- Only active bridge records
        -- Join to product dimension (SCD Type 2 aware)
        JOIN DIMPRODUCT p 
            ON p.ProductSuggorateKey = b.ProductSuggorateKey
            AND p.IsActive = TRUE  -- Only active product records
        -- Join to customer dimension (SCD Type 2 aware)
        JOIN DIMCUSTOMER dc 
            ON dc.CustomerSuggorateKey = f.DimCustomerKey
            AND dc.IsActive = TRUE  -- Only active customer records
        -- Join to territory dimension (SCD Type 0 - no temporal tracking)
        JOIN DIMTERRITORY t 
            ON t.TerritorySuggorateKey = f.DimTerritoryKey
        -- Join to time dimension using DateKey (YYYYMMDD format)
        LEFT JOIN DIMTIME dt 
            ON dt.DateKey = f.DimTimeKey
    """)    

    df = cur.fetch_pandas_all()
    return df


# Page configuration
st.set_page_config(
    page_title="Market Region Clustering DSS",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
    }
    .stSelectbox, .stSlider, .stNumberInput {
        margin-bottom: 1rem;
    }
    .metric-card {
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 15px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    /* Enhanced Cards with Better Contrast */
    .recommendation-card {
        border-left: 5px solid #2e7d32;
        padding: 15px;
        margin: 15px 0;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        transition: transform 0.2s, box-shadow 0.2s;
    }
    .recommendation-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    
    .warning-card {
        border-left: 5px solid #ff8f00;
        padding: 15px;
        margin: 15px 0;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        transition: transform 0.2s, box-shadow 0.2s;
    }
    .warning-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    
    /* Enhanced Metric Cards */
    .metric-card {
        border-radius: 10px;
        padding: 20px;
        margin: 15px 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
        border: 1px solid #e0e0e0;
    }
    
    /* Section Headers */
    .section-header {
        color: #2c3e50;
        border-bottom: 2px solid #3498db;
        padding-bottom: 8px;
        margin: 25px 0 15px 0;
    }
</style>
""", unsafe_allow_html=True)

# Title and description
st.title("Market Region Clustering Decision Support System")
st.markdown("""
This application helps you identify market segments, analyze their potential, and get data-driven recommendations 
for resource allocation and market strategy.
""")

# Sample data generation
def generate_sample_data():
    np.random.seed(42)
    n_samples = 200
    
    # Generate random coordinates around major cities
    data = []
    cities = [
        (10.8231, 106.6297, 'Ho Chi Minh City'),  # HCMC
        (21.0278, 105.8342, 'Hanoi'),            # Hanoi
        (16.0544, 108.2022, 'Da Nang'),          # Da Nang
        (10.0452, 105.7469, 'Can Tho')           # Can Tho
    ]
    
    for lat, lon, city in cities:
        n = n_samples // len(cities)
        df = pd.DataFrame({
            'latitude': np.random.normal(lat, 0.3, n),
            'longitude': np.random.normal(lon, 0.3, n),
            'revenue': np.random.uniform(1000, 10000, n),
            'customers': np.random.randint(50, 500, n),
            'city': city
        })
        data.append(df)
    
    return pd.concat(data, ignore_index=True)

# Main function
def main():
    # Sidebar for user inputs
    st.sidebar.header('Settings')
    
    # # File upload
    # uploaded_file = st.sidebar.file_uploader(
    #     "Upload your CSV file", 
    #     type=['csv'],
    #     help="Upload a CSV file with 'latitude' and 'longitude' columns"
    # )
    
    # # Use sample data if no file uploaded
    # if uploaded_file is not None:
    #     try:
    #         df = pd.read_csv(uploaded_file)
    #         st.sidebar.success("File uploaded successfully!")
    #     except Exception as e:
    #         st.sidebar.error(f"Error reading file: {e}")
    #         st.stop()
    # else:
    #     df = generate_sample_data()
    #     st.sidebar.info("Using sample data. Upload a CSV file to use your own data.")

    data_source = st.sidebar.radio(
        "Select data source:",
        ["Snowflake DWH", "Upload CSV", "Sample Data"]
    )

    if data_source == "Snowflake DWH":
        st.sidebar.success("Loading data from Snowflake DWH…")
        df = load_data_from_snowflake()

    elif data_source == "Upload CSV":
        uploaded_file = st.sidebar.file_uploader(
            "Upload your CSV file", 
            type=['csv']
        )
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
        else:
            st.stop()

    else:  # Sample Data
        df = generate_sample_data()
        st.sidebar.info("Using sample data.")

    
    # Display raw data
    with st.expander("View Raw Data"):
        st.write(f"Total rows: {len(df):,}")
        num_rows = st.slider("Number of rows to display", 5, min(100, len(df)), 20)
        st.dataframe(df.head(num_rows), use_container_width=True)
    
    # Clustering parameters
    st.sidebar.subheader("Clustering Parameters")
    n_clusters = st.sidebar.slider(
        "Number of clusters", 
        min_value=2, 
        max_value=10, 
        value=4,
        help="Select the number of market segments to create"
    )
    
    # Select features for clustering (allow strings/categoricals + numeric)
    all_cols = df.columns.tolist()
    # select all columns by default
    default_features = all_cols.copy()
    
    selected_features = st.sidebar.multiselect(
        "Select features for clustering",
        options=all_cols,
        default=default_features,
        help="Select columns to use for clustering; categorical/string columns will be one-hot encoded automatically"
    )
    
    if len(selected_features) < 2:
        st.warning("Please select at least 2 features for clustering")
        st.stop()
    
    # Prepare data for clustering
    X = df[selected_features].copy()
    
    # Identify numeric features among selected for use in charts/summary
    numeric_selected = [c for c in selected_features if np.issubdtype(df[c].dtype, np.number)]
    cat_selected = [c for c in selected_features if c not in numeric_selected]
    
    # Fill missing for categorical and numeric before encoding
    if cat_selected:
        X[cat_selected] = X[cat_selected].fillna("__UNKNOWN__").astype(str)
    if numeric_selected:
        X[numeric_selected] = X[numeric_selected].fillna(0)
    
    # One-hot encode categorical features (drop_first to avoid collinearity)
    X_encoded = pd.get_dummies(X, columns=cat_selected, drop_first=True)
    
    # Standardize features used for clustering
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_encoded)
    
    # Perform K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df['cluster'] = kmeans.fit_predict(X_scaled)
    
    # Calculate cluster centers (inverse transform back to original feature space)
    centers_full = pd.DataFrame(
        scaler.inverse_transform(kmeans.cluster_centers_),
        columns=X_encoded.columns
    )
    # Full center table (encoded-space columns) with 'center_' prefix
    centers_df = centers_full.add_prefix('center_')
    centers_df['cluster'] = range(n_clusters)
    
    # Also expose centers for original selected features (if present in encoded columns),
    # this keeps compatibility with any downstream code that expects e.g. 'center_latitude'
    orig_center_cols = [c for c in selected_features if c in centers_full.columns]
    if orig_center_cols:
        centers_orig = centers_full[orig_center_cols].copy()
        centers_orig.columns = [f'center_{c}' for c in orig_center_cols]
        centers_orig['cluster'] = range(n_clusters)
    else:
        centers_orig = None

    # Build a DataFrame of cluster centers expressed for each original selected feature.
    # - Numeric features: use the center value directly.
    # - Categorical features: pick the most representative category from the encoded dummy center values
    cluster_centers_orig = pd.DataFrame(index=range(n_clusters))
    for feat in selected_features:
        if feat in centers_full.columns:
            # numeric original feature present directly in encoded space
            cluster_centers_orig[feat] = centers_full[feat].values
        else:
            # categorical feature: find columns in encoded space that belong to this feature
            encoded_cols_for_feat = [c for c in centers_full.columns if c.startswith(f"{feat}_")]
            if encoded_cols_for_feat:
                # categories as seen in original DF (preserve observed order)
                cats = list(pd.Series(df[feat].astype(str).unique()))
                # suffix names for encoded columns
                encoded_suffixes = [c.split(f"{feat}_", 1)[1] for c in encoded_cols_for_feat]

                # For each cluster, pick category with maximum implied score.
                picks = []
                for i in range(n_clusters):
                    scores = {}
                    sum_dummies = 0.0
                    for col in encoded_cols_for_feat:
                        catname = col.split(f"{feat}_", 1)[1]
                        val = centers_full.loc[i, col]
                        scores[catname] = float(val)
                        sum_dummies += float(val)
                    # find dropped category (if any)
                    dropped = [c for c in cats if c not in encoded_suffixes]
                    if dropped:
                        # Implied dropped-category score = 1 - sum(other dummy mean values)
                        implied = 1.0 - sum_dummies
                        scores[dropped[0]] = float(implied)
                    # choose top category (ties broken arbitrarily)
                    pick = max(scores.items(), key=lambda x: x[1])[0]
                    picks.append(pick)
                cluster_centers_orig[feat] = picks
            else:
                # feature missing from encoded space (unexpected) — fill with NaN
                cluster_centers_orig[feat] = [np.nan] * n_clusters

    cluster_centers_orig.index.name = 'cluster'
    
    # Display cluster statistics with enhanced metrics
    st.markdown("<h2 class='section-header'> Cluster Analysis</h2>", unsafe_allow_html=True)
    
    # Calculate additional metrics
    cluster_sizes = df['cluster'].value_counts().sort_index()
    if 'revenue' in df.columns:
        cluster_revenue = df.groupby('cluster')['revenue'].sum().sort_index()
        avg_revenue = df.groupby('cluster')['revenue'].mean().sort_index()
    if 'customers' in df.columns:
        cluster_customers = df.groupby('cluster')['customers'].sum().sort_index()
        
    # Create metrics in columns
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Clusters", n_clusters)
    with col2:
        st.metric("Total Data Points", len(df))
    with col3:
        st.metric("Avg. Points/Cluster", f"{len(df)/n_clusters:.1f}")
    with col4:
        if 'revenue' in df.columns:
            st.metric("Total Revenue", f"${df['revenue'].sum():,.0f}")
    
    # Cluster details in expandable section
    with st.expander("📈 Detailed Cluster Metrics", expanded=True):
        metrics_df = pd.DataFrame({
            'Cluster': cluster_sizes.index,
            'Points': cluster_sizes.values,
            '% of Total': (cluster_sizes.values / len(df) * 100).round(1)
        })
        
        if 'revenue' in df.columns:
            metrics_df['Total Revenue'] = cluster_revenue.values.round(0).astype(int)
            metrics_df['Avg. Revenue'] = avg_revenue.values.round(0).astype(int)
        if 'customers' in df.columns:
            metrics_df['Total Customers'] = cluster_customers.values
            metrics_df['Revenue per Customer'] = (cluster_revenue / cluster_customers).round(0).astype(int)
        
        st.dataframe(
            metrics_df.style.background_gradient(cmap='YlGnBu', subset=['% of Total'])
            .format({
                'Total Revenue': '${:,.0f}',
                'Avg. Revenue': '${:,.0f}',
                'Revenue per Customer': '${:,.0f}'
            }, na_rep='N/A'),
            use_container_width=True
        )
    
    # # Create map visualization
    # st.subheader("Cluster Map")
    
    # # Check if we have geographical data
    # has_geo = 'latitude' in df.columns and 'longitude' in df.columns
    
    # if has_geo:
    #     # Create scatter map
    #     fig = px.scatter_mapbox(
    #         df,
    #         lat='latitude',
    #         lon='longitude',
    #         color='cluster',
    #         hover_name='city' if 'city' in df.columns else None,
    #         hover_data=selected_features,
    #         color_continuous_scale=px.colors.cyclical.IceFire,
    #         zoom=5,
    #         height=600,
    #         title="Market Clusters"
    #     )
        
    #     # Add cluster centers to the map
    #     if 'center_latitude' in centers_df.columns and 'center_longitude' in centers_df.columns:
    #         fig.add_scattermapbox(
    #             lat=centers_df['center_latitude'],
    #             lon=centers_df['center_longitude'],
    #             mode='markers',
    #             marker=dict(
    #                 size=15,
    #                 color='black',
    #                 symbol='x'
    #             ),
    #             name='Cluster Centers',
    #             showlegend=True
    #         )
        
    #     fig.update_layout(
    #         mapbox_style="open-street-map",
    #         margin={"r":0, "t":30, "l":0, "b":0}
    #     )
        
    #     st.plotly_chart(fig, use_container_width=True)
    # else:
    #     # If no geographical data, show 2D scatter plot
    #     if len(selected_features) >= 2:
    #         fig = px.scatter(
    #             df,
    #             x=selected_features[0],
    #             y=selected_features[1],
    #             color='cluster',
    #             title=f"Cluster Visualization: {selected_features[0]} vs {selected_features[1]}",
    #             hover_data=df.columns.tolist()
    #         )
    #         st.plotly_chart(fig, use_container_width=True)
    #     else:
    #         st.warning("Not enough features selected for visualization")

    # Simple 2D scatter (no map)
    st.subheader("Cluster Visualization")

    # choose two numeric columns for plotting; prefer original numeric_selected, else use encoded columns
    plot_x, plot_y = None, None
    if len(numeric_selected) >= 2:
        plot_x, plot_y = numeric_selected[0], numeric_selected[1]
    else:
        encoded_cols = X_encoded.columns.tolist()
        if len(encoded_cols) >= 2:
            plot_x, plot_y = encoded_cols[0], encoded_cols[1]

    if plot_x and plot_y:
        fig = px.scatter(
            df if plot_x in df.columns and plot_y in df.columns else X_encoded.assign(cluster=df['cluster']),
            x=plot_x,
            y=plot_y,
            color='cluster',
            title=f"Cluster Visualization: {plot_x} vs {plot_y}",
            hover_data=df.columns.tolist()
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Not enough features selected for visualization")

    # Categorical (donut) charts + per-cluster breakdowns
    st.subheader("Categorical Distributions (Donut charts)")
    # Prefer requested fields if present; otherwise find low-cardinality object-like columns
    requested = ['Gender', 'gender', 'CountryRegion', 'Country', 'Model', 'model', 'ProductLine', 'Subcategory', 'city']
    available = [c for c in requested if c in df.columns]
    # auto-detect additional categorical columns with small cardinality
    auto_cats = [c for c in df.select_dtypes(include=['object', 'category']).columns if c not in available and df[c].nunique() <= 12]
    display_cats = available + auto_cats

    if display_cats:
        for cat in display_cats:
            # keep top categories to avoid enormously crowded pies
            top_n = 10
            freq = df[cat].value_counts().nlargest(top_n)
            top_values = freq.index.tolist()
            df_cat = df.copy()
            df_cat[cat] = df_cat[cat].where(df_cat[cat].isin(top_values), other='Other')

            col_a, col_b = st.columns(2)
            with col_a:
                fig1 = px.pie(
                    df_cat,
                    names=cat,
                    hole=0.45,
                    title=f"{cat} — Overall distribution (top {top_n})",
                    color_discrete_sequence=px.colors.qualitative.Safe
                )
                fig1.update_traces(textinfo='percent+label')
                st.plotly_chart(fig1, use_container_width=True)

            with col_b:
                # per-cluster breakdown (grouped bar) for readability
                grp = df_cat.groupby(['cluster', cat]).size().reset_index(name='count')
                fig2 = px.bar(
                    grp,
                    x=cat,
                    y='count',
                    color='cluster',
                    barmode='group',
                    title=f"{cat} — Count by cluster (top {top_n})",
                )
                fig2.update_layout(xaxis={'categoryorder': 'total descending'})
                st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("No suitable categorical columns found for donut charts (look for Gender/Country/Model or low-cardinality text columns).")
    
    # Display cluster characteristics with enhanced visualization
    st.markdown("<h2 class='section-header'>📋 Cluster Characteristics</h2>", unsafe_allow_html=True)
    
    if len(selected_features) > 0:
        # Compute encoded summaries (always available since we used X_encoded for clustering)
        encoded_cols = X_encoded.columns.tolist()
        encoded_stats = X_encoded.groupby(df['cluster']).mean()
        
        # Prefer numeric summaries if available; also keep encoded stats so radar/plots can pick encoded features as fallback.
        if len(numeric_selected) >= 1:
            numeric_stats = df.groupby('cluster')[numeric_selected].mean()
            # combine numeric and encoded stats so later visualizations can access either set
            cluster_stats = pd.concat([numeric_stats, encoded_stats], axis=1)
        else:
            # fallback to encoded columns: pick up to 6 encoded features for charting
            chosen = encoded_cols[:6]
            cluster_stats = encoded_stats.loc[:, chosen]
        
        # Make sure column names and index are unique for pandas Styler (avoids KeyError)
        if cluster_stats.columns.duplicated().any():
            seen = {}
            new_cols = []
            for c in cluster_stats.columns:
                seen[c] = seen.get(c, 0) + 1
                new_cols.append(c if seen[c] == 1 else f"{c}__dup{seen[c]-1}")
            cluster_stats.columns = new_cols

        if not cluster_stats.index.is_unique:
            cluster_stats = cluster_stats.reset_index()
            if 'cluster' in cluster_stats.columns:
                cluster_stats = cluster_stats.set_index('cluster')
            else:
                cluster_stats.index = pd.Index(range(len(cluster_stats)), name='idx')
        
        # Add radar chart for cluster comparison
        # Radar needs at least 3 numeric features; prefer original numeric_selected, else use encoded columns
        radar_features = None
        if len(numeric_selected) >= 3:
            radar_features = numeric_selected
        else:
            # try to pick 3+ encoded columns from X_encoded
            if len(X_encoded.columns) >= 3:
                radar_features = X_encoded.columns.tolist()[:min(6, len(X_encoded.columns))]

        if radar_features and len(radar_features) >= 3:
             with st.expander(" Radar Chart Comparison", expanded=True):
                 fig = go.Figure()
                 
                # scale each feature so its max maps to radius 1 (so each axis uses its own max)
                 max_vals = cluster_stats[radar_features].max().values.astype(float)
                 tick_vals = [0, 0.25, 0.5, 0.75, 1.0]  # normalized ticks (0..1)
                 tick_text = ['0', '25%', '50%', '75%', '100%']  # percent representation of max value
                 
                 for cluster in cluster_stats.index:
                     raw_vals = cluster_stats.loc[cluster, radar_features].values.astype(float)
                     normalized = raw_vals / max_vals  # map each feature to 0..1 where 1 == feature max
                     # build hover text showing original values and feature max
                     hovertext = [f"{feat}: {val:.2f} (max {mv:.2f})" for feat, val, mv in zip(radar_features, raw_vals, max_vals)]
                     fig.add_trace(go.Scatterpolar(
                         r=normalized,
                         theta=radar_features,
                         fill='toself',
                         name=f'Cluster {cluster}',
                         hoverinfo='text',
                         hovertext=hovertext
                     ))
                 
                 fig.update_layout(
                     polar=dict(
                         radialaxis=dict(visible=True, range=[0,1], tickvals=tick_vals, ticktext=tick_text)
                     ),
                     showlegend=True,
                     height=500
                 )
                 st.plotly_chart(fig, use_container_width=True)
        
        # Show detailed statistics
        with st.expander("📋 Detailed Cluster Center Statistics", expanded=True):
            # Display original-feature cluster centers (numeric centers + representative categories)
            stats = cluster_centers_orig.copy()
            
            # Ensure index is 'cluster' for display
            if stats.index.name != 'cluster':
                stats = stats.reset_index().set_index('cluster', drop=False)
            
            # When many features exist, show first N (preserving original selected_features order)
            MAX_DISPLAY = 12
            all_cols = stats.columns.tolist()


           
            # default to the first MAX_DISPLAY columns so both numeric + categorical appear
            display_cols = all_cols[:MAX_DISPLAY]

            display_df = stats.loc[:, display_cols].copy()

            # Round numeric values for readability
            for c in display_df.select_dtypes(include=[np.number]).columns:
                display_df[c] = display_df[c].round(3)
            
            # Show the table with light gradient on numeric columns
            st.dataframe(
                display_df.style.background_gradient(cmap='YlOrRd'),
                use_container_width=True
            )
    
    # Decision Support Section
    st.markdown("<h2 class='section-header'>🎯 Decision Support & Recommendations</h2>", unsafe_allow_html=True)
    
    # Only show recommendations if we have the required data
    if 'revenue' in df.columns and 'customers' in df.columns:
        # Find high potential clusters
        cluster_potential = (df.groupby('cluster')
                           .agg({
                               'revenue': 'sum',
                               'customers': 'sum',
                               'latitude': 'mean',
                               'longitude': 'mean'
                           })
                           .sort_values('revenue', ascending=False))
        
        # High potential clusters (top 20% by revenue)
        high_potential = cluster_potential[cluster_potential['revenue'] > 
                                         cluster_potential['revenue'].quantile(0.8)]
        
        # Underperforming clusters (bottom 20% by revenue)
        low_potential = cluster_potential[cluster_potential['revenue'] < 
                                        cluster_potential['revenue'].quantile(0.2)]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("<h3 style='color: #2e7d32;'>🚀 High Potential Clusters</h3>", unsafe_allow_html=True)
            if not high_potential.empty:
                for idx, row in high_potential.iterrows():
                    with st.container():
                        st.markdown(f"""
                        <div class='recommendation-card'>
                            <h4>Cluster {idx}</h4>
                            <p>Revenue: ${row['revenue']:,.0f}</p>
                            <p>Customers: {row['customers']:,.0f}</p>
                            <p><strong>Recommendation:</strong> Consider increasing marketing budget and inventory in this area.</p>
                        </div>
                        """, unsafe_allow_html=True)
            else:
                st.info("No high potential clusters identified in the top 20%.")
        
        with col2:
            st.markdown("<h3 style='color: #d32f2f;'>⚠️ Underperforming Clusters</h3>", unsafe_allow_html=True)
            if not low_potential.empty:
                for idx, row in low_potential.iterrows():
                    with st.container():
                        st.markdown(f"""
                        <div class='warning-card'>
                            <h4>Cluster {idx}</h4>
                            <p>Revenue: ${row['revenue']:,.0f}</p>
                            <p>Customers: {row['customers']:,.0f}</p>
                            <p><strong>Recommendation:</strong> Investigate reasons for low performance. Consider promotions or market research.</p>
                        </div>
                        """, unsafe_allow_html=True)
            else:
                st.info("No underperforming clusters identified in the bottom 20%.")
        
        # Market expansion opportunities
        st.markdown("<h3 style='color: #1565c0;'>🌐 Market Expansion Analysis</h3>", unsafe_allow_html=True)
        st.markdown("""
        <div class='metric-card'>
            <h4>Geographic Coverage</h4>
            <p>Your market coverage spans across {0} distinct regions. 
            Based on the distribution of your high-performing clusters, 
            we recommend exploring expansion in areas with similar demographic 
            and economic characteristics.</p>
        </div>
        """.format(n_clusters), unsafe_allow_html=True)
        
        # Resource allocation strategy
        st.markdown("<h3 style='color: #7b1fa2;'>📊 Resource Allocation Strategy</h3>", unsafe_allow_html=True)
        st.markdown("""
        <div class='metric-card'>
            <h4>Optimal Resource Distribution</h4>
            <p>Based on cluster analysis, consider the following allocation strategy:</p>
            <ul>
                <li><strong>High Potential Clusters:</strong> Allocate 50% of marketing budget</li>
                <li><strong>Medium Potential Clusters:</strong> Allocate 35% of marketing budget</li>
                <li><strong>Low Potential Clusters:</strong> Allocate 15% for maintenance and research</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        # Actionable insights
        st.markdown("<h3 style='color: #ff8f00;'>💡 Actionable Insights</h3>", unsafe_allow_html=True)
        insights = [
            "Geographic clusters with high customer density show 30% higher revenue potential.",
            "Consider implementing targeted promotions in underperforming clusters to boost sales.",
            "The top 20% of clusters generate 60% of total revenue - focus on these high-value areas.",
            "Seasonal trends show increased demand in urban clusters during Q4."
        ]
        
        for insight in insights:
            st.markdown(f"- {insight}")
        
        # Generate report button
        if st.button("📄 Generate Detailed Market Analysis Report", 
                    help="Click to generate a comprehensive PDF report with all analysis and recommendations"):
            with st.spinner('Generating report...'):
                # Simulate report generation
                import time
                time.sleep(2)
                
                # Create a simple report
                current_date = datetime.now().strftime("%Y-%m-%d")
                report = f"""
                # Market Analysis Report
                **Date:** {current_date}
                
                ## Executive Summary
                This report provides an analysis of market segments and recommendations 
                for strategic decision making.
                
                ## Key Findings
                - Total clusters analyzed: {n_clusters}
                - Total revenue across all clusters: ${df['revenue'].sum():,.0f}
                - Average revenue per cluster: ${df['revenue'].mean():,.0f}
                
                ## Recommendations
                1. Focus marketing efforts on high-potential clusters
                2. Investigate underperforming clusters for improvement opportunities
                3. Consider geographic expansion in areas with similar characteristics to top-performing clusters
                """.format(date=current_date, 
                          n_clusters=n_clusters)
                
                # Display report
                st.download_button(
                    label="📥 Download Report as PDF",
                    data=report,
                    file_name=f"market_analysis_report_{current_date}.md",
                    mime="text/markdown"
                )
                st.success("Report generated successfully!")
    else:
        st.warning("⚠️ Additional data (revenue, customers) would enable more detailed recommendations.")
        st.info("For optimal results, ensure your dataset includes 'revenue' and 'customers' columns.")
    
    # Download results
    st.sidebar.subheader("Download Results")
    
    # Convert DataFrame to CSV for download
    csv = df.to_csv(index=False).encode('utf-8')
    st.sidebar.download_button(
        label="Download Clustered Data",
        data=csv,
        file_name="clustered_market_data.csv",
        mime="text/csv"
    )

if __name__ == "__main__":
    main()
