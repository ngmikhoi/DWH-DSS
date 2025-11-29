import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import io
import zipfile
import matplotlib.pyplot as plt
from io import BytesIO
from datetime import datetime
import re

import snowflake.connector
import sys
import subprocess

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
    page_icon="🧩",
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

# Function to find column by name case-insensitively
def find_column_case_insensitive(df, name):
    """Return actual column name from df matching 'name' case-insensitively, or None."""
    if df is None:
        return None
    for c in df.columns:
        if str(c).strip().lower() == str(name).strip().lower():
            return c
    return None

def pretty_label(col_name: str) -> str:
    """Return a human-friendly label for a feature/column name."""
    if col_name is None:
        return ""
    s = str(col_name)
    # common explicit mappings
    mapping = {
        'revenue': 'Revenue (USD)',
        'customers': 'Customers (count)',
        'total_revenue': 'Total Revenue (USD)',
        'avg_revenue': 'Avg. Revenue (USD)',
        'total_customers': 'Total Customers',
        'avg_customers': 'Avg. Customers',
        'productquantity': 'Product Quantity',
        'product_quantity': 'Product Quantity',
        'avg_product_qty': 'Avg Product Qty',
        'life_time_value': 'Lifetime Value',
        'lifetimevalue': 'Lifetime Value',
        'avg_ltv': 'Avg Lifetime Value',
        'loyalty': 'Loyalty',
        'loyalty_score': 'Loyalty Score',
        'latitude': 'Latitude',
        'longitude': 'Longitude',
        'city': 'City',
        'cluster': 'Cluster',
        'priority_score': 'Priority Score',
        'revenue_per_customer': 'Revenue / Customer (USD)',
    }
    if s in mapping:
        return mapping[s]
    key = s.strip()
    # normalize keys (strip, lower, remove spaces/underscores)
    k = re.sub(r'[\s_]+', '', key).lower()
    # fallback: split snake_case / camelCase to words and title-case
    # add spaces between camelCase boundaries
    spaced = re.sub(r'(?<!^)(?=[A-Z])', ' ', key).replace('_', ' ').replace('-', ' ')
    return spaced.strip().title()

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

    # detect revenue/customers columns case-insensitively so code works with different schemas
    revenue_col = find_column_case_insensitive(df, "revenue")
    customers_col = find_column_case_insensitive(df, "customers")

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
        value=6,
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
    if revenue_col is not None:
        cluster_revenue = df.groupby('cluster')[revenue_col].sum().sort_index()
        avg_revenue = df.groupby('cluster')[revenue_col].mean().sort_index()
    # Customer counts: if there's a 'customers' column use it, otherwise treat each row as 1 customer
    if customers_col is not None:
        cluster_customers = df.groupby('cluster')[customers_col].sum().sort_index()
    else:
        # number of rows per cluster = customers
        cluster_customers = df.groupby('cluster').size().sort_index()

    # --- Ensure a lightweight cluster_summary exists early so visualizations (radar, etc.) can use it
    # This prevents UnboundLocalError when later code references cluster_summary before it's rebuilt.
    cluster_summary = pd.DataFrame(index=range(n_clusters))
    cluster_summary['size'] = cluster_sizes.reindex(range(n_clusters), fill_value=0)
    if revenue_col is not None:
        cluster_summary['total_revenue'] = cluster_revenue.reindex(range(n_clusters), fill_value=0)
        cluster_summary['avg_revenue'] = avg_revenue.reindex(range(n_clusters), fill_value=0)
    if customers_col is not None:
        cluster_summary['total_customers'] = cluster_customers.reindex(range(n_clusters), fill_value=0)
        cluster_summary['avg_customers'] = df.groupby('cluster')[customers_col].mean().reindex(range(n_clusters), fill_value=0)
    else:
        cluster_summary['total_customers'] = cluster_customers.reindex(range(n_clusters), fill_value=0)
        cluster_summary['avg_customers'] = cluster_summary['total_customers'].astype(float)
    # --- end early cluster_summary

    # Create metrics in columns
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Clusters", n_clusters)
    with col2:
        st.metric("Total Data Points", len(df))
    with col3:
        st.metric("Avg. Points/Cluster", f"{len(df)/n_clusters:.1f}")
    with col4:
        if revenue_col is not None:
            st.metric("Total Revenue", f"${df[revenue_col].sum():,.0f}")
    
    # Cluster details in expandable section
    with st.expander("📈 Detailed Cluster Metrics", expanded=True):
        metrics_df = pd.DataFrame({
            'Cluster': cluster_sizes.index,
            'Points': cluster_sizes.values,
            '% of Total': (cluster_sizes.values / len(df) * 100).round(1)
        })
        
        if revenue_col is not None:
            metrics_df['Total Revenue'] = cluster_revenue.values.round(0).astype(int)
            metrics_df['Avg. Revenue'] = avg_revenue.values.round(0).astype(int)
        if customers_col is not None:
            metrics_df['Total Customers'] = cluster_customers.values
            metrics_df['Revenue per Customer'] = (cluster_revenue / cluster_customers.replace(0, np.nan)).round(0)
            # convert to int when safe (keep NaN if no customers)
            metrics_df['Revenue per Customer'] = metrics_df['Revenue per Customer'].fillna(0).astype(int)
        
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

    # ------------------------- Cluster Map (by country) -------------------------
    # Detect possible country columns
    country_candidates = [c for c in df.columns if re.search(r"country|region|territory|nation|countryname", c, re.I)]
    if country_candidates:
        country_col = st.sidebar.selectbox("Country column for map (auto-detected)", options=country_candidates, index=0)
        map_metric_opts = ["Counts", "Total revenue", "Dominant cluster"]
        map_metric = st.sidebar.selectbox("Country map metric", options=map_metric_opts, index=0)

        st.markdown("<h3>Cluster Map — by country</h3>", unsafe_allow_html=True)

        # Build aggregated country x cluster table
        # aggregate country x cluster table (use case-insensitive revenue column if available)
        agg_ops = {"count": ("cluster", "size")}
        if revenue_col is not None:
            agg_ops["revenue"] = (revenue_col, "sum")
        else:
            # create a zero revenue column for consistent shape
            agg_ops["revenue"] = ("cluster", lambda s: 0)
        agg = df.groupby([country_col, "cluster"]).agg(**agg_ops).reset_index()

        if map_metric == "Counts":
            country_summary = agg.groupby(country_col)["count"].sum().reset_index(name="value")
            title = "Data point count per country"
        elif map_metric == "Total revenue":
            if revenue_col is None:
                st.warning("No 'revenue' column detected — pick a different metric or upload revenue data.")
                country_summary = agg.groupby(country_col)["count"].sum().reset_index(name="value")
                title = "Data point count per country (revenue missing)"
            else:
                country_summary = agg.groupby(country_col)["revenue"].sum().reset_index(name="value")
                title = "Total revenue per country"
        else:  # Dominant cluster
            dominant = agg.loc[agg.groupby(country_col)["count"].idxmax()].copy()
            dominant["dominant_cluster"] = dominant["cluster"].astype(str)
            country_summary = dominant[[country_col, "dominant_cluster"]]
            title = "Dominant cluster by country"

        try:
            if map_metric != "Dominant cluster":
                fig = px.choropleth(
                    country_summary,
                    locations=country_col,
                    locationmode='country names',
                    color='value',
                    hover_name=country_col,
                    title=title,
                    color_continuous_scale=px.colors.sequential.Plasma
                )
            else:
                fig = px.choropleth(
                    country_summary,
                    locations=country_col,
                    locationmode='country names',
                    color='dominant_cluster',
                    hover_name=country_col,
                    title=title,
                    color_discrete_sequence=px.colors.qualitative.Safe
                )
            fig.update_layout(margin={"r":0,"t":30,"l":0,"b":0}, height=450)
            st.plotly_chart(fig, use_container_width=True)
        except Exception:
            st.warning("Choropleth map rendering failed (country names might not match plotly's country list). Falling back to bar chart.")
            # fallback: bar chart
            if map_metric != "Dominant cluster":
                st.bar_chart(country_summary.sort_values('value', ascending=False).set_index(country_col)['value'])
            else:
                st.dataframe(country_summary)

        # Stacked bar: cluster composition per country (top countries only for readability)
        st.markdown("#### Cluster composition by country (top countries)")
        top_countries = agg.groupby(country_col)["count"].sum().nlargest(12).index.tolist()
        comp = agg[agg[country_col].isin(top_countries)].pivot(index=country_col, columns="cluster", values="count").fillna(0)
        if not comp.empty:
            comp = comp.reset_index()
            fig2 = px.bar(comp, x=country_col, y=[c for c in comp.columns if c != country_col],
                          title="Cluster counts (stacked) for top countries",
                          labels={"value":"Count", country_col: "Country"})
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("No cluster-country composition to show.")
    else:
        st.info("No country-like column found — add a Country / CountryRegion field to enable country map.")
    # ------------------------- end Cluster Map -------------------------

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
                    title=f"{cat} — Overall distribution",
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
                    title=f"{cat} — Count by cluster",
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
        # Radar chart — prefer these five metrics when available:
        # total_customers, total_revenue, product quantity, lifetime value, loyalty status (scored)
        product_qty_col = find_column_case_insensitive(df, "productquantity") or find_column_case_insensitive(df, "product_quantity")
        ltv_col = find_column_case_insensitive(df, "lifetimevalue") or find_column_case_insensitive(df, "life_time_value") or find_column_case_insensitive(df, "lifeTimeValue")
        loyalty_col = find_column_case_insensitive(df, "loyaltystatus") or find_column_case_insensitive(df, "loyalty_status") or find_column_case_insensitive(df, "loyalty")

        radar_df = pd.DataFrame(index=range(n_clusters))
        # total customers (prefer explicit, fallback to cluster size)
        if 'total_customers' in cluster_summary.columns:
            radar_df['total_customers'] = cluster_summary['total_customers'].astype(float)
        else:
            radar_df['total_customers'] = cluster_summary['size'].astype(float)

        # total revenue (0 if not available)
        radar_df['total_revenue'] = cluster_summary['total_revenue'].astype(float) if 'total_revenue' in cluster_summary.columns else 0.0

        # product quantity (mean per cluster when available)
        if product_qty_col and product_qty_col in df.columns:
            radar_df['avg_product_qty'] = df.groupby('cluster')[product_qty_col].mean().reindex(range(n_clusters), fill_value=0.0).astype(float)

        # lifetime value (mean per cluster)
        if ltv_col and ltv_col in df.columns:
            radar_df['avg_ltv'] = pd.to_numeric(df[ltv_col], errors='coerce').groupby(df['cluster']).mean().reindex(range(n_clusters), fill_value=0.0).astype(float)

        # loyalty status -> numeric score (try to rank categories by revenue if revenue exists, else by counts)
        if loyalty_col and loyalty_col in df.columns:
            # compute category ranking
            if revenue_col is not None:
                metric_by_cat = df.groupby(loyalty_col)[revenue_col].sum().sort_values(ascending=False)
            else:
                metric_by_cat = df.groupby(loyalty_col).size().sort_values(ascending=False)
            cats = metric_by_cat.index.tolist()
            if len(cats) > 0:
                # map top category -> 1.0, bottom -> 0.0 (linear)
                ranks = pd.Series(np.linspace(1.0, 0.0, num=len(cats)), index=cats)
                df['_loyalty_score_tmp'] = df[loyalty_col].astype(str).map(ranks).fillna(0.0).astype(float)
                radar_df['loyalty_score'] = df.groupby('cluster')['_loyalty_score_tmp'].mean().reindex(range(n_clusters), fill_value=0.0).astype(float)
                df.drop(columns=['_loyalty_score_tmp'], inplace=True, errors='ignore')

        # choose at least three radar axes — prefer the requested five when present
        radar_features = [c for c in ['total_customers', 'total_revenue', 'avg_product_qty', 'avg_ltv', 'loyalty_score'] if c in radar_df.columns]

        # fallback to previously computed cluster_stats if fewer than 3 requested metrics available
        if len(radar_features) < 3:
            fallback = [c for c in (cluster_stats.columns.tolist()) if np.issubdtype(cluster_stats[c].dtype, np.number)]
            radar_features = (radar_features + fallback)[:max(0, min(len(fallback), 6))]
        
        if len(radar_features) >= 3:
            with st.expander(" Radar Chart Comparison", expanded=True):
                fig = go.Figure()
                max_vals = radar_df[radar_features].max().replace(0, 1).values.astype(float)
                tick_vals = [0, 0.25, 0.5, 0.75, 1.0]
                tick_text = ['0', '25%', '50%', '75%', '100%']
                display_labels = [pretty_label(c) for c in radar_features]
                for cluster in radar_df.index:
                    raw_vals = radar_df.loc[cluster, radar_features].values.astype(float)
                    normalized = raw_vals / max_vals
                    hovertext = [f"{feat}: {val:.2f} (max {mv:.2f})" for feat, val, mv in zip(radar_features, raw_vals, max_vals)]
                    fig.add_trace(go.Scatterpolar(
                        r=normalized,
                        theta=display_labels,
                        fill='toself',
                        name=f'Cluster {cluster}',
                        hoverinfo='text',
                        hovertext=hovertext
                    ))
                fig.update_layout(
                    polar=dict(radialaxis=dict(visible=True, range=[0, 1], tickvals=tick_vals, ticktext=tick_text)),
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
    
    # Insights and Decision Support / Recommendations
    st.markdown("<h2 class='section-header'> Insights & Decision Support</h2>", unsafe_allow_html=True)

    # Prepare base stats
    cluster_summary = pd.DataFrame(index=range(n_clusters))
    cluster_summary['size'] = df.groupby('cluster').size().reindex(range(n_clusters), fill_value=0)
    if revenue_col is not None:
        cluster_summary['avg_revenue'] = df.groupby('cluster')[revenue_col].mean().reindex(range(n_clusters), fill_value=0)
        cluster_summary['total_revenue'] = df.groupby('cluster')[revenue_col].sum().reindex(range(n_clusters), fill_value=0)
    # customers: prefer explicit 'customers' column; otherwise use row-count as number of customers
    if customers_col is not None:
        cluster_summary['avg_customers'] = df.groupby('cluster')['customers'].mean().reindex(range(n_clusters), fill_value=0)
        cluster_summary['total_customers'] = df.groupby('cluster')['customers'].sum().reindex(range(n_clusters), fill_value=0)
    else:
        cluster_summary['total_customers'] = df.groupby('cluster').size().reindex(range(n_clusters), fill_value=0)
        # average per-cluster customers = total_customers (no per-customer breakdown available)
        cluster_summary['avg_customers'] = cluster_summary['total_customers'].astype(float)

    # compute revenue_per_customer from totals when totals exist (safe divide)
    if 'total_revenue' in cluster_summary.columns and 'total_customers' in cluster_summary.columns:
        cluster_summary['revenue_per_customer'] = cluster_summary.apply(
            lambda r: (r['total_revenue'] / r['total_customers']),
            axis=1
        )
    # Simple priority score: combine normalized size and revenue (if available)
    # Safe normalization for size (avoid division by zero)
    size_max = cluster_summary['size'].max() if (cluster_summary['size'].max() and not np.isnan(cluster_summary['size'].max())) else 1
    size_norm = cluster_summary['size'] / size_max
    
    # Compute revenue normalization only when total_revenue exists and has valid numbers
    if 'total_revenue' in cluster_summary.columns and cluster_summary['total_revenue'].notna().any():
        rev_max = cluster_summary['total_revenue'].max() if (cluster_summary['total_revenue'].max() and not np.isnan(cluster_summary['total_revenue'].max())) else 1
        rev_norm = cluster_summary['total_revenue'] / rev_max
        # combine size and revenue equally when revenue is present
        cluster_summary['priority_score'] = 0.5 * size_norm + 0.5 * rev_norm
    else:
        # fallback: only size determines priority
        cluster_summary['priority_score'] = size_norm

    # Identify top numeric deviations per cluster vs dataset mean
    numeric_feats = [c for c in selected_features if np.issubdtype(df[c].dtype, np.number)]
    global_means = df[numeric_feats].mean() if numeric_feats else pd.Series(dtype=float)

    top_diffs = {}
    for i in range(n_clusters):
        diffs = {}
        for feat in numeric_feats:
            center_val = cluster_centers_orig.loc[i, feat] if feat in cluster_centers_orig.columns else np.nan
            diffs[feat] = (center_val - global_means.get(feat, 0))
        # pick top absolute deviations
        top = sorted(diffs.items(), key=lambda x: abs(x[1]) if not pd.isna(x[1]) else -1, reverse=True)[:4]
        top_diffs[i] = top

    # For categorical features, show the chosen representative value per cluster
    cat_feats = [c for c in selected_features if c not in numeric_feats]
    cat_representatives = cluster_centers_orig[cat_feats].copy() if cat_feats else pd.DataFrame()

    # Display summary table and charts
    col_a, col_b = st.columns([2, 1])
    with col_a:
        # st.markdown("### Cluster summary")
        # display_table = cluster_summary.copy()
        # # round numeric columns for readability
        # for c in display_table.select_dtypes(include=[np.number]).columns:
        #     display_table[c] = display_table[c].round(3)
        # st.dataframe(display_table, use_container_width=True)

        # Priority chart
        fig_prior = px.bar(
            cluster_summary.reset_index().rename(columns={'index': 'cluster'}),
            x='cluster',
            y='priority_score',
            title='Cluster priority score (higher = recommend investment)',
            labels={'priority_score': 'Priority'}
        )
        st.plotly_chart(fig_prior, use_container_width=True)

    with col_b:
        # Top clusters by revenue per customer (if available)
        if 'revenue_per_customer' in cluster_summary.columns:
            st.markdown("### Revenue / Customer")
            fig_rpc = px.bar(
                cluster_summary.reset_index().rename(columns={'index': 'cluster'}),
                x='cluster',
                y='revenue_per_customer',
                title='Avg revenue per customer by cluster',
                labels={'revenue_per_customer': 'Revenue / Customer'}
            )
            st.plotly_chart(fig_rpc, use_container_width=True)
        else:
            st.info("Revenue/customers data not present — KPIs using these fields will be absent.")

    # Produce plain-language recommendations per cluster
    st.markdown("### Recommendations per cluster")
    for i in range(n_clusters):
        score = float(cluster_summary.loc[i, 'priority_score'])
        size = int(cluster_summary.loc[i, 'size'])
        recs = []
        # base recommendations
        if score >= 0.7:
            recs.append("High priority for investment: consider expanding offers, targeted premium campaigns, and direct sales outreach.")
        elif score >= 0.4:
            recs.append("Medium priority: target with marketing campaigns and retention measures, test new offers.")
        else:
            recs.append("Low priority: monitor and keep supporting; focus on cost-effective engagement or automation.")
        # revenue/customer suggestions
        if 'revenue_per_customer' in cluster_summary.columns and not pd.isna(cluster_summary.loc[i, 'revenue_per_customer']):
            rpc = cluster_summary.loc[i, 'revenue_per_customer']
            if rpc >= cluster_summary['revenue_per_customer'].quantile(0.75):
                recs.append("This segment has high revenue per customer — prioritize retention and upsell.")
            elif rpc <= cluster_summary['revenue_per_customer'].quantile(0.25):
                recs.append("Low revenue per customer — consider product re-bundling, pricing, or cross-sell.")
        # behavior-based suggestions using numeric top deviations
        deviations_text = ", ".join([f"{f} ({(d):+.2f})" for f, d in top_diffs.get(i, []) if f is not None])
        if deviations_text:
            recs.append(f"Distinct drivers: {deviations_text}")

        # categorical highlights
        # build a nicer visual for top category indicators: badges + small deviation chart
        cat_badge_html = ""
        if not cat_representatives.empty:
            vals = cat_representatives.loc[i].dropna().to_dict()
            if vals:
                badges = []
                # compute percentage presence of the chosen category in this cluster
                for feat, val in vals.items():
                    try:
                        pct = df.loc[df['cluster'] == i, feat].astype(str).value_counts(normalize=True).get(str(val), 0.0) * 100.0
                    except Exception:
                        pct = 0.0
                    badges.append((feat, val, pct))

                # create compact badge HTML
                badge_html = ""
                for feat, val, pct in badges:
                    badge_html += (
                        f"<span style='display:inline-block;margin:4px 6px;padding:6px 10px;"
                        f"border-radius:14px;background:#f4f9f4;color:#0b3d25;border:1px solid rgba(0,0,0,0.06);"
                        f"font-size:0.9rem;'>"
                        f"<b style=\"margin-right:6px\">{feat}</b>"
                        f"<span style='opacity:0.95'>{val}</span>"
                        f"<small style='margin-left:8px;opacity:0.65'>({pct:.0f}%)</small>"
                        f"</span>"
                    )
                cat_badge_html = f"<div style='margin-top:8px'>{badge_html}</div>"

                # add a short textual marker so the same info is included in the recommendations list
                recs.append("Category indicators")
            else:
                cat_badge_html = ""
        else:
            cat_badge_html = ""

        # render card
        with st.container():
            st.markdown(f"<div class='recommendation-card'><b>Cluster {i}</b> — size {size}, priority {score:.2f}</div>", unsafe_allow_html=True)
            for r in recs:
                st.write("- " + r)
            # render badges for top categories (HTML) if present
            if cat_badge_html:
                st.markdown(cat_badge_html, unsafe_allow_html=True)

            # show small horizontal bar chart for top numeric deviations (if any)
            diffs = top_diffs.get(i, [])
            # if diffs:
            #     # build a small DataFrame for plotting
            #     import pandas as _pd  # local import to avoid top-of-file changes
            #     df_d = _pd.DataFrame(diffs, columns=['feature', 'diff']).dropna()
            #     if not df_d.empty:
            #         df_d['color'] = df_d['diff'].apply(lambda x: 'pos' if x >= 0 else 'neg')
            #         # Keep bars short and unobtrusive
            #         fig_dev = px.bar(
            #             df_d.sort_values('diff'),
            #             x='diff',
            #             y='feature',
            #             orientation='h',
            #             color='color',
            #             color_discrete_map={'pos': '#2e7d32', 'neg': '#d32f2f'},
            #             title="Top numeric deviations vs global mean",
            #             labels={'diff': 'Deviation'}
            #         )
            #         fig_dev.update_layout(showlegend=False, height=220, margin=dict(l=60, r=10, t=30, b=10))
            #         st.plotly_chart(fig_dev, use_container_width=True)
    # High-level platform recommendations
    st.markdown("### Actionable platform recommendations")
    actions = []
    # emphasize segments for effort
    top_clusters = cluster_summary['priority_score'].sort_values(ascending=False).head(3).index.tolist()
    actions.append(f"Focus acquisition and sales on clusters: {', '.join(map(str, top_clusters))}")
    if revenue_col is not None:
        best_rev_cluster = cluster_summary['total_revenue'].idxmax()
        actions.append(f"Top revenue cluster: {best_rev_cluster} — design premium/retention programs.")
    if cat_feats:
        actions.append("Create targeted messaging for clusters using the dominant categorical features shown above.")
    actions.append("A/B test 2–3 targeted offers for high-priority segments and measure conversion/CLTV.")
    for a in actions:
        st.markdown(f"- {a}")
        
    ft = """
    <style>
    a:link , a:visited{
    color: #BFBFBF;  /* theme's text color hex code at 75 percent brightness*/
    background-color: transparent;
    text-decoration: none;
    }

    a:hover,  a:active {
    color: #0283C3; /* theme's primary color*/
    background-color: transparent;
    text-decoration: underline;
    }

    #page-container {
    position: relative;
    min-height: 10vh;
    }

    footer{
        visibility:hidden;
    }

    .footer {
    position: relative;
    left: 0;
    top:230px;
    bottom: 0;
    width: 100%;
    background-color: transparent;
    color: #808080; /* theme's text color hex code at 50 percent brightness*/
    text-align: left; /* you can replace 'left' with 'center' or 'right' if you want*/
    }
    </style>

    <div id="page-container">

    <div class="footer">
    <p style='font-size: 0.875em;'>Made by <img src="https://em-content.zobj.net/source/skype/289/red-heart_2764-fe0f.png" alt="heart" height= "10"/><a style='display: inline; text-align: left;' href="https://github.com/ngmikhoi" target="_blank"> ngmikhoi</a></p>
    </div>

    </div>
    """
    st.write(ft, unsafe_allow_html=True)

    # # ------------------------- Actionable Platform Recommendations -------------------------
    # st.markdown("<h2 class='section-header'>🎯 Actionable Platform Recommendations</h2>", unsafe_allow_html=True)

    # # Build a robust allocation and tactical suggestion per-cluster using cluster_summary KPIs
    # cs = cluster_summary.copy()
    # # Ensure priority_score exists
    # if 'priority_score' not in cs.columns:
    #     cs['priority_score'] = cs['size'] / cs['size'].max() if cs['size'].max() else 1.0

    # # Normalize priority for allocation
    # total_priority = cs['priority_score'].sum() if cs['priority_score'].sum() > 0 else 1.0
    # cs['priority_weight'] = cs['priority_score'] / total_priority

    # # Translate weights to recommended platform/marketing budget % (scale to e.g. 70% of flexible budget)
    # FLEXIBLE_SHARE = 0.7  # 70% of flexible spend may be allocated by priority; remaining 30% for experiments/maintenance
    # cs['recommended_pct_of_flexible_budget'] = (cs['priority_weight'] * 100 * FLEXIBLE_SHARE).round(1)
    # cs['recommended_pct_of_total_budget'] = cs['recommended_pct_of_flexible_budget'] * 0.8  # example: flexible = 80% of total (tunable)

    # # Classify cluster strategy based on size and revenue_per_customer
    # def classify_cluster(row):
    #     size = row.get('size', 0)
    #     rpc = row.get('revenue_per_customer', np.nan)
    #     # rules:
    #     if not np.isnan(rpc):
    #         if rpc >= cs['revenue_per_customer'].quantile(0.75):
    #             if size / max(1, cs['size'].max()) < 0.25:
    #                 return "Niche VIP — Retain & Upsell"
    #             else:
    #                 return "High-Value — Scale & Retain"
    #         elif rpc <= cs['revenue_per_customer'].quantile(0.25):
    #             if size / max(1, cs['size'].max()) > 0.4:
    #                 return "Mass Low-Value — Efficiency & Acquisition"
    #             else:
    #                 return "Underperforming — Investigate & Test"
    #     # fallback on size only
    #     if size >= cs['size'].quantile(0.75):
    #         return "Large — Growth & Monetize"
    #     if size <= cs['size'].quantile(0.25):
    #         return "Small — Test/Research"
    #     return "Balanced"

    # cs['strategy_bucket'] = cs.apply(classify_cluster, axis=1)

    # # Suggest tactical playbook for each strategy bucket
    # bucket_playbook = {
    #     "Niche VIP — Retain & Upsell": [
    #         "High-touch retention programs, personalized offers, loyalty incentives",
    #         "Focus on Customer Success, NPS, and upsell automation",
    #         "KPI: CLTV, retention rate, upsell conversion"
    #     ],
    #     "High-Value — Scale & Retain": [
    #         "Increase acquisition spend selectively, keep retention offers",
    #         "Run lookalike campaigns and premium bundles",
    #         "KPI: CAC payback, ARPU, retention rate"
    #     ],
    #     "Mass Low-Value — Efficiency & Acquisition": [
    #         "Optimize cost-per-acquisition, emphasize automation and self-serve",
    #         "Use promotions to increase average order value, cross-sell at checkout",
    #         "KPI: CAC, AOV, profit margin"
    #     ],
    #     "Underperforming — Investigate & Test": [
    #         "Run small experiments to test product/price/message; collect qualitative feedback",
    #         "Close monitoring and diagnostic analytics",
    #         "KPI: lift in conversion, test success rate"
    #     ],
    #     "Large — Growth & Monetize": [
    #         "Invest for scale, run regional expansion tests and channel diversification",
    #         "Increase inventory and logistics readiness",
    #         "KPI: incremental revenue, growth rate"
    #     ],
    #     "Small — Test/Research": [
    #         "Use low-cost experiments and exploratory offers; gather data",
    #         "Consider targeted surveys and observational research",
    #         "KPI: experiment conversion, hypothesis validation"
    #     ],
    #     "Balanced": [
    #         "Balanced mix of acquisition and retention; monitor KPIs and operate A/B tests",
    #         "KPI: conversion rate, retention"
    #     ]
    # }

    # # Produce summary table and a visual allocation chart
    # cs_display = cs.reset_index().rename(columns={'index': 'cluster'})
    # cs_display = cs_display[['cluster', 'size', 'total_revenue', 'avg_revenue', 'priority_score', 'recommended_pct_of_flexible_budget', 'strategy_bucket']].fillna('N/A')

    # st.markdown("#### Recommended budget allocation (lens: priority score)")
    # st.dataframe(cs_display.style.format({
    #     'priority_score': '{:.3f}',
    #     'recommended_pct_of_flexible_budget': '{:.1f}%'
    # }), use_container_width=True)

    # # Allocation chart
    # fig_alloc = px.bar(
    #     cs_base,  # use numeric-safe base for plotting
    #     x='cluster',
    #     y='recommended_pct_of_flexible_budget',
    #     color='strategy_bucket',
    #     title='Recommended % of Flexible Budget by Cluster',
    #     labels={'recommended_pct_of_flexible_budget': '% of flexible budget'}
    # )
    # st.plotly_chart(fig_alloc, use_container_width=True)

    # # Render actionable recommendations per cluster using the playbook
    # st.markdown("### Cluster-level tactical recommendations (concise)")
    # for idx, row in cs.iterrows():
    #     bucket = row.get('strategy_bucket', 'Balanced')
    #     plays = bucket_playbook.get(bucket, bucket_playbook['Balanced'])
    #     pct = row.get('recommended_pct_of_flexible_budget', 0.0)
    #     st.markdown(f"**Cluster {idx} — {bucket}** (recommended flexible budget ≈ {pct:.1f}%)")
    #     for p in plays:
    #         st.write("- " + p)

    # # Recommended experiments and KPIs to run across clusters
    # st.markdown("### Recommended experiments & KPIs")
    # st.markdown("- Set up 2–3 priority experiments per top-priority cluster (A/B tests on price, bundling, message).")
    # st.markdown("- Track KPIs by cluster: conversion rate, revenue per customer, retention rate, CAC, and CLTV.")
    # st.markdown("- Re-evaluate cluster priorities quarterly and re-allocate flexible budget based on observed lift.")

    # # optional: small suggestion on next steps
    # st.info("Next steps: pick 1–2 clusters to test high-priority tactics this quarter. Monitor KPIs and iterate.")
    # --------------------------------------------------------------------------------
    # # ------------------------- Generate report (zip) -------------------------
    # st.markdown("### Export / Generate report")
    # st.write("Create a downloadable ZIP with cluster data, Excel workbook, plain-text summary, and a small chart.")
    # 
    # if st.button("Generate report"):
    #     try:
    #         # Build summary text
    #         ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    #         lines = []
    #         lines.append(f"Market clustering report — generated {datetime.now().isoformat()}")
    #         lines.append("")
    #         lines.append("Cluster summary:")
    #         lines.append(cluster_summary.to_string())
    #         lines.append("")
    #         lines.append("Top recommendations:")
    #         for i in range(n_clusters):
    #             score = float(cluster_summary.loc[i, 'priority_score'])
    #             size = int(cluster_summary.loc[i, 'size'])
    #             lines.append(f"- Cluster {i}: size={size}, priority={score:.3f}")
    #         lines.append("")
    #         lines.append("Action items:")
    #         lines.extend(actions)
    #         summary_text = "\n".join(lines)
    # 
    #         # Build Excel workbook in-memory
    #         excel_buf = BytesIO()
    #         with pd.ExcelWriter(excel_buf, engine='xlsxwriter') as writer:
    #             try:
    #                 df.to_excel(writer, sheet_name='clustered_data', index=False)
    #             except Exception:
    #                 # fallback to CSV write behavior — still try
    #                 pass
    #             if 'centers_orig' in locals() and centers_orig is not None:
    #                 centers_orig.to_excel(writer, sheet_name='centers_orig', index=False)
    #             if 'cluster_summary' in locals():
    #                 cluster_summary.to_excel(writer, sheet_name='cluster_summary')
    #             if 'cluster_centers_orig' in locals():
    #                 cluster_centers_orig.to_excel(writer, sheet_name='cluster_centers_orig')
    #             writer.save()
    #         excel_bytes = excel_buf.getvalue()
    # 
    #         # CSV snapshot
    #         csv_bytes = df.to_csv(index=False).encode('utf-8')
    # 
    #         # Small priority chart (matplotlib) into PNG
    #         png_buf = BytesIO()
    #         try:
    #             fig, ax = plt.subplots(figsize=(6, 3))
    #             cluster_summary['priority_score'].plot(kind='bar', ax=ax, color='C0')
    #             ax.set_title('Cluster priority score')
    #             ax.set_xlabel('Cluster')
    #             ax.set_ylabel('Priority')
    #             fig.tight_layout()
    #             fig.savefig(png_buf, format='png', dpi=120)
    #             plt.close(fig)
    #             png_bytes = png_buf.getvalue()
    #         except Exception:
    #             png_bytes = b""
    # 
    #         # Create ZIP with files
    #         zip_buf = BytesIO()
    #         with zipfile.ZipFile(zip_buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
    #             zf.writestr(f"clustered_data_{ts}.csv", csv_bytes)
    #             zf.writestr(f"clustered_data_{ts}.xlsx", excel_bytes)
    #             zf.writestr(f"summary_{ts}.txt", summary_text.encode('utf-8'))
    #             if png_bytes:
    #                 zf.writestr(f"priority_{ts}.png", png_bytes)
    # 
    #         zip_buf.seek(0)
    # 
    #         st.success("Report generated — click below to download")
    #         st.download_button(
    #             "Download report (ZIP)",
    #             data=zip_buf.getvalue(),
    #             file_name=f"market_clustering_report_{ts}.zip",
    #             mime="application/zip"
    #         )
    #     except Exception as e:
    #         st.error(f"Failed to generate report: {e}")
    # -----------------------------------------------------------------------
    # # Decision Support Section
    # st.markdown("<h2 class='section-header'>🎯 Decision Support & Recommendations</h2>", unsafe_allow_html=True)
    
    # # Only show recommendations if we have the required data
    # if 'revenue' in df.columns and 'customers' in df.columns:
    #     # Find high potential clusters
    #     cluster_potential = (df.groupby('cluster')
    #                        .agg({
    #                            'revenue': 'sum',
    #                            'customers': 'sum',
    #                            'latitude': 'mean',
    #                            'longitude': 'mean'
    #                        })
    #                        .sort_values('revenue', ascending=False))
        
    #     # High potential clusters (top 20% by revenue)
    #     high_potential = cluster_potential[cluster_potential['revenue'] > 
    #                                      cluster_potential['revenue'].quantile(0.8)]
        
    #     # Underperforming clusters (bottom 20% by revenue)
    #     low_potential = cluster_potential[cluster_potential['revenue'] < 
    #                                     cluster_potential['revenue'].quantile(0.2)]
        
    #     col1, col2 = st.columns(2)
        
    #     with col1:
    #         st.markdown("<h3 style='color: #2e7d32;'>🚀 High Potential Clusters</h3>", unsafe_allow_html=True)
    #         if not high_potential.empty:
    #             for idx, row in high_potential.iterrows():
    #                 with st.container():
    #                     st.markdown(f"""
    #                     <div class='recommendation-card'>
    #                         <h4>Cluster {idx}</h4>
    #                         <p>Revenue: ${row['revenue']:,.0f}</p>
    #                         <p>Customers: {row['customers']:,.0f}</p>
    #                         <p><strong>Recommendation:</strong> Consider increasing marketing budget and inventory in this area.</p>
    #                     </div>
    #                     """, unsafe_allow_html=True)
    #         else:
    #             st.info("No high potential clusters identified in the top 20%.")
        
    #     with col2:
    #         st.markdown("<h3 style='color: #d32f2f;'>⚠️ Underperforming Clusters</h3>", unsafe_allow_html=True)
    #         if not low_potential.empty:
    #             for idx, row in low_potential.iterrows():
    #                 with st.container():
    #                     st.markdown(f"""
    #                     <div class='warning-card'>
    #                         <h4>Cluster {idx}</h4>
    #                         <p>Revenue: ${row['revenue']:,.0f}</p>
    #                         <p>Customers: {row['customers']:,.0f}</p>
    #                         <p><strong>Recommendation:</strong> Investigate reasons for low performance. Consider promotions or market research.</p>
    #                     </div>
    #                     """, unsafe_allow_html=True)
    #         else:
    #             st.info("No underperforming clusters identified in the bottom 20%.")
        
    #     # Market expansion opportunities
    #     st.markdown("<h3 style='color: #1565c0;'>🌐 Market Expansion Analysis</h3>", unsafe_allow_html=True)
    #     st.markdown("""
    #     <div class='metric-card'>
    #         <h4>Geographic Coverage</h4>
    #         <p>Your market coverage spans across {0} distinct regions. 
    #         Based on the distribution of your high-performing clusters, 
    #         we recommend exploring expansion in areas with similar demographic 
    #         and economic characteristics.</p>
    #     </div>
    #     """.format(n_clusters), unsafe_allow_html=True)
        
    #     # Resource allocation strategy
    #     st.markdown("<h3 style='color: #7b1fa2;'>📊 Resource Allocation Strategy</h3>", unsafe_allow_html=True)
    #     st.markdown("""
    #     <div class='metric-card'>
    #         <h4>Optimal Resource Distribution</h4>
    #         <p>Based on cluster analysis, consider the following allocation strategy:</p>
    #         <ul>
    #             <li><strong>High Potential Clusters:</strong> Allocate 50% of marketing budget</li>
    #             <li><strong>Medium Potential Clusters:</strong> Allocate 35% of marketing budget</li>
    #             <li><strong>Low Potential Clusters:</strong> Allocate 15% for maintenance and research</li>
    #         </ul>
    #     </div>
    #     """, unsafe_allow_html=True)
        
    #     # Actionable insights
    #     st.markdown("<h3 style='color: #ff8f00;'>💡 Actionable Insights</h3>", unsafe_allow_html=True)
    #     insights = [
    #         "Geographic clusters with high customer density show 30% higher revenue potential.",
    #         "Consider implementing targeted promotions in underperforming clusters to boost sales.",
    #         "The top 20% of clusters generate 60% of total revenue - focus on these high-value areas.",
    #         "Seasonal trends show increased demand in urban clusters during Q4."
    #     ]
        
    #     for insight in insights:
    #         st.markdown(f"- {insight}")
        
    #     # Generate report button
    #     if st.button("📄 Generate Detailed Market Analysis Report", 
    #                 help="Click to generate a comprehensive PDF report with all analysis and recommendations"):
    #         with st.spinner('Generating report...'):
    #             # Simulate report generation
    #             import time
    #             time.sleep(2)
                
    #             # Create a simple report
    #             current_date = datetime.now().strftime("%Y-%m-%d")
    #             report = f"""
    #             # Market Analysis Report
    #             **Date:** {current_date}
    #             
    #             ## Executive Summary
    #             This report provides an analysis of market segments and recommendations 
    #             for strategic decision making.
    #             
    #             ## Key Findings
    #             - Total clusters analyzed: {n_clusters}
    #             - Total revenue across all clusters: ${df['revenue'].sum():,.0f}
    #             - Average revenue per cluster: ${df['revenue'].mean():,.0f}
    #             
    #             ## Recommendations
    #             1. Focus marketing efforts on high-potential clusters
    #             2. Investigate underperforming clusters for improvement opportunities
    #             3. Consider geographic expansion in areas with similar characteristics to top-performing clusters
    #             """.format(date=current_date, 
    #                       n_clusters=n_clusters)
                
    #             # Display report
    #             st.download_button(
    #                 label="📥 Download Report as PDF",
    #                 data=report,
    #                 file_name=f"market_analysis_report_{current_date}.md",
    #                 mime="text/markdown"
    #             )
    #             st.success("Report generated successfully!")
    # else:
    #     st.warning("⚠️ Additional data (revenue, customers) would enable more detailed recommendations.")
    #     st.info("For optimal results, ensure your dataset includes 'revenue' and 'customers' columns.")
    
    # # Download results
    # st.sidebar.subheader("Download Results")
    
    # # Convert DataFrame to CSV for download
    # csv = df.to_csv(index=False).encode('utf-8')
    # st.sidebar.download_button(
    #     label="Download Clustered Data",
    #     data=csv,
    #     file_name="clustered_market_data.csv",
    #     mime="text/csv"
    # )

    # st.write("Columns:", df.columns.tolist())
    # st.dataframe(df.dtypes)

    # # auto-detect revenue-like columns
    # revenue_candidates = [c for c in df.columns if re.search(r"revenue|sales|amount|total|price", c, re.I)]
    # customers_candidates = [c for c in df.columns if re.search(r"customers?|customer_count|num_customers|qty|quantity", c, re.I)]

    # revenue_col = st.sidebar.selectbox("Revenue column (if any)", options=[None] + revenue_candidates, index=0)
    # customers_col = st.sidebar.selectbox("Customers column (if any)", options=[None] + customers_candidates, index=0)

    # # coerce to numeric if chosen
    # if revenue_col:
    #     df[revenue_col] = pd.to_numeric(df[revenue_col], errors='coerce')
    # # Use row-count as customers when not provided
    # if customers_col:
    #     df[customers_col] = pd.to_numeric(df[customers_col], errors='coerce')
    # else:
    #     # create a helper column for customers = 1 per row
    #     df['_customer_row_count'] = 1
    #     customers_col = '_customer_row_count'

if __name__ == "__main__":
    main()
