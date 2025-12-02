# Data Warehouse and Decision Support System (DWH-DSS)

## Project Overview

This project implements a complete data warehouse and decision support system for analyzing sales data from the AdventureWorks database. The system includes ETL pipelines, dimensional modeling with Slowly Changing Dimensions (SCD), and advanced analytics using a **Hybrid Deep Learning approach** for **Dynamic Monthly Market Segmentation**.

## System Architecture

### Complete Data Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│                        SOURCE SYSTEM                                │
│                   AdventureWorks SQL Server                         │
│  (Sales, Production, Person schemas)                                │
└────────────────────────┬────────────────────────────────────────────┘
                         │
                         │ Airbyte (ELT Pipeline)
                         │ - Extract from SQL Server
                         │ - Load to Snowflake
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      SNOWFLAKE DATA WAREHOUSE                       │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │                    DIMENSION TABLES                          │   │
│  │  - DimProduct (SCD Type 2)                                   │   │
│  │  - DimSpecialOffer (SCD Type 2)                              │   │
│  │  - DimCustomer (SCD Type 2)                                  │   │
│  │  - DimTerritory (SCD Type 0)                                 │   │
│  │  - DimTime (SCD Type 0)                                      │   │
│  │  - BridgeProductSpecialOffer (SCD Type 2)                    │   │
│  └──────────────────────────────────────────────────────────────┘   │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │                      FACT TABLES                             │   │
│  │  - FactSale (Transactional sales data)                       │   │
│  │  - FactMarketSegmentation (ML-generated insights)            │   │
│  └──────────────────────────────────────────────────────────────┘   │
└────────────────────────┬────────────────────────────────────────────┘
                         │
                         │ Read (SELECT queries)
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────────┐
│              ANALYTICS & ML PIPELINE (Python)                       │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │  Dynamic Monthly Segmentation Pipeline                       │   │
│  │  - Extract time-series & static features                     │   │
│  │  - Hybrid Model: LSTM (Time-series) + Static Features        │   │
│  │  - Rolling Window Processing (5-month window)                │   │
│  │  - Generate monthly segment assignments                      │   │
│  └──────────────────────────────────────────────────────────────┘   │
└────────────────────────┬────────────────────────────────────────────┘
                         │
                         │ Write (MERGE statements)
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────────┐
│                   SNOWFLAKE DATA WAREHOUSE                          │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │         FactMarketSegmentation (Updated)                     │   │
│  │  - Monthly segment assignments                           │   │
│  │  - Embeddings and confidence scores                          │   │
│  └──────────────────────────────────────────────────────────────┘   │
└────────────────────────┬────────────────────────────────────────────┘
                         │
                         │ Read for visualization
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    STREAMLIT APPLICATIONS                           │
│  - Market Clustering App (Dashboard + Pipeline Management)          │
│  - Data Warehouse Verification Tool                                 │
└─────────────────────────────────────────────────────────────────────┘
```

### Technology Stack

- **Data Warehouse**: Snowflake
- **ETL Tool**: Airbyte
- **Programming Language**: Python 3.8+
- **Deep Learning Framework**: PyTorch (LSTM Autoencoder)
- **Machine Learning**: scikit-learn (K-Means, PCA, StandardScaler)
- **Visualization**: Streamlit, Plotly
- **Database Connector**: snowflake-connector-python

## File Structure and Roles

```bash
DWH-DSS/
├── .streamlit/
│   └── secrets.toml          # Snowflake credentials
├── data_transform.sql        # ELT transformations & SCD logic
├── dwh_init.sql              # Data Warehouse Schema DDL
├── market_clustering_app.py  # Streamlit Dashboard & UI
├── market_segmentation_pipeline.py # Core ML & Clustering Logic
├── requirements.txt          # Python dependencies
├── .env                      # Environment variables (Gemini API Key)
└── README.md                 # Project documentation
```

### Core Pipeline Files

#### `market_segmentation_pipeline.py`
**Role**: Core machine learning pipeline for market segmentation.

**Responsibilities**:
- **Data Extraction**: Pulls time-series sales data and static market features from Snowflake.
- **Hybrid Feature Engineering**: Combines LSTM-generated temporal embeddings with static features (Product Lines, Subcategories).
- **Dynamic Monthly Segmentation**: Implements a rolling window approach (e.g., 5 months) to segment markets for *each specific month*, allowing for tracking of segment evolution over time.
- **Model Training**: Trains an LSTM Autoencoder to learn temporal patterns from sales history.
- **Clustering**: Applies K-Means clustering on the combined feature set.
- **Persistence**: Saves results to `FactMarketSegmentation` in Snowflake.

**Key Classes**:
- `MarketSegmentationPipeline`: Main pipeline orchestrator.
- `LSTMEncoder`: Neural network for time-series embedding.
- `TimeSeriesDataset`: PyTorch dataset wrapper.

#### `market_clustering_app.py`
**Role**: Unified Streamlit application for both **Dashboarding** and **Pipeline Management**.

**Modes**:
1.  **📊 Official Dashboard**:
    *   **Snapshot View**: Filter segmentation results by specific month (e.g., "2014-06").
    *   **Key Metrics**: Total Segments, Markets, Revenue, Quantity.
    *   **Visualizations**:
        *   Revenue Share (Donut Chart)
        *   Market Count Distribution (Donut Chart)
        *   Avg Revenue per Market (Bar Chart)
        *   Total Quantity Sold (Bar Chart)
    *   **Drill-down**: Detailed market performance table per segment.
    *   **Temporal Trends**: Line charts showing how metrics evolve over time.

2.  **⚙️ Pipeline Management**:
    *   **Configuration**: Set target segments, training epochs, batch size, and device (CPU/GPU).
    *   **Execution**: Trigger the "Dynamic Monthly Segmentation" process.
    *   **Monitoring**: View real-time logs and progress of the segmentation run.
   
3.  **🛝 Playground (Ad-hoc Analysis)**:
    *   **Role**: Interactive sandbox for data scientists to experiment with raw data without affecting the official pipeline.
    *   **Data Source**: Direct connection to `FactSale` and Dimension tables (bypassing the LSTM pipeline).
    *   **Features**:
        *   **Raw Data Loading**: Fetch sample data directly from Snowflake.
        *   **Feature Selection**: Manually select columns (Revenue, Quantity, Discount, etc.) for clustering.
        *   **Interactive Clustering**: Run K-Means on-the-fly with adjustable $K$.
        *   **Visualizations**:
            *   PCA Projection (2D/3D) of raw features.
            *   Correlation Heatmaps.
            *   Elbow Method / Silhouette Analysis.

### Database & ETL Scripts

#### `dwh_init.sql`
**Role**: SQL DDL script for initializing the Snowflake data warehouse schema.
- Creates Dimension tables (Product, Customer, Territory, Time, SpecialOffer).
- Creates Fact tables (FactSale, FactMarketSegmentation).
- Sets up SCD Type 2 tracking columns (`ValidFrom`, `ValidTo`, `IsActive`).

#### `data_transform.sql`
**Role**: SQL script for performing ELT data transformations and loading data into the Data Warehouse.
- **SCD Type 2 Implementation**: Handles versioning for `DimProduct`, `DimSpecialOffer`, and `DimCustomer` to track historical changes.
- **Dimension Loading**: Populates `DimTerritory` and generates `DimTime` data.
- **Fact Table Loading**: Transforms and loads transactional data into `FactSale`, handling surrogate key lookups.
- **Bridge Table Management**: Manages the `BridgeProductSpecialOffer` table for many-to-many relationships.

## Machine Learning Pipeline Details

### Dynamic Monthly Segmentation Methodology

**Objective**: Group markets into segments based on *both* their recent sales trends and static characteristics, updating these assignments monthly.

**Approach**: **Hybrid Model** (LSTM + Static Features) with **Rolling Window**.

**Steps**:

1.  **Data Extraction**:
    *   Extract recent 12 months of sales data.
    *   Extract static features: `TotalProductLines`, `TotalSubcategories`.

2.  **Rolling Window Processing**:
    *   The pipeline iterates through each available month in the dataset.
    *   For each target month $T$, it looks back at a fixed window (e.g., $T-4, ..., T$).
    *   Markets with insufficient history for the window are skipped for that specific month.

3.  **Hybrid Embedding Generation**:
    *   **Temporal Features**: The 5-month sequence of sales metrics (Revenue, Quantity, Growth, etc.) is fed into a trained **LSTM Autoencoder** to produce a 32-dimensional embedding vector.
    *   **Static Features**: Static attributes are normalized and concatenated with the LSTM embedding.
    *   **Combined Vector**: The final feature vector represents both "how the market is performing recently" and "what the market structure is".

4.  **Clustering**:
    *   **K-Means** is applied to the combined vectors to group markets into $K$ segments.
    *   This is done independently for each month (or using a global model applied monthly), allowing markets to move between segments as their performance changes.

5.  **Result Storage**:
    *   Results are saved with a `DateKey` corresponding to the target month.
    *   This enables the "Snapshot" view in the dashboard.

## Setup and Usage

### Prerequisites
1.  **Python 3.8+**
2.  **Snowflake Account** with appropriate permissions.
3.  **`.streamlit/secrets.toml`** configured with Snowflake credentials.
4.  **A Gemini API key** for AI suggestion in the Playground mode.

### Installation
```bash
pip install -r requirements.txt
```

### Running the Application
To launch the unified dashboard and pipeline manager:
```bash
python -m streamlit run market_clustering_app.py
```

### Workflow
1.  **Go to "Pipeline Management"**:
    *   Set "Target Segments" (e.g., 5).
    *   Click "🚀 Run Monthly Segmentation".
    *   Wait for the pipeline to process all available months.
2.  **Go to "Official Dashboard"**:
    *   Select a "Month (Snapshot)" from the sidebar filter.
    *   Analyze the segments for that specific month.
    *   Use the "Temporal Trends" section to see how segments have evolved.

## License

This project is developed for academic purposes as part of the CO4031 Data Warehouse course.