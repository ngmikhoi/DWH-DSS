# Data Warehouse and Decision Support System (DWH-DSS)

## Project Overview

This project implements a complete data warehouse and decision support system for analyzing sales data from the AdventureWorks database. The system includes ETL pipelines, dimensional modeling with Slowly Changing Dimensions (SCD), and advanced analytics using deep learning for market segmentation.

## System Architecture

### Complete Data Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│                        SOURCE SYSTEM                                 │
│                   AdventureWorks SQL Server                          │
│  (Sales, Production, Person schemas)                                │
└────────────────────────┬────────────────────────────────────────────┘
                         │
                         │ Airbyte (ELT Pipeline)
                         │ - Extract from SQL Server
                         │ - Load to Snowflake
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      SNOWFLAKE DATA WAREHOUSE                        │
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
│              ANALYTICS & ML PIPELINE (Python)                        │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │  Market Segmentation Pipeline                                │   │
│  │  - Extract time-series data from FactSale                    │   │
│  │  - LSTM autoencoder for embedding generation                 │   │
│  │  - K-Means/DBSCAN clustering                                 │   │
│  │  - Generate market segments                                  │   │
│  └──────────────────────────────────────────────────────────────┘   │
└────────────────────────┬────────────────────────────────────────────┘
                         │
                         │ Write (MERGE statements)
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────────┐
│                   SNOWFLAKE DATA WAREHOUSE                           │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │         FactMarketSegmentation (Updated)                     │   │
│  │  - Segment assignments per market                            │   │
│  │  - Embeddings and confidence scores                          │   │
│  └──────────────────────────────────────────────────────────────┘   │
└────────────────────────┬────────────────────────────────────────────┘
                         │
                         │ Read for visualization
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    STREAMLIT APPLICATIONS                            │
│  - Market Segmentation Pipeline UI                                  │
│  - Market Clustering Analysis Dashboard                             │
│  - Data Warehouse Verification Tool                                 │
└─────────────────────────────────────────────────────────────────────┘
```

### Technology Stack

- **Data Warehouse**: Snowflake
- **ETL Tool**: Airbyte
- **Programming Language**: Python 3.8+
- **Deep Learning Framework**: PyTorch
- **Machine Learning**: scikit-learn
- **Visualization**: Streamlit, Plotly
- **Database Connector**: snowflake-connector-python

## File Structure and Roles

### Core Pipeline Files

#### `market_segmentation_pipeline.py`
**Role**: Core machine learning pipeline for market segmentation

**Responsibilities**:
- Extract time-series sales data from Snowflake (aggregated by market and month)
- Prepare sequences for LSTM model training
- Train LSTM autoencoder to generate market embeddings
- Perform clustering (K-Means or DBSCAN) on embeddings
- Save segmentation results to FactMarketSegmentation table

**Key Classes**:
- `MarketSegmentationPipeline`: Main pipeline orchestrator
- `LSTMEncoder`: Neural network for time-series embedding
- `TimeSeriesDataset`: PyTorch dataset wrapper

**Usage**: Imported by Streamlit applications, not run directly

---

#### `run_segmentation_pipeline.py`
**Role**: Streamlit web application for running and monitoring the segmentation pipeline

**Responsibilities**:
- Provide interactive UI for pipeline configuration
- Display real-time progress during pipeline execution
- Visualize training metrics and clustering results
- Export segmentation results as CSV
- Show PCA projections of market embeddings

**Features**:
- Configurable model parameters (sequence length, clusters, epochs)
- Training loss visualization
- Segment distribution charts
- Market-segment mapping tables
- Embedding space visualization

**Usage**: Run with `streamlit run run_segmentation_pipeline.py`

---

#### `market_clustering_app.py`
**Role**: Streamlit dashboard for analyzing existing market segmentation results

**Responsibilities**:
- Load and display data from FactMarketSegmentation
- Provide interactive filtering and exploration
- Visualize segment characteristics
- Compare markets within segments
- Analyze temporal trends in segmentation

**Usage**: Run with `streamlit run market_clustering_app.py`

---

### Database Schema Files

#### `dwh_init.sql`
**Role**: SQL DDL script for initializing the Snowflake data warehouse schema

**Responsibilities**:
- Create all dimension tables with appropriate SCD types
- Create fact tables (FactSale, FactMarketSegmentation)
- Define primary keys and constraints
- Set up bridge table for many-to-many relationships

**Tables Created**:
- `DimProduct`: Product dimension with SCD Type 2
- `DimSpecialOffer`: Special offer dimension with SCD Type 2
- `DimCustomer`: Customer dimension with SCD Type 2
- `DimTerritory`: Territory dimension with SCD Type 0
- `DimTime`: Time dimension with SCD Type 0
- `BridgeProductSpecialOffer`: Bridge table with SCD Type 2
- `FactSale`: Sales fact table
- `FactMarketSegmentation`: Market segmentation results fact table

**Usage**: Execute in Snowflake before running ETL pipelines

---

#### `schema.txt`
**Role**: Human-readable documentation of the data warehouse schema

**Content**:
- Detailed column descriptions
- Foreign key relationships
- SCD type justifications
- Business logic for derived columns
- Mapping from source (AdventureWorks) to target (DWH) tables

**Usage**: Reference documentation for developers and analysts

---

### Verification and Testing

#### `verify_dwh_data.py`
**Role**: Streamlit application for validating data warehouse integrity

**Responsibilities**:
- Verify SCD Type 2 implementation correctness
- Check referential integrity across tables
- Validate data quality and completeness
- Display data distribution statistics
- Identify and report data anomalies

**Validation Checks**:
- Active record flags (IsActive)
- ValidFrom/ValidTo date ranges
- Orphaned records in fact tables
- Duplicate surrogate keys
- NULL value analysis

**Usage**: Run with `streamlit run verify_dwh_data.py`

---

### Configuration Files

#### `requirements.txt`
**Role**: Python package dependencies

**Key Dependencies**:
- `streamlit`: Web application framework
- `pandas`: Data manipulation
- `numpy`: Numerical computing
- `torch`: Deep learning framework
- `scikit-learn`: Machine learning algorithms
- `plotly`: Interactive visualizations
- `snowflake-connector-python`: Snowflake database connector

**Usage**: Install with `pip install -r requirements.txt`

---

#### `.streamlit/secrets.toml`
**Role**: Secure storage for Snowflake credentials (not in repository)

**Required Configuration**:
```toml
snowflake_user = "YOUR_USERNAME"
snowflake_password = "YOUR_PASSWORD"
snowflake_account = "YOUR_ACCOUNT"
snowflake_warehouse = "YOUR_WAREHOUSE"
snowflake_database = "YOUR_DATABASE"
snowflake_schema = "DWH"
```

**Usage**: Create this file before running any Streamlit applications

---

## Pipeline Execution Order

### Phase 1: Initial Setup

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure Snowflake Connection**
   - Create `.streamlit/secrets.toml` with Snowflake credentials
   - Ensure user has CREATE, INSERT, UPDATE, SELECT permissions

3. **Initialize Data Warehouse Schema**
   - Execute `dwh_init.sql` in Snowflake
   - Verify all tables are created successfully

### Phase 2: Data Ingestion (ETL)

4. **Configure Airbyte**
   - Set up SQL Server source connector (AdventureWorks database)
   - Set up Snowflake destination connector
   - Configure table mappings for all source tables

5. **Run Airbyte Sync**
   - Extract data from AdventureWorks
   - Load into Snowflake staging area
   - Transform and load into dimension and fact tables
   - Apply SCD Type 2 logic for historical tracking

6. **Verify Data Load**
   ```bash
   streamlit run verify_dwh_data.py
   ```
   - Check record counts
   - Validate SCD implementation
   - Verify referential integrity

### Phase 3: Market Segmentation (Analytics)

7. **Run Market Segmentation Pipeline**
   ```bash
   streamlit run run_segmentation_pipeline.py
   ```
   - Configure model parameters in sidebar
   - Click "Run Pipeline" button
   - Monitor training progress
   - Review segmentation results
   - Download results if needed

8. **Analyze Segmentation Results**
   ```bash
   streamlit run market_clustering_app.py
   ```
   - Explore segment characteristics
   - Compare markets within segments
   - Analyze temporal trends

### Phase 4: Ongoing Operations

9. **Scheduled Updates**
   - Run Airbyte sync on schedule (daily/weekly)
   - Re-run segmentation pipeline monthly to update segments
   - Monitor data quality with verification tool

10. **Model Iteration**
    - Adjust hyperparameters based on silhouette scores
    - Experiment with different sequence lengths
    - Try alternative clustering methods (K-Means vs DBSCAN)

## Data Warehouse Schema Details

### Slowly Changing Dimensions (SCD) Strategy

**SCD Type 0** (No changes tracked):
- `DimTime`: Time attributes never change
- `DimTerritory`: Geographic boundaries are static

**SCD Type 2** (Full history tracked):
- `DimProduct`: Track product attribute changes over time
- `DimSpecialOffer`: Track offer modifications
- `DimCustomer`: Track customer tier and loyalty changes
- `BridgeProductSpecialOffer`: Track product-offer relationships

**Implementation**:
- `ValidFrom`: Start date of record validity
- `ValidTo`: End date of record validity (9999-12-31 for active records)
- `IsActive`: Boolean flag for current records

### Fact Tables

**FactSale**:
- Grain: One record per sales order detail line
- Measures: Revenue, ProductQuantity
- Dimensions: Product, SpecialOffer, Territory, Time, Customer

**FactMarketSegmentation**:
- Grain: One record per market per analysis date
- Measures: Revenue, Quantity, ConfidenceScore
- Dimensions: Time (DateKey), Market (MarketNameKey)
- Attributes: SegmentID, EmbeddingJSON, ModelVersion

## Machine Learning Pipeline Details

### Market Segmentation Methodology

**Objective**: Group markets (countries/regions) into segments based on temporal sales patterns

**Approach**: Deep learning-based time-series clustering

**Steps**:

1. **Data Extraction**
   - Aggregate sales data by market and month
   - Calculate features: revenue, quantity, order count, customer count, growth rates
   - Extract most recent N months (configurable, default 12)

2. **Sequence Preparation**
   - Create fixed-length sequences for each market
   - Each sequence contains M months of historical data (configurable, default 6)
   - Features per time step: 9 metrics (revenue, quantity, etc.)

3. **Normalization**
   - Apply StandardScaler to ensure all features have similar scales
   - Fit on training data to prevent data leakage

4. **Model Training**
   - Architecture: LSTM autoencoder
   - Encoder: LSTM layers + fully connected projection
   - Decoder: Fully connected layers for reconstruction
   - Loss: Mean Squared Error (MSE) between input and reconstruction
   - Optimization: Adam optimizer

5. **Embedding Generation**
   - Extract encoder output (compressed representation)
   - Embedding dimension: 32 (configurable)
   - Each market represented as a 32-dimensional vector

6. **Clustering**
   - Algorithm: K-Means or DBSCAN
   - Input: Market embeddings
   - Output: Segment assignments
   - Evaluation: Silhouette score (higher is better, >0.3 is good)

7. **Result Storage**
   - Write to FactMarketSegmentation table
   - Include segment ID, embeddings (JSON), confidence score
   - Use MERGE statement to update existing records

### Model Configuration

**Recommended Settings**:
- Sequence Length: 6-12 months
- Number of Segments: 5-7
- Training Epochs: 50-100
- Batch Size: 16
- Learning Rate: 0.001

**Performance Considerations**:
- CPU training: 2-5 minutes for typical datasets
- GPU training: 30-60 seconds (if CUDA available)
- Inference: Near real-time

## Troubleshooting

### Common Issues

**Issue**: "No data extracted from Snowflake"
- **Cause**: Empty FactSale table or date filter excluding all data
- **Solution**: Verify Airbyte sync completed, check date ranges in data

**Issue**: "Insufficient data for sequences"
- **Cause**: Markets with fewer months than sequence_length parameter
- **Solution**: Reduce sequence_length or ensure sufficient historical data

**Issue**: "Low silhouette score"
- **Cause**: Poor clustering quality, markets not well-separated
- **Solution**: Adjust number of clusters, try different clustering method, increase training epochs

**Issue**: "CUDA out of memory"
- **Cause**: GPU memory insufficient for batch size
- **Solution**: Reduce batch_size or switch to CPU

**Issue**: "Connection to Snowflake failed"
- **Cause**: Invalid credentials or network issues
- **Solution**: Verify secrets.toml configuration, check network connectivity

## Performance Optimization

### Data Warehouse
- Create indexes on foreign keys in fact tables
- Partition FactSale by date for faster queries
- Use clustering keys on frequently filtered columns
- Materialize aggregations for common queries

### ML Pipeline
- Use GPU for faster training (10x speedup)
- Increase batch size for better GPU utilization
- Cache normalized sequences to avoid recomputation
- Parallelize embedding generation across batches

## Security Considerations

- Store Snowflake credentials in `.streamlit/secrets.toml` (excluded from git)
- Use role-based access control (RBAC) in Snowflake
- Limit warehouse permissions to minimum required
- Encrypt data in transit and at rest
- Audit pipeline executions and data access

## Future Enhancements

- Automated hyperparameter tuning
- Multi-model ensemble for robust segmentation
- Real-time segmentation updates via streaming
- Integration with BI tools (Tableau, Power BI)
- Automated anomaly detection in sales patterns
- Predictive analytics for future segment evolution

## Contact and Support

For questions or issues, please refer to the project documentation or contact the development team.

## License

This project is developed for academic purposes as part of the CO4031 Data Warehouse course.