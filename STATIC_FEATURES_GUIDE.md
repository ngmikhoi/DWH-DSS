# Static Features Reference Guide

## Overview
This document describes the static features extracted for each market in the hybrid segmentation pipeline.

## Feature List

### 1. **AvgDiscount** (Numeric: 0.0 - 1.0)
- **Description**: Average discount percentage offered in this market
- **Source**: `DIMSPECIALOFFER.DiscountPct`
- **Calculation**: `AVG(DiscountPct)` across all sales in the market
- **Business Meaning**: 
  - High values → Price-sensitive market, heavy promotions
  - Low values → Premium market, less reliance on discounts

### 2. **PlatinumShare** (Numeric: 0.0 - 1.0)
- **Description**: Proportion of Platinum-tier customers
- **Source**: `DIMCUSTOMER.LoyatyStatus`
- **Calculation**: `COUNT(Platinum) / COUNT(All Customers)`
- **Business Meaning**:
  - High values → High-value customer base
  - Low values → Fewer premium customers

### 3. **GoldShare** (Numeric: 0.0 - 1.0)
- **Description**: Proportion of Gold-tier customers
- **Source**: `DIMCUSTOMER.LoyatyStatus`
- **Calculation**: `COUNT(Gold) / COUNT(All Customers)`
- **Business Meaning**:
  - Indicates mid-tier customer concentration

### 4. **SilverShare** (Numeric: 0.0 - 1.0)
- **Description**: Proportion of Silver-tier customers
- **Source**: `DIMCUSTOMER.LoyatyStatus`
- **Calculation**: `COUNT(Silver) / COUNT(All Customers)`
- **Business Meaning**:
  - Indicates entry-level loyal customer concentration

### 5. **AvgLTV** (Numeric: Currency)
- **Description**: Average customer lifetime value in this market
- **Source**: `DIMCUSTOMER.LifeTimeValue`
- **Calculation**: `AVG(LifeTimeValue)` across all customers in the market
- **Business Meaning**:
  - High values → Valuable customer base
  - Low values → Lower customer value, potential for growth

### 6. **TotalProductLines** (Integer)
- **Description**: Number of distinct product lines sold in this market
- **Source**: `DIMPRODUCT.ProductLine`
- **Calculation**: `COUNT(DISTINCT ProductLine)`
- **Business Meaning**:
  - High values → Diverse product portfolio
  - Low values → Specialized/focused product offering

### 7. **TotalSubcategories** (Integer)
- **Description**: Number of distinct product subcategories sold
- **Source**: `DIMPRODUCT.Subcategory`
- **Calculation**: `COUNT(DISTINCT Subcategory)`
- **Business Meaning**:
  - High values → Wide product variety
  - Low values → Narrow product focus

## Feature Normalization

All static features are normalized using `StandardScaler` before clustering:
- **Mean**: Centered at 0
- **Std Dev**: Scaled to 1
- **Purpose**: Ensures features have equal weight in clustering

## Example Market Profile

```
Market: "United Kingdom"
├── AvgDiscount: 0.12 (12% average discount)
├── PlatinumShare: 0.25 (25% Platinum customers)
├── GoldShare: 0.35 (35% Gold customers)
├── SilverShare: 0.30 (30% Silver customers)
├── AvgLTV: $2,500
├── TotalProductLines: 4
└── TotalSubcategories: 15

Interpretation: Premium market with high-value customers,
moderate discounting, and diverse product portfolio.
```

## Combining with LSTM Embeddings

The static features are concatenated with LSTM embeddings:

```
Final Feature Vector = [
    LSTM_dim_0, LSTM_dim_1, ..., LSTM_dim_31,  # 32 dimensions
    AvgDiscount_normalized,                     # 1 dimension
    PlatinumShare_normalized,                   # 1 dimension
    GoldShare_normalized,                       # 1 dimension
    SilverShare_normalized,                     # 1 dimension
    AvgLTV_normalized,                          # 1 dimension
    TotalProductLines_normalized,               # 1 dimension
    TotalSubcategories_normalized               # 1 dimension
]
# Total: 39 dimensions
```

## Data Quality Checks

Before running the pipeline, verify:

1. **LoyatyStatus Values**: Should be one of {Platinum, Gold, Silver, Bronze}
2. **LifeTimeValue**: Should be non-negative numeric values
3. **DiscountPct**: Should be between 0.0 and 1.0
4. **No NULLs**: All markets should have values for all features

## Troubleshooting

### Issue: "No static features found for market X"
- **Cause**: Market exists in time-series data but not in static feature query
- **Solution**: Check that the market has active customers and sales in FACTSALE

### Issue: All static features are 0
- **Cause**: Data quality issue or incorrect join conditions
- **Solution**: Verify SCD Type 2 `IsActive = TRUE` filters are correct

### Issue: Silhouette score decreased after adding static features
- **Cause**: Static features may be too noisy or not informative
- **Solution**: 
  - Check feature distributions
  - Consider removing low-variance features
  - Adjust normalization strategy
