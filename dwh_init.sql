USE ASSIGNMENT;

DROP TABLE IF EXISTS DWH.DIMPRODUCT;
CREATE TABLE DWH.DIMPRODUCT(
ProductSuggorateKey NUMBER AUTOINCREMENT PRIMARY KEY,
Product_ID NUMBER NOT NULL,
ProductName VARCHAR,
Subcategory VARCHAR,
Model VARCHAR,
ProductLine VARCHAR,
Class VARCHAR,
Style VARCHAR,
ValidFrom TIMESTAMP_NTZ,
ValidTo TIMESTAMP_NTZ,
IsActive BOOLEAN DEFAULT TRUE
);


DROP TABLE IF EXISTS DWH.DIMSPECIALOFFER;
CREATE TABLE DWH.DIMSPECIALOFFER(
SpecialOfferSuggorateKey NUMBER AUTOINCREMENT PRIMARY KEY,
SpecialOffer_ID NUMBER NOT NULL,
DiscountPct NUMBER(5,2),
Type VARCHAR,
Category VARCHAR,
ValidFrom TIMESTAMP_NTZ,
ValidTo TIMESTAMP_NTZ,
IsActive BOOLEAN DEFAULT TRUE
);

DROP TABLE IF EXISTS DWH.DIMTERRITORY;
CREATE TABLE DWH.DIMTERRITORY (
TerritorySuggorateKey NUMBER AUTOINCREMENT PRIMARY KEY,
Territory_ID NUMBER NOT NULL,
CountryRegion VARCHAR
);

-- Create table
--DimTime--
DROP TABLE IF EXISTS DWH.DIMTIME;
CREATE TABLE DWH.DIMTIME(
    TimeSuggorateKey NUMBER AUTOINCREMENT PRIMARY KEY,
    DateKey NUMBER NOT NULL UNIQUE,  -- YYYYMMDD format
    DAY INT,
    Month INT,
    Quarter INT,
    Season VARCHAR,
    Year INT
);


DROP TABLE IF EXISTS DWH.DIMCUSTOMER;
CREATE TABLE DWH.DIMCUSTOMER(
CustomerSuggorateKey NUMBER AUTOINCREMENT PRIMARY KEY,
Customer_ID NUMBER NOT NULL,
Gender VARCHAR,
LifeTimeValue VARCHAR,
LoyaltyStatus VARCHAR,
ValidFrom TIMESTAMP_NTZ,
ValidTo TIMESTAMP_NTZ,
IsActive BOOLEAN DEFAULT TRUE
);

--SCD TYPE 2
DROP TABLE IF EXISTS DWH.BridgeProductSpecialOffer;
CREATE TABLE DWH.BridgeProductSpecialOffer(
BrdgProductSpecialOfferKey NUMBER AUTOINCREMENT PRIMARY KEY,
ProductSuggorateKey NUMBER NOT NULL,
SpecialOfferSuggorateKey NUMBER NOT NULL,
ValidFrom TIMESTAMP_NTZ,
ValidTo TIMESTAMP_NTZ,
IsActive BOOLEAN DEFAULT TRUE
);
-- Lấy dữ liệu từ bảng orderdetails


DROP TABLE IF EXISTS DWH.FACTSALE;
CREATE TABLE DWH.FACTSALE(
   SalesOrderID NUMBER,
   SalesOrderDetailID NUMBER,
   BrdgProductSpecialOfferKey NUMBER,
   DimTerritoryKey NUMBER,
   DimTimeKey NUMBER,
   DimCustomerKey NUMBER,
   Revenue NUMBER(18,2),
   ProductQuantity NUMBER,
   ProductRange VARCHAR,
   PRIMARY KEY (SalesOrderID, SalesOrderDetailID)
);

-- Market Segmentation Fact Table (populated by data mining pipeline)
DROP TABLE IF EXISTS DWH.FACTMARKETSEGMENTATION;
CREATE TABLE DWH.FACTMARKETSEGMENTATION(
    DateKey NUMBER NOT NULL,                    -- Reference to DimTime.DateKey (YYYYMMDD format)
    MarketNameKey VARCHAR NOT NULL,             -- Market identifier (e.g., CountryRegion)
    SegmentID NUMBER NOT NULL,                  -- Cluster/Segment ID from ML model
    Revenue NUMBER(18,2),                       -- Aggregated revenue for this market-date
    Quantity NUMBER,                            -- Aggregated quantity sold
    ProductRange VARCHAR,                       -- Product range category
    ModelVersion VARCHAR,                       -- ML model version used for segmentation
    EmbeddingJSON VARCHAR,                      -- JSON string of the embedding vector (optional)
    ConfidenceScore NUMBER(5,4),               -- Clustering confidence/silhouette score
    CreatedAt TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP(),
    PRIMARY KEY (DateKey, MarketNameKey)
);




