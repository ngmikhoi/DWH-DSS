USE ASSIGNMENT;
USE SCHEMA DWH;
/* =========================
   DIMPRODUCT (SCD2)
   ========================= */
CREATE OR REPLACE TEMP TABLE stg_product AS
WITH latest AS (
  SELECT
    p.ProductID                                 AS Product_ID,
    p.Name                                      AS ProductName,
    pm.Name                                     AS Subcategory,
    psc.Name                                    AS Model,
    p.ProductLine,
    p.Class,
    p.Style,
    COALESCE(p.ModifiedDate, CURRENT_DATE()) AS change_ts,
    ROW_NUMBER() OVER (PARTITION BY p.ProductID ORDER BY p.ModifiedDate DESC NULLS LAST) AS rn
  FROM AIRBYTE.PRODUCT p
  LEFT JOIN AIRBYTE.PRODUCTMODEL pm ON pm.ProductModelID = p.ProductModelID
  LEFT JOIN AIRBYTE.PRODUCTSUBCATEGORY psc ON psc.ProductSubcategoryID = p.ProductSubcategoryID
)
SELECT * FROM latest WHERE rn = 1;

-- 1) Close changed active product rows
UPDATE DWH.DIMPRODUCT AS tgt
SET
  ValidTo = DATEADD(second, -1, s.change_ts),
  IsActive = FALSE
FROM stg_product AS s
WHERE tgt.Product_ID = s.Product_ID
  AND tgt.IsActive = TRUE
  AND (
       NVL(tgt.ProductName,'') <> NVL(s.ProductName,'')
    OR NVL(tgt.Subcategory,'') <> NVL(s.Subcategory,'')
    OR NVL(tgt.Model,'')       <> NVL(s.Model,'')
    OR NVL(tgt.ProductLine,'') <> NVL(s.ProductLine,'')
    OR NVL(tgt.Class,'')       <> NVL(s.Class,'')
    OR NVL(tgt.Style,'')       <> NVL(s.Style,'')
  );

-- 2) Insert new product versions where no active row exists
INSERT INTO DWH.DIMPRODUCT (
  Product_ID, ProductName, Subcategory, Model, ProductLine,
  Class, Style, ValidFrom, ValidTo, IsActive
)
SELECT
  s.Product_ID, s.ProductName, s.Subcategory, s.Model, s.ProductLine,
  s.Class, s.Style,
  COALESCE(s.change_ts, CURRENT_DATE()),
  TIMESTAMP '9999-12-31 23:59:59',
  TRUE
FROM stg_product s
WHERE NOT EXISTS (
  SELECT 1 FROM DWH.DIMPRODUCT t WHERE t.Product_ID = s.Product_ID AND t.IsActive = TRUE
);


/* =========================
   DIMSPECIALOFFER (SCD2)
   ========================= */
CREATE OR REPLACE TEMP TABLE stg_specialoffer AS
WITH latest AS (
  SELECT
    so.SpecialOfferID AS SpecialOffer_ID,
    so.DiscountPct,
    so.Type,
    so.Category,
    COALESCE(so.ModifiedDate, CURRENT_DATE()) AS change_ts,
    ROW_NUMBER() OVER (PARTITION BY so.SpecialOfferID ORDER BY so.ModifiedDate DESC NULLS LAST) AS rn
  FROM AIRBYTE.SPECIALOFFER so
)
SELECT * FROM latest WHERE rn = 1;

-- 1) Close changed active specialoffer rows
UPDATE DWH.DIMSPECIALOFFER AS tgt
SET
  ValidTo = DATEADD(second, -1, s.change_ts),
  IsActive = FALSE
FROM stg_specialoffer AS s
WHERE tgt.SpecialOffer_ID = s.SpecialOffer_ID
  AND tgt.IsActive = TRUE
  AND (
       NVL(tgt.DiscountPct,-1) <> NVL(s.DiscountPct,-1)
    OR NVL(tgt.Type,'')        <> NVL(s.Type,'')
    OR NVL(tgt.Category,'')    <> NVL(s.Category,'')
  );

-- 2) Insert new specialoffer versions
INSERT INTO DWH.DIMSPECIALOFFER (
  SpecialOffer_ID, DiscountPct, Type, Category, ValidFrom, ValidTo, IsActive
)
SELECT
  s.SpecialOffer_ID, s.DiscountPct, s.Type, s.Category,
  COALESCE(s.change_ts, CURRENT_DATE()),
  TIMESTAMP '9999-12-31 23:59:59',
  TRUE
FROM stg_specialoffer s
WHERE NOT EXISTS (
  SELECT 1 FROM DWH.DIMSPECIALOFFER t WHERE t.SpecialOffer_ID = s.SpecialOffer_ID AND t.IsActive = TRUE
);


-- =========================
-- DIMCUSTOMER (SCD2) - 
-- =========================
CREATE OR REPLACE TEMP TABLE stg_customer_norm AS
SELECT
  c.CustomerID AS Customer_ID,
  UPPER(TRIM(
    CASE
      WHEN p.Title = 'Mr.' THEN 'Male'
      WHEN p.Title IN ('Mrs.','Ms.','Miss') THEN 'Female'
      ELSE 'Unknown'
    END
  )) AS Gender,
  ROUND(COALESCE(SUM(sod.LineTotal), 0), 2) AS LifeTimeValue,
  COALESCE(c.ModifiedDate, MAX(soh.OrderDate), CURRENT_DATE()) AS change_ts
FROM AIRBYTE.CUSTOMER c
LEFT JOIN AIRBYTE.PERSON p ON p.BusinessEntityID = c.PersonID
LEFT JOIN AIRBYTE.SALESORDERHEADER soh ON soh.CustomerID = c.CustomerID
LEFT JOIN AIRBYTE.SALESORDERDETAIL sod ON sod.SalesOrderID = soh.SalesOrderID
GROUP BY c.CustomerID, p.Title, c.ModifiedDate;

-- 2) 12-month revenue (rounded)
CREATE OR REPLACE TEMP TABLE rev_12m AS
SELECT
  h.CustomerID,
  ROUND(COALESCE(SUM(d.LineTotal), 0), 2) AS revenue_12m
FROM AIRBYTE.SALESORDERHEADER h
JOIN AIRBYTE.SALESORDERDETAIL d ON h.SalesOrderID = d.SalesOrderID
WHERE h.OrderDate BETWEEN DATEADD(month, -12, CURRENT_DATE()) AND CURRENT_DATE()
GROUP BY h.CustomerID;

-- 3) FINAL STAGING – with normalized loyalty labels
CREATE OR REPLACE TEMP TABLE stg_customer_final_norm AS
SELECT
  s.Customer_ID,
  s.Gender,
  s.LifeTimeValue,
  s.change_ts,
  CASE
    WHEN COALESCE(r.revenue_12m, 0) >= 2000 THEN 'PLATINUM'
    WHEN COALESCE(r.revenue_12m, 0) >= 1000 THEN 'GOLD'
    WHEN COALESCE(r.revenue_12m, 0) >= 500  THEN 'SILVER'
    ELSE 'BRONZE'
  END AS LoyaltyStatus
FROM stg_customer_norm s
LEFT JOIN rev_12m r ON r.CustomerID = s.Customer_ID;

-- 4) BUILD CHANGESET – compare normalized staging to current active dim
CREATE OR REPLACE TEMP TABLE stg_customer_changes AS
SELECT
  s.Customer_ID,
  s.Gender                       AS src_Gender,
  s.LifeTimeValue                AS src_LifeTimeValue,
  s.change_ts                    AS src_change_ts,
  s.LoyaltyStatus                AS src_LoyaltyStatus,

  t.CustomerSuggorateKey,
  UPPER(TRIM(t.Gender))          AS tgt_Gender,
  ROUND(t.LifeTimeValue,2)       AS tgt_LifeTimeValue,
  UPPER(TRIM(t.LoyaltyStatus))   AS tgt_LoyaltyStatus,

  CASE
    WHEN t.Customer_ID IS NULL THEN TRUE
    WHEN NVL(UPPER(TRIM(t.Gender)), '') <> NVL(s.Gender, '') THEN TRUE
    WHEN NVL(ROUND(t.LifeTimeValue,2), 0) <> NVL(s.LifeTimeValue, 0) THEN TRUE
    WHEN NVL(UPPER(TRIM(t.LoyaltyStatus)), '') <> NVL(s.LoyaltyStatus, '') THEN TRUE
    ELSE FALSE
  END AS is_changed
FROM stg_customer_final_norm s
LEFT JOIN (
  SELECT Customer_ID, CustomerSuggorateKey, Gender, LifeTimeValue, LoyaltyStatus
  FROM DWH.DIMCUSTOMER
  WHERE IsActive = TRUE
) t
  ON t.Customer_ID = s.Customer_ID;

BEGIN TRANSACTION;
UPDATE DWH.DIMCUSTOMER AS tgt
SET
  ValidTo = DATEADD(second, -1, c.src_change_ts),
  IsActive = FALSE
FROM stg_customer_changes AS c
WHERE tgt.Customer_ID = c.Customer_ID
  AND tgt.IsActive = TRUE
  AND c.is_changed = TRUE;
  
INSERT INTO DWH.DIMCUSTOMER (
  Customer_ID,
  Gender,
  LifeTimeValue,
  LoyaltyStatus,
  ValidFrom,
  ValidTo,
  IsActive
)
SELECT
  c.Customer_ID,
  c.src_Gender,
  c.src_LifeTimeValue,
  c.src_LoyaltyStatus,
  COALESCE(c.src_change_ts, CURRENT_DATE()),
  TIMESTAMP '9999-12-31 23:59:59',
  TRUE
FROM stg_customer_changes c
WHERE c.is_changed = TRUE
  AND NOT EXISTS (
    SELECT 1
    FROM DWH.DIMCUSTOMER t
    WHERE t.Customer_ID = c.Customer_ID
      AND t.IsActive = TRUE
  );
COMMIT;

/* =========================
   BridgeProductSpecialOffer
   ========================= */

-- 1) staging: map productID & specialOfferID -> suggorate keys (only active dim rows)
CREATE OR REPLACE TEMP TABLE stg_bridge AS
WITH raw AS (
  SELECT DISTINCT
    sop.ProductID,
    sop.SpecialOfferID,
    COALESCE(sop.ModifiedDate, CURRENT_DATE()) AS change_ts
  FROM AIRBYTE.SPECIALOFFERPRODUCT sop
)
SELECT
  dp.ProductSuggorateKey,
  dso.SpecialOfferSuggorateKey,
  r.change_ts
FROM raw r
LEFT JOIN DWH.DIMPRODUCT dp
  ON dp.Product_ID = r.ProductID
 AND dp.IsActive = TRUE
LEFT JOIN DWH.DIMSPECIALOFFER dso
  ON dso.SpecialOffer_ID = r.SpecialOfferID
 AND dso.IsActive = TRUE
WHERE dp.ProductSuggorateKey IS NOT NULL
  AND dso.SpecialOfferSuggorateKey IS NOT NULL;

-- 2) Close (deactivate) active bridge rows that are NOT present in the staging feed
BEGIN TRANSACTION;

UPDATE DWH.BridgeProductSpecialOffer tgt
SET
  ValidTo = DATEADD(second, -1, CURRENT_TIMESTAMP()),
  IsActive = FALSE
FROM (
  SELECT ProductSuggorateKey, SpecialOfferSuggorateKey FROM stg_bridge
) s
WHERE tgt.IsActive = TRUE
  AND NOT EXISTS (
    SELECT 1
    FROM stg_bridge sb
    WHERE sb.ProductSuggorateKey = tgt.ProductSuggorateKey
      AND sb.SpecialOfferSuggorateKey = tgt.SpecialOfferSuggorateKey
  );

-- 3) Insert new bridge rows for pairs present in staging that do not have an active row
INSERT INTO DWH.BridgeProductSpecialOffer (
  ProductSuggorateKey,
  SpecialOfferSuggorateKey,
  ValidFrom,
  ValidTo,
  IsActive
)
SELECT
  s.ProductSuggorateKey,
  s.SpecialOfferSuggorateKey,
  COALESCE(s.change_ts, CURRENT_DATE()) AS ValidFrom,
  TIMESTAMP '9999-12-31 23:59:59' AS ValidTo,
  TRUE
FROM stg_bridge s
WHERE NOT EXISTS (
  SELECT 1
  FROM DWH.BridgeProductSpecialOffer t
  WHERE t.ProductSuggorateKey = s.ProductSuggorateKey
    AND t.SpecialOfferSuggorateKey = s.SpecialOfferSuggorateKey
    AND t.IsActive = TRUE
);

COMMIT;


/* =========================
   DIMTERRIT0RY (SCD1)
   ========================= */
/* =========================
   DIMTERRIT0RY (SCD Type 0)
   ========================= */
CREATE OR REPLACE TEMP TABLE stg_territory AS
SELECT DISTINCT
  st.TerritoryID AS Territory_ID,
  cr.Name       AS CountryRegion
FROM AIRBYTE.SALESTERRITORY st
JOIN AIRBYTE.COUNTRYREGION cr ON cr.CountryRegionCode = st.CountryRegionCode;
MERGE INTO DWH.DIMTERRITORY tgt
USING stg_territory src
ON tgt.Territory_ID = src.Territory_ID
WHEN NOT MATCHED THEN
  INSERT (Territory_ID, CountryRegion) VALUES (src.Territory_ID, src.CountryRegion);


/* =========================
   DIMTIME (idempotent)
   ========================= */
MERGE INTO DWH.DIMTIME tgt
USING (
  SELECT
    TO_NUMBER(TO_CHAR(dt, 'YYYYMMDD')) AS DateKey,
    DATE_PART(day, dt)   AS DAY,
    DATE_PART(month, dt) AS Month,
    DATE_PART(quarter, dt) AS Quarter,
    CASE
      WHEN DATE_PART(month, dt) IN (12,1,2) THEN 'Winter'
      WHEN DATE_PART(month, dt) IN (3,4,5) THEN 'Spring'
      WHEN DATE_PART(month, dt) IN (6,7,8) THEN 'Summer'
      ELSE 'Fall'
    END AS Season,
    DATE_PART(year, dt) AS Year
  FROM (
    SELECT DATEADD(day, seq4(), TO_DATE('2000-01-01')) AS dt
    FROM TABLE(GENERATOR(ROWCOUNT => 15000))
  ) t
) src
ON tgt.DateKey = src.DateKey
WHEN NOT MATCHED THEN
  INSERT (DateKey, DAY, Month, Quarter, Season, Year)
  VALUES (src.DateKey, src.DAY, src.Month, src.Quarter, src.Season, src.Year);



  
/* =========================
   FACTSALE load
   ========================= */

INSERT INTO DWH.FACTSALE(
  SalesOrderID,
  SalesOrderDetailID,
  BrdgProductSpecialOfferKey,
  DimTerritoryKey,
  DimTimeKey,
  DimCustomerKey,
  Revenue,
  ProductQuantity
)
SELECT
  sod.SalesOrderID,
  sod.SalesOrderDetailID,
  COALESCE(b.BrdgProductSpecialOfferKey, -1)                             AS BrdgProductSpecialOfferKey,
  COALESCE(terr.TerritorySuggorateKey, -1)                               AS DimTerritoryKey,
  COALESCE(dt.DateKey, TO_NUMBER(TO_VARCHAR(CURRENT_DATE,'YYYYMMDD')))   AS DimTimeKey,
  COALESCE(dc.CustomerSuggorateKey, -1)                                  AS DimCustomerKey,
  sod.LineTotal,
  sod.OrderQty
FROM AIRBYTE.SALESORDERDETAIL sod
JOIN AIRBYTE.SALESORDERHEADER soh
  ON soh.SalesOrderID = sod.SalesOrderID

LEFT JOIN DWH.DIMPRODUCT dp
  ON dp.Product_ID = sod.ProductID
 -- AND CAST(soh.OrderDate AS DATE) >= CAST(dp.ValidFrom AS DATE)
 -- AND (dp.ValidTo IS NULL OR CAST(soh.OrderDate AS DATE) < CAST(dp.ValidTo AS DATE))

LEFT JOIN DWH.DIMSPECIALOFFER dso
  ON dso.SpecialOffer_ID = sod.SpecialOfferID
 -- AND CAST(soh.OrderDate AS DATE) >= CAST(dso.ValidFrom AS DATE)
 -- AND (dso.ValidTo IS NULL OR CAST(soh.OrderDate AS DATE) < CAST(dso.ValidTo AS DATE))

LEFT JOIN DWH.BridgeProductSpecialOffer b
  ON b.ProductSuggorateKey = dp.ProductSuggorateKey
 AND b.SpecialOfferSuggorateKey = dso.SpecialOfferSuggorateKey

LEFT JOIN AIRBYTE.ADDRESS addr
  ON addr.AddressID = soh.BillToAddressID
LEFT JOIN AIRBYTE.STATEPROVINCE sp
  ON sp.StateProvinceID = addr.StateProvinceID
LEFT JOIN AIRBYTE.SALESTERRITORY st
  ON st.TerritoryID = sp.TerritoryID
LEFT JOIN DWH.DIMTERRITORY terr
  ON terr.Territory_ID = st.TerritoryID

LEFT JOIN DWH.DIMCUSTOMER dc
  ON dc.Customer_ID = soh.CustomerID

LEFT JOIN DWH.DIMTIME dt
  ON dt.DateKey = TO_NUMBER(TO_VARCHAR(CAST(soh.OrderDate AS DATE), 'YYYYMMDD'));

