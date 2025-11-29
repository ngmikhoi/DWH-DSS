"""
Data Warehouse Verification Script
This script checks if data is loaded correctly in Snowflake DWH
"""

import streamlit as st
import pandas as pd
import snowflake.connector

def get_snowflake_connection():
    """Establish connection to Snowflake"""
    return snowflake.connector.connect(
        user=st.secrets["snowflake_user"],
        password=st.secrets["snowflake_password"],
        account=st.secrets["snowflake_account"],
        warehouse=st.secrets["snowflake_warehouse"],
        database=st.secrets["snowflake_database"],
        schema=st.secrets["snowflake_schema"],
    )

def check_table_counts(conn):
    """Check row counts for all tables"""
    cur = conn.cursor()
    
    tables = [
        'FACTSALE',
        'DIMPRODUCT', 
        'DIMSPECIALOFFER',
        'DIMTERRITORY',
        'DIMTIME',
        'DIMCUSTOMER',
        'BRIDGEPRODUCTSPECIALOFFER'
    ]
    
    results = {}
    for table in tables:
        try:
            cur.execute(f"SELECT COUNT(*) FROM {table}")
            count = cur.fetchone()[0]
            results[table] = count
        except Exception as e:
            results[table] = f"Error: {str(e)}"
    
    return results

def discover_table_schemas(conn):
    """Discover actual column names in each table"""
    cur = conn.cursor()
    
    tables = [
        'FACTSALE',
        'DIMPRODUCT', 
        'DIMSPECIALOFFER',
        'DIMTERRITORY',
        'DIMTIME',
        'DIMCUSTOMER',
        'BRIDGEPRODUCTSPECIALOFFER'
    ]
    
    schemas = {}
    for table in tables:
        try:
            cur.execute(f"SELECT * FROM {table} LIMIT 0")
            columns = [desc[0] for desc in cur.description]
            schemas[table] = columns
        except Exception as e:
            schemas[table] = f"Error: {str(e)}"
    
    return schemas

def check_join_validity(conn):
    """Check if foreign key relationships are valid"""
    cur = conn.cursor()
    
    checks = []
    
    # Check 1: FactSale -> BridgeProductSpecialOffer
    cur.execute("""
        SELECT COUNT(*) as orphaned_records
        FROM FACTSALE f
        LEFT JOIN BRIDGEPRODUCTSPECIALOFFER b 
            ON f.BrdgProductSpecialOfferKey = b.BrdgProductSpecialOfferKey
        WHERE b.BrdgProductSpecialOfferKey IS NULL
    """)
    orphaned = cur.fetchone()[0]
    checks.append({
        'Check': 'FactSale -> Bridge',
        'Orphaned Records': orphaned,
        'Status': '✅ OK' if orphaned == 0 else '❌ ISSUE'
    })
    
    # Check 2: Bridge -> DimProduct
    cur.execute("""
        SELECT COUNT(*) as orphaned_records
        FROM BRIDGEPRODUCTSPECIALOFFER b
        LEFT JOIN DIMPRODUCT p 
            ON b.ProductSuggorateKey = p.ProductSuggorateKey
        WHERE p.ProductSuggorateKey IS NULL
    """)
    orphaned = cur.fetchone()[0]
    checks.append({
        'Check': 'Bridge -> DimProduct',
        'Orphaned Records': orphaned,
        'Status': '✅ OK' if orphaned == 0 else '❌ ISSUE'
    })
    
    # Check 3: FactSale -> DimCustomer
    cur.execute("""
        SELECT COUNT(*) as orphaned_records
        FROM FACTSALE f
        LEFT JOIN DIMCUSTOMER c 
            ON f.DimCustomerKey = c.CustomerSuggorateKey
        WHERE c.CustomerSuggorateKey IS NULL
    """)
    orphaned = cur.fetchone()[0]
    checks.append({
        'Check': 'FactSale -> DimCustomer',
        'Orphaned Records': orphaned,
        'Status': '✅ OK' if orphaned == 0 else '❌ ISSUE'
    })
    
    # Check 4: FactSale -> DimTerritory
    cur.execute("""
        SELECT COUNT(*) as orphaned_records
        FROM FACTSALE f
        LEFT JOIN DIMTERRITORY t 
            ON f.DimTerritoryKey = t.TerritorySuggorateKey
        WHERE t.TerritorySuggorateKey IS NULL
    """)
    orphaned = cur.fetchone()[0]
    checks.append({
        'Check': 'FactSale -> DimTerritory',
        'Orphaned Records': orphaned,
        'Status': '✅ OK' if orphaned == 0 else '❌ ISSUE'
    })
    
    # Check 5: FactSale -> DimTime (DateKey is the join column)
    cur.execute("""
        SELECT COUNT(*) as orphaned_records
        FROM FACTSALE f
        LEFT JOIN DIMTIME t 
            ON f.DimTimeKey = t.DateKey
        WHERE t.DateKey IS NULL
    """)
    orphaned = cur.fetchone()[0]
    checks.append({
        'Check': 'FactSale -> DimTime',
        'Orphaned Records': orphaned,
        'Status': '✅ OK' if orphaned == 0 else '❌ ISSUE'
    })
    
    return pd.DataFrame(checks)

def check_scd_type2_validity(conn):
    """Check SCD Type 2 implementation for dimension tables"""
    cur = conn.cursor()
    
    scd2_checks = []
    
    # Tables with SCD Type 2 (excluding bridge table - handled separately)
    scd2_tables = {
        'DIMPRODUCT': ('Product_ID', 'ProductSuggorateKey'),
        'DIMSPECIALOFFER': ('SpecialOffer_ID', 'SpecialOfferSuggorateKey'),
        'DIMCUSTOMER': ('Customer_ID', 'CustomerSuggorateKey')
    }
    
    for table, (natural_key, surrogate_key) in scd2_tables.items():
        # Check 1: Multiple active records for same natural key
        cur.execute(f"""
            SELECT COUNT(*) as issue_count
            FROM (
                SELECT {natural_key}
                FROM {table}
                WHERE IsActive = TRUE
                GROUP BY {natural_key}
                HAVING COUNT(*) > 1
            )
        """)
        multiple_active = cur.fetchone()[0]
        scd2_checks.append({
            'Table': table,
            'Check': 'Multiple Active Records',
            'Issues Found': multiple_active,
            'Status': '✅ OK' if multiple_active == 0 else '❌ ISSUE'
        })
        
        # Check 2: Overlapping date ranges for same natural key
        cur.execute(f"""
            SELECT COUNT(*) as issue_count
            FROM {table} t1
            JOIN {table} t2
                ON t1.{natural_key} = t2.{natural_key}
                AND t1.{surrogate_key} < t2.{surrogate_key}
            WHERE t1.ValidFrom < t2.ValidTo
                AND t1.ValidTo > t2.ValidFrom
        """)
        overlapping = cur.fetchone()[0]
        scd2_checks.append({
            'Table': table,
            'Check': 'Overlapping Date Ranges',
            'Issues Found': overlapping,
            'Status': '✅ OK' if overlapping == 0 else '❌ ISSUE'
        })
        
        # Check 3: Active records should have ValidTo = '9999-12-31 23:59:59'
        cur.execute(f"""
            SELECT COUNT(*) as issue_count
            FROM {table}
            WHERE IsActive = TRUE
                AND ValidTo != '9999-12-31 23:59:59'
        """)
        invalid_active = cur.fetchone()[0]
        scd2_checks.append({
            'Table': table,
            'Check': 'Active Records ValidTo Check',
            'Issues Found': invalid_active,
            'Status': '✅ OK' if invalid_active == 0 else '❌ ISSUE'
        })
    
    # Special handling for Bridge table - natural key is Product + SpecialOffer combination
    table = 'BRIDGEPRODUCTSPECIALOFFER'
    
    # Check 1: Multiple active records for same Product+SpecialOffer combination
    cur.execute("""
        SELECT COUNT(*) as issue_count
        FROM (
            SELECT ProductSuggorateKey, SpecialOfferSuggorateKey
            FROM BRIDGEPRODUCTSPECIALOFFER
            WHERE IsActive = TRUE
            GROUP BY ProductSuggorateKey, SpecialOfferSuggorateKey
            HAVING COUNT(*) > 1
        )
    """)
    multiple_active = cur.fetchone()[0]
    scd2_checks.append({
        'Table': table,
        'Check': 'Multiple Active Records (Product+Offer)',
        'Issues Found': multiple_active,
        'Status': '✅ OK' if multiple_active == 0 else '❌ ISSUE'
    })
    
    # Check 2: Overlapping date ranges for same Product+SpecialOffer combination
    cur.execute("""
        SELECT COUNT(*) as issue_count
        FROM BRIDGEPRODUCTSPECIALOFFER t1
        JOIN BRIDGEPRODUCTSPECIALOFFER t2
            ON t1.ProductSuggorateKey = t2.ProductSuggorateKey
            AND t1.SpecialOfferSuggorateKey = t2.SpecialOfferSuggorateKey
            AND t1.BrdgProductSpecialOfferKey < t2.BrdgProductSpecialOfferKey
        WHERE t1.ValidFrom < t2.ValidTo
            AND t1.ValidTo > t2.ValidFrom
    """)
    overlapping = cur.fetchone()[0]
    scd2_checks.append({
        'Table': table,
        'Check': 'Overlapping Date Ranges',
        'Issues Found': overlapping,
        'Status': '✅ OK' if overlapping == 0 else '❌ ISSUE'
    })
    
    # Check 3: Active records should have ValidTo = '9999-12-31 23:59:59'
    cur.execute("""
        SELECT COUNT(*) as issue_count
        FROM BRIDGEPRODUCTSPECIALOFFER
        WHERE IsActive = TRUE
            AND ValidTo != '9999-12-31 23:59:59'
    """)
    invalid_active = cur.fetchone()[0]
    scd2_checks.append({
        'Table': table,
        'Check': 'Active Records ValidTo Check',
        'Issues Found': invalid_active,
        'Status': '✅ OK' if invalid_active == 0 else '❌ ISSUE'
    })
    
    
    df = pd.DataFrame(scd2_checks)
    df['Issues Found'] = df['Issues Found'].astype(str)
    return df


def check_dimtime_validity(conn):
    """Check DimTime table structure and data quality"""
    cur = conn.cursor()
    
    dimtime_checks = []
    
    # Check 1: DateKey format (should be YYYYMMDD)
    cur.execute("""
        SELECT COUNT(*) as issue_count
        FROM DIMTIME
        WHERE LENGTH(CAST(DateKey AS VARCHAR)) != 8
    """)
    invalid_format = cur.fetchone()[0]
    dimtime_checks.append({
        'Check': 'DateKey Format (YYYYMMDD)',
        'Issues Found': invalid_format,
        'Status': '✅ OK' if invalid_format == 0 else '❌ ISSUE'
    })
    
    # Check 2: Duplicate DateKeys
    cur.execute("""
        SELECT COUNT(*) as issue_count
        FROM (
            SELECT DateKey
            FROM DIMTIME
            GROUP BY DateKey
            HAVING COUNT(*) > 1
        )
    """)
    duplicates = cur.fetchone()[0]
    dimtime_checks.append({
        'Check': 'Duplicate DateKeys',
        'Issues Found': duplicates,
        'Status': '✅ OK' if duplicates == 0 else '❌ ISSUE'
    })
    
    # Check 3: Date range coverage
    cur.execute("""
        SELECT 
            MIN(DateKey) as min_date,
            MAX(DateKey) as max_date,
            COUNT(*) as total_dates
        FROM DIMTIME
    """)
    result = cur.fetchone()
    dimtime_checks.append({
        'Check': 'Date Range Coverage',
        'Issues Found': f'{result[0]} to {result[1]} ({result[2]:,} dates)',
        'Status': '✅ INFO'
    })
    
    
    df = pd.DataFrame(dimtime_checks)
    df['Issues Found'] = df['Issues Found'].astype(str)
    return df


def check_bridge_table_validity(conn):
    """Check Bridge table relationships and data quality"""
    cur = conn.cursor()
    
    bridge_checks = []
    
    # Check 1: Bridge records pointing to non-existent products
    cur.execute("""
        SELECT COUNT(*) as orphaned_count
        FROM BRIDGEPRODUCTSPECIALOFFER b
        LEFT JOIN DIMPRODUCT p 
            ON b.ProductSuggorateKey = p.ProductSuggorateKey
        WHERE p.ProductSuggorateKey IS NULL
    """)
    orphaned_products = cur.fetchone()[0]
    bridge_checks.append({
        'Check': 'Orphaned Product References',
        'Issues Found': orphaned_products,
        'Status': '✅ OK' if orphaned_products == 0 else '❌ ISSUE'
    })
    
    # Check 2: Bridge records pointing to non-existent special offers
    cur.execute("""
        SELECT COUNT(*) as orphaned_count
        FROM BRIDGEPRODUCTSPECIALOFFER b
        LEFT JOIN DIMSPECIALOFFER so 
            ON b.SpecialOfferSuggorateKey = so.SpecialOfferSuggorateKey
        WHERE so.SpecialOfferSuggorateKey IS NULL
    """)
    orphaned_offers = cur.fetchone()[0]
    bridge_checks.append({
        'Check': 'Orphaned SpecialOffer References',
        'Issues Found': orphaned_offers,
        'Status': '✅ OK' if orphaned_offers == 0 else '❌ ISSUE'
    })
    
    # Check 3: Count of active vs inactive bridge records
    cur.execute("""
        SELECT 
            SUM(CASE WHEN IsActive = TRUE THEN 1 ELSE 0 END) as active_count,
            SUM(CASE WHEN IsActive = FALSE THEN 1 ELSE 0 END) as inactive_count
        FROM BRIDGEPRODUCTSPECIALOFFER
    """)
    result = cur.fetchone()
    bridge_checks.append({
        'Check': 'Active Bridge Records',
        'Issues Found': str(result[0]) if result[0] is not None else '0',
        'Status': '✅ INFO'
    })
    bridge_checks.append({
        'Check': 'Inactive Bridge Records',
        'Issues Found': str(result[1]) if result[1] is not None else '0',
        'Status': '✅ INFO'
    })
    
    
    df = pd.DataFrame(bridge_checks)
    df['Issues Found'] = df['Issues Found'].astype(str)
    return df



def test_corrected_query(conn):
    """Test the corrected query with proper joins - simplified version"""
    cur = conn.cursor()
    
    # First, try a minimal query to verify joins work
    cur.execute("""
        SELECT
            f.Revenue,
            f.ProductQuantity,
            p.ProductName,
            t.CountryRegion,
            dc.Gender
        FROM FACTSALE f
        -- Correct join through bridge table
        JOIN BRIDGEPRODUCTSPECIALOFFER b 
            ON b.BrdgProductSpecialOfferKey = f.BrdgProductSpecialOfferKey
        JOIN DIMPRODUCT p 
            ON p.ProductSuggorateKey = b.ProductSuggorateKey
        JOIN DIMCUSTOMER dc 
            ON dc.CustomerSuggorateKey = f.DimCustomerKey
        JOIN DIMTERRITORY t 
            ON t.TerritorySuggorateKey = f.DimTerritoryKey
        JOIN DIMTIME dt 
            ON dt.DateKey = f.DimTimeKey
        LIMIT 100
    """)
    
    df = cur.fetch_pandas_all()
    return df

def main():
    st.set_page_config(page_title="DWH Data Verification", layout="wide")
    
    st.title("🔍 Data Warehouse Verification")
    st.markdown("This tool verifies that data is loaded correctly in your Snowflake DWH")
    
    try:
        conn = get_snowflake_connection()
        st.success("✅ Successfully connected to Snowflake!")
        
        # Section 1: Table Counts
        st.header("1️⃣ Table Row Counts")
        with st.spinner("Checking table counts..."):
            counts = check_table_counts(conn)
            counts_df = pd.DataFrame(list(counts.items()), columns=['Table', 'Row Count'])
            st.dataframe(counts_df, use_container_width=True)
        
        # Section 2: Discover Actual Schemas
        st.header("2️⃣ Actual Table Schemas")
        st.markdown("**Discovering actual column names in your Snowflake tables...**")
        with st.spinner("Discovering schemas..."):
            schemas = discover_table_schemas(conn)
            
            for table, columns in schemas.items():
                with st.expander(f"📋 {table}", expanded=False):
                    if isinstance(columns, str):
                        st.error(columns)
                    else:
                        st.write(f"**Columns ({len(columns)}):**")
                        # Display in columns for better readability
                        cols = st.columns(3)
                        for idx, col in enumerate(columns):
                            cols[idx % 3].write(f"• `{col}`")
        
        # Section 3: Foreign Key Validation
        st.header("3️⃣ Foreign Key Relationship Validation")
        with st.spinner("Validating foreign key relationships..."):
            fk_checks = check_join_validity(conn)
            st.dataframe(fk_checks, use_container_width=True)
            
            if (fk_checks['Orphaned Records'] > 0).any():
                st.error("⚠️ Found orphaned records! Some foreign key relationships are broken.")
            else:
                st.success("✅ All foreign key relationships are valid!")
        
        # Section 3.5: SCD Type 2 Validation
        st.header("3️⃣.5️⃣ SCD Type 2 Validation")
        st.markdown("**Checking temporal data quality for slowly changing dimensions...**")
        with st.spinner("Validating SCD Type 2 implementation..."):
            scd2_checks = check_scd_type2_validity(conn)
            st.dataframe(scd2_checks, use_container_width=True)
            
            if (scd2_checks['Status'].str.contains('ISSUE')).any():
                st.warning("⚠️ Found SCD Type 2 issues! Check for overlapping dates or multiple active records.")
            else:
                st.success("✅ All SCD Type 2 implementations are valid!")
        
        # Section 3.6: DimTime Validation
        st.header("3️⃣.6️⃣ DimTime Validation")
        st.markdown("**Checking time dimension data quality...**")
        with st.spinner("Validating DimTime table..."):
            dimtime_checks = check_dimtime_validity(conn)
            st.dataframe(dimtime_checks, use_container_width=True)
        
        # Section 3.7: Bridge Table Validation
        st.header("3️⃣.7️⃣ Bridge Table Validation")
        st.markdown("**Checking Product-SpecialOffer bridge relationships...**")
        with st.spinner("Validating bridge table..."):
            bridge_checks = check_bridge_table_validity(conn)
            st.dataframe(bridge_checks, use_container_width=True)
            
            if (bridge_checks['Status'].str.contains('ISSUE')).any():
                st.error("⚠️ Found issues in bridge table relationships!")
            else:
                st.success("✅ Bridge table relationships are valid!")
        
        # Section 4: Sample Data with Corrected Query
        st.header("4️⃣ Sample Data (Corrected Query)")
        st.markdown("**This uses the CORRECT join through the bridge table:**")
        
        with st.expander("📝 View SQL Query", expanded=False):
            st.code("""
SELECT
    f.Revenue,
    f.ProductQuantity,
    p.ProductName,
    t.CountryRegion,
    dc.Gender
FROM FACTSALE f
JOIN BRIDGEPRODUCTSPECIALOFFER b 
    ON b.BrdgProductSpecialOfferKey = f.BrdgProductSpecialOfferKey
JOIN DIMPRODUCT p 
    ON p.ProductSuggorateKey = b.ProductSuggorateKey
JOIN DIMCUSTOMER dc 
    ON dc.CustomerSuggorateKey = f.DimCustomerKey
JOIN DIMTERRITORY t 
    ON t.TerritorySuggorateKey = f.DimTerritoryKey
JOIN DIMTIME dt 
    ON dt.DateKey = f.DimTimeKey
LIMIT 100
            """, language="sql")
        
        with st.spinner("Loading sample data..."):
            sample_df = test_corrected_query(conn)
            st.write(f"**Loaded {len(sample_df)} sample records**")
            st.dataframe(sample_df, use_container_width=True)
            
            # Show data summary
            st.subheader("📊 Data Summary")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Revenue (sample)", f"${sample_df['REVENUE'].sum():,.2f}")
            with col2:
                st.metric("Avg Revenue", f"${sample_df['REVENUE'].mean():,.2f}")
            with col3:
                st.metric("Total Quantity", f"{sample_df['PRODUCTQUANTITY'].sum():,}")
            with col4:
                st.metric("Unique Products", sample_df['PRODUCTNAME'].nunique())
        
        conn.close()
        
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        st.exception(e)

if __name__ == "__main__":
    main()
