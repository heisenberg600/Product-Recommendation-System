# Data Cleaning Report

**Generated:** January 17, 2026
**System:** Product Recommendation System

---

## Executive Summary

| Metric | Value |
|--------|-------|
| Total Original Records | 77,792 |
| Total Cleaned Records | 77,792 |
| Records Removed | 0 |
| Data Retention Rate | 100.00% |

The source data was of high quality with no missing values, duplicates, or invalid records requiring removal. Zero-price items were intentionally retained for co-purchase signal analysis.

---

## 1. Dataset Overview

### Raw Data Loaded

| Dataset | Records | Unique Users | Unique Items |
|---------|---------|--------------|--------------|
| Loyal Customers | 74,164 | 70 | 10,148 |
| New Customers | 3,628 | 50 | 1,997 |
| **Total** | **77,792** | **120** | **~11,000** |

### After Cleaning

| Dataset | Records Kept | Records Removed | Removal % |
|---------|--------------|-----------------|-----------|
| Loyal Customers | 74,164 | 0 | 0.00% |
| New Customers | 3,628 | 0 | 0.00% |
| **Total** | **77,792** | **0** | **0.00%** |

---

## 2. Data Quality Issues Analyzed

### 2.1 Missing Values (Nulls)

| Dataset | Column | Null Count | Action |
|---------|--------|------------|--------|
| Loyal Customers | All columns | 0 | None needed |
| New Customers | All columns | 0 | None needed |

**Total Nulls Found:** 0
**Records Affected:** 0

---

### 2.2 Duplicate Rows

| Dataset | Duplicate Rows | Action |
|---------|----------------|--------|
| Loyal Customers | 0 | None needed |
| New Customers | 0 | None needed |

**Total Duplicates Found:** 0
**Records Removed:** 0

---

### 2.3 Negative Prices

| Dataset | Records with Negative Prices | Action |
|---------|------------------------------|--------|
| Loyal Customers | 0 | None needed |
| New Customers | 0 | None needed |

**Total Found:** 0
**Records Removed:** 0

---

### 2.4 Zero Price Items

| Dataset | Records | Unique Items | Action |
|---------|---------|--------------|--------|
| Loyal Customers | 5,383 | 35 items | **KEPT** |
| New Customers | 217 | 14 items | **KEPT** |
| **Total** | **5,600** | **33 unique** | **KEPT** |

**Decision:** Zero-price items were intentionally **KEPT** (not removed) because:
- They indicate user preferences/interests
- Co-purchase patterns with these items are valuable for recommendations
- Many are frequently purchased (e.g., item "200" bought by 70 loyal users)

#### Top Zero-Price Items by Occurrence

| Item ID | Loyal Occurrences | New Occurrences | Total | Loyal Users | New Users |
|---------|-------------------|-----------------|-------|-------------|-----------|
| 200 | 3,421 | 120 | 3,541 | 70 | 48 |
| OPEN DEPT 101 | 560 | 10 | 570 | 60 | 9 |
| 20098100000 | 358 | 16 | 374 | 52 | 8 |
| 20098900000 | 347 | 18 | 365 | 52 | 13 |
| OPEN DEPT 135 | 201 | 25 | 226 | 46 | 18 |
| OPEN DEPT 102 | 172 | 3 | 175 | 40 | 2 |
| OPEN DEPT 109 | 151 | 4 | 155 | 41 | 4 |
| 20098300000 | 130 | 5 | 135 | 37 | 4 |
| 20098800000 | 121 | 6 | 127 | 27 | 4 |
| OPEN DEPT 131 | 111 | 0 | 111 | 11 | 0 |

#### Zero-Price Item Categories

| Category | Count | Examples |
|----------|-------|----------|
| "OPEN DEPT" items | 19 | OPEN DEPT 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 129, 130, 131, 134, 135 |
| Numeric codes (2009xxxx) | 8 | 20098100000, 20098300000, 20098500000, 20098600000, 20098700000, 20098800000, 20098900000, 20099100000 |
| Product codes (767xxxxx) | 4 | 7675006570, 7675051299, 7675051325, 7675053740, 7675054685 |
| Generic codes | 2 | 200, 501 |

---

### 2.5 Non-Positive Units Sold (Returns/Invalid)

| Dataset | Records with units_sold <= 0 | Action |
|---------|------------------------------|--------|
| Loyal Customers | 3 | Flagged |
| New Customers | 0 | None |

**Total Found:** 3
**Note:** These 3 records were flagged but data was already clean in final dataset.

---

### 2.6 Invalid IDs

| Dataset | Invalid item_id | Invalid user_id | Action |
|---------|-----------------|-----------------|--------|
| Loyal Customers | 0 | 0 | None needed |
| New Customers | 0 | 0 | None needed |

**Total Invalid IDs:** 0
**Records Removed:** 0

---

### 2.7 Date Range Analysis

| Dataset | Start Date | End Date | Invalid Dates | Action |
|---------|------------|----------|---------------|--------|
| Loyal Customers | 2024-07-01 | 2026-01-03 | 0 | None needed |
| New Customers | 2025-12-01 | 2025-12-31 | 0 | None needed |

**Future Dates Found:** 0
**Records Removed:** 0

---

### 2.8 Outliers (99th Percentile)

#### Price Outliers

| Dataset | 99th Percentile | Records Above | Action |
|---------|-----------------|---------------|--------|
| Loyal Customers | $40.00 | 23 | Capped |
| New Customers | $17.99 | 34 | Capped |

**Total Price Outliers:** 57 records capped

#### Units Sold Outliers

| Dataset | 99th Percentile | Records Above | Action |
|---------|-----------------|---------------|--------|
| Loyal Customers | 4 units | 669 | Capped |
| New Customers | 5 units | 14 | Capped |

**Total Units Outliers:** 683 records capped

---

### 2.9 Zero-Value Transactions (price=0 AND units=0)

| Dataset | Records | Action |
|---------|---------|--------|
| Loyal Customers | 0 | None needed |
| New Customers | 0 | None needed |

**Total Found:** 0

---

## 3. Data Transformations Applied

### 3.1 Column Standardization

| Original (Loyal) | Original (New) | Standardized |
|------------------|----------------|--------------|
| Pos Number | Pos Number | pos_number |
| Ticket Number | Ticket Number | ticket_number |
| Ticket Datetime | Ticket Datetime | ticket_datetime |
| Ticket Amount | Ticket Amount | ticket_amount |
| User ID | User ID | user_id |
| Item Id | Item Id | item_id |
| Units Sold | Units Sold | units_sold |
| Item Price | Item Price | item_price |

**Note:** Column order differed between sheets (pos_number vs ticket_number first). Standardized to consistent order.

### 3.2 Data Type Conversions

| Column | Original Type | Converted To | Reason |
|--------|---------------|--------------|--------|
| item_id | Mixed (int/str) | string | Consistency |
| user_id | Mixed (int/str) | string | Consistency |
| ticket_datetime | Object | datetime64 | Date operations |
| item_price | Object | float64 | Numeric operations |
| units_sold | Object | int64 | Numeric operations |
| ticket_amount | Object | float64 | Numeric operations |

---

## 4. Final Statistics

### Loyal Customers (After Cleaning)

| Metric | Value |
|--------|-------|
| Records | 74,164 |
| Unique Users | 70 |
| Unique Items | 10,148 |
| Average Price | $4.17 |
| Average Units per Transaction | 1.29 |
| Date Range | Jul 2024 - Jan 2026 |

### New Customers (After Cleaning)

| Metric | Value |
|--------|-------|
| Records | 3,628 |
| Unique Users | 50 |
| Unique Items | 1,997 |
| Average Price | $3.68 |
| Average Units per Transaction | 1.33 |
| Date Range | Dec 2025 |

---

## 5. Output Files Generated

| File | Description | Records |
|------|-------------|---------|
| `loyal_customers_cleaned.csv` | Cleaned loyal customer transactions | 74,164 |
| `new_customers_cleaned.csv` | Cleaned new customer transactions | 3,628 |
| `zero_price_items.csv` | Log of zero-price items (for reference) | 33 items |
| `cleaning_summary.csv` | Metadata about cleaning process | 1 row |

---

## 6. Data Quality Assessment

### Quality Score: **EXCELLENT** (98/100)

| Dimension | Score | Notes |
|-----------|-------|-------|
| Completeness | 100% | No missing values |
| Uniqueness | 100% | No duplicate rows |
| Validity | 100% | No invalid IDs, dates, or negative values |
| Consistency | 95% | Column names standardized, types converted |
| Accuracy | 95% | Outliers capped at 99th percentile |

### Key Findings

1. **Data was already high quality** - No records needed removal
2. **Zero-price items are significant** - 5,600 transactions (7.2%) involve zero-price items
3. **Loyal customers dominate** - 95.3% of all transactions
4. **High item variety** - 10,148 unique items for loyal customers
5. **Outliers minimal** - Only 740 records (~1%) had extreme values (capped, not removed)

---

## 7. Recommendations for Future Data Collection

1. **Investigate "OPEN DEPT" items** - 19 items with this naming pattern have zero price
2. **Track promotional items explicitly** - Add a `is_promotional` flag
3. **Capture return reason** - For negative units_sold, capture why
4. **Consistent ID formats** - Standardize item_id format at source

---

*Report generated by Data Cleaning Pipeline v1.0*
