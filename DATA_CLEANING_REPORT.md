# Data Cleaning Report

**Generated:** January 17, 2026
**System:** Product Recommendation System

---

## Executive Summary

| Metric | Loyal | New | Total |
|--------|-------|-----|-------|
| Original Records | 74,164 | 3,628 | 77,792 |
| Records Removed | 21 | 0 | **21** |
| Final Records | 74,143 | 3,628 | **77,771** |
| Data Retention Rate | 99.97% | 100% | **99.97%** |

---

## Records Removed - Detailed Breakdown

### Loyal Customers

| Issue | Records Found | Action | Records Removed |
|-------|---------------|--------|-----------------|
| Negative `units_sold` (returns) | 3 | **REMOVED** | 3 |
| Negative `ticket_amount` (refunds) | 18 | **REMOVED** | 18 |
| Duplicates | 0 | N/A | 0 |
| Null values | 0 | N/A | 0 |
| **Total** | **21** | | **21** |

### New Customers

| Issue | Records Found | Action | Records Removed |
|-------|---------------|--------|-----------------|
| Negative `units_sold` (returns) | 0 | N/A | 0 |
| Negative `ticket_amount` (refunds) | 0 | N/A | 0 |
| Duplicates | 0 | N/A | 0 |
| Null values | 0 | N/A | 0 |
| **Total** | **0** | | **0** |

---

## Detailed Issue Analysis

### 1. Negative `units_sold` (-1 Values) - REMOVED

**Found:** 3 records in Loyal Customers
**Action:** REMOVED (these are product returns)

| Row | user_id | item_id | units_sold | item_price | Interpretation |
|-----|---------|---------|------------|------------|----------------|
| 21013 | 41668606614 | 2100001856 | **-1** | $2.50 | Return |
| 21014 | 41668606614 | 2100003131 | **-1** | $2.50 | Return |
| 41374 | 41668606614 | 2150020600 | **-1** | $2.19 | Return |

**Note:** All 3 returns were by the same user (41668606614).

---

### 2. Negative `ticket_amount` - REMOVED

**Found:** 18 records in Loyal Customers
**Action:** REMOVED (these are refunds/adjustments)

| Value | Count | Interpretation |
|-------|-------|----------------|
| -$0.03 | Multiple | Small adjustment/rounding |
| -$2.10 | Multiple | Refund |

---

### 3. Null Values

| Dataset | Column | Null Count | Action |
|---------|--------|------------|--------|
| Loyal | All columns | **0** | None needed |
| New | All columns | **0** | None needed |

**Total Nulls Found:** 0

---

### 4. Duplicate Rows

| Dataset | Duplicates Found | Action |
|---------|------------------|--------|
| Loyal | **0** | None needed |
| New | **0** | None needed |

**Total Duplicates:** 0

---

### 5. Zero Price Items - KEPT

**Found:** 33 unique items with $0 price
**Action:** **KEPT** (provides valuable co-purchase signals)

| Dataset | Records | Unique Items |
|---------|---------|--------------|
| Loyal | 5,383 | 26 |
| New | 217 | 14 |
| **Combined** | **5,600** | **33** |

#### Top 10 Zero-Price Items by Occurrence

| Rank | Item ID | Loyal | New | Total | Loyal Users | New Users |
|------|---------|-------|-----|-------|-------------|-----------|
| 1 | 200 | 3,421 | 120 | 3,541 | 70 | 48 |
| 2 | OPEN DEPT 101 | 560 | 10 | 570 | 60 | 9 |
| 3 | 20098100000 | 358 | 16 | 374 | 52 | 8 |
| 4 | 20098900000 | 347 | 18 | 365 | 52 | 13 |
| 5 | OPEN DEPT 135 | 201 | 25 | 226 | 46 | 18 |
| 6 | OPEN DEPT 102 | 172 | 3 | 175 | 40 | 2 |
| 7 | OPEN DEPT 109 | 151 | 4 | 155 | 41 | 4 |
| 8 | 20098300000 | 130 | 5 | 135 | 37 | 4 |
| 9 | 20098800000 | 121 | 6 | 127 | 27 | 4 |
| 10 | OPEN DEPT 131 | 111 | 0 | 111 | 11 | 0 |

#### All Zero-Price Items (33 total)

```
200, 501, 7675006570, 7675051299, 7675051325, 7675053740, 7675054685,
20098100000, 20098300000, 20098500000, 20098600000, 20098700000,
20098800000, 20098900000, 20099100000,
OPEN DEPT 101, OPEN DEPT 102, OPEN DEPT 103, OPEN DEPT 104, OPEN DEPT 105,
OPEN DEPT 106, OPEN DEPT 107, OPEN DEPT 108, OPEN DEPT 109, OPEN DEPT 110,
OPEN DEPT 111, OPEN DEPT 112, OPEN DEPT 113, OPEN DEPT 129, OPEN DEPT 130,
OPEN DEPT 131, OPEN DEPT 134, OPEN DEPT 135
```

---

## Data Transformations Applied

### Column Standardization

| Original Name | Standardized Name |
|---------------|-------------------|
| Pos Number | pos_number |
| Ticket Number | ticket_number |
| Ticket Datetime | ticket_datetime |
| Ticket Amount | ticket_amount |
| User ID | user_id |
| Item Id | item_id |
| Units Sold | units_sold |
| Item Price | item_price |

### Data Type Conversions

| Column | Converted To | Reason |
|--------|--------------|--------|
| item_id | string | Consistency |
| user_id | string | Consistency |
| ticket_datetime | datetime64 | Date operations |
| item_price | float64 | Numeric operations |
| units_sold | int64 | Numeric operations |
| ticket_amount | float64 | Numeric operations |

---

## Final Dataset Statistics

### Loyal Customers (After Cleaning)

| Metric | Value |
|--------|-------|
| Records | 74,143 |
| Unique Users | 70 |
| Unique Items | 10,148 |
| Date Range | Jul 1, 2024 - Jan 3, 2026 |

### New Customers (After Cleaning)

| Metric | Value |
|--------|-------|
| Records | 3,628 |
| Unique Users | 50 |
| Unique Items | 1,997 |
| Date Range | Dec 1, 2025 - Dec 31, 2025 |

---

## Output Files

| File | Description | Records |
|------|-------------|---------|
| `loyal_customers_cleaned.csv` | Cleaned loyal transactions | 74,143 |
| `new_customers_cleaned.csv` | Cleaned new transactions | 3,628 |
| `zero_price_items.csv` | Log of zero-price items | 33 items |
| `cleaning_summary.csv` | Cleaning metadata & stats | 1 row |

---

## Summary of Actions Taken

| Issue | Found | Action | Removed |
|-------|-------|--------|---------|
| `units_sold < 0` (returns) | 3 | **REMOVED** | 3 |
| `ticket_amount < 0` (refunds) | 18 | **REMOVED** | 18 |
| `item_price = 0` (zero price) | 5,600 | **KEPT** | 0 |
| Null values | 0 | N/A | 0 |
| Duplicate rows | 0 | N/A | 0 |
| **TOTAL REMOVED** | | | **21** |

---

## Data Quality Score: EXCELLENT (99.97% retention)

| Dimension | Score | Notes |
|-----------|-------|-------|
| Completeness | 100% | No missing values |
| Uniqueness | 100% | No duplicate rows |
| Validity | 100% | Invalid records removed |
| Consistency | 100% | Column names and types standardized |

---

*Report generated by Data Cleaning Pipeline v1.0*
