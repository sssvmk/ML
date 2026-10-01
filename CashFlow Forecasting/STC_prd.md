# Cash Flow Forecasting PRD: SAP AR, AP & Bank Data Logic

## 1. Overview
This document defines the data extraction and aggregation logic to build a comprehensive cash flow forecast using standard SAP tables. 

To provide a complete picture, the logic is broken down into **four distinct datasets**. Every dataset is strictly aggregated by the same core dimensions to allow for unified reporting:
1.  **Date** (Net Due Date, Cleared Date, or Bank Value Date)
2.  **Company Code** (`BUKRS`)
3.  **Company Currency** (Local Currency)

*(Note: "Vendor payment received in bank" is treated as "Vendor payments made from bank" — i.e., bank outflows — to represent the AP side of cash leaving the accounts).*

---

## Dataset 1: Customer Invoices (Expected & Cleared)
This dataset tracks the Accounts Receivable (AR) subledger. It contains both historical customer payments (actuals) and forecasted customer payments (expected).

**Source Tables:** `BSAD` (Cleared), `BSID` (Open)
*   **Indicator:** `SHKZG` = 'S' (Debit)
*   **Amount Field:** `DMBTR` (Amount in Local Currency)

### SQL Pseudocode
```sql
-- ACTUALS (Historical Cleared Customer Invoices)
SELECT 
    'AR_Actual_Cleared' AS Status,
    BUKRS               AS Company_Code,
    AUGDT               AS Cash_Flow_Date,     
    SUM(DMBTR)          AS Amount_Local_Currency  
FROM BSAD
WHERE SHKZG = 'S'                             
  AND AUGDT IS NOT NULL                       
GROUP BY BUKRS, AUGDT

UNION ALL

-- FORECAST (Expected Unpaid Customer Invoices)
SELECT 
    'AR_Expected_Unpaid' AS Status,
    BUKRS                AS Company_Code,
    CASE 
        WHEN ZBD3T > 0 THEN (ZFBDT + ZBD3T)
        WHEN ZBD2T > 0 THEN (ZFBDT + ZBD2T)
        ELSE (ZFBDT + ZBD1T) 
    END                  AS Cash_Flow_Date,     
    SUM(DMBTR)           AS Amount_Local_Currency
FROM BSID
WHERE SHKZG = 'S'
GROUP BY BUKRS, 
    CASE 
        WHEN ZBD3T > 0 THEN (ZFBDT + ZBD3T)
        WHEN ZBD2T > 0 THEN (ZFBDT + ZBD2T)
        ELSE (ZFBDT + ZBD1T) 
    END;
```

---

## Dataset 2: Customer Payments Received in Bank (Bank Inflows)
This dataset captures actual cash hitting the bank accounts from external statements (EBS) or internal bank accounts (IHC). It looks purely at incoming funds (+).

**Source Tables:** `FEBKO` & `FEBEP` (EBS), `BKKIT` (IHC)
*   **EBS Indicator:** `FEBEP-EPVOZ` = '+' (Incoming cash)
*   **IHC Indicator:** `BKKIT-A_S_H` = 'C' (Credit/Incoming)

### SQL Pseudocode
```sql
-- EXTERNAL BANK STATEMENTS (Incoming)
SELECT 
    'Bank_Inflow_EBS'   AS Status,
    A.BUKRS             AS Company_Code,
    A.WAERS             AS Company_Currency,
    B.VALUT             AS Cash_Flow_Date,  -- Value Date
    SUM(B.KWERT)        AS Amount_Local_Currency
FROM FEBKO A
JOIN FEBEP B ON A.KUKEY = B.KUKEY
WHERE B.EPVOZ = '+'                         -- Incoming receipt
GROUP BY A.BUKRS, A.WAERS, B.VALUT

UNION ALL

-- IN-HOUSE CASH (Incoming)
SELECT 
    'Bank_Inflow_IHC'   AS Status,
    BKKRS               AS Company_Code,    -- IHC Bank Area mapped to CoCode
    LCUR                AS Company_Currency, 
    VALUT               AS Cash_Flow_Date,
    SUM(LCURAM)         AS Amount_Local_Currency
FROM BKKIT
WHERE A_S_H = 'C'                           -- Credit
GROUP BY BKKRS, LCUR, VALUT;
```

---

## Dataset 3: Vendor Invoices (Expected & Cleared)
This dataset tracks the Accounts Payable (AP) subledger. It contains historical vendor payments (actuals) and forecasted vendor payments (expected).

**Source Tables:** `BSAK` (Cleared), `BSIK` (Open)
*   **Indicator:** `SHKZG` = 'H' (Credit)
*   **Amount Field:** `DMBTR` (Amount in Local Currency)
*   **Forecast Filter:** Exclude payment blocks (`ZLSPR` != '')

### SQL Pseudocode
```sql
-- ACTUALS (Historical Cleared Vendor Invoices)
SELECT 
    'AP_Actual_Cleared' AS Status,
    BUKRS               AS Company_Code,
    AUGDT               AS Cash_Flow_Date,     
    SUM(DMBTR)          AS Amount_Local_Currency  
FROM BSAK
WHERE SHKZG = 'H'                             
  AND AUGDT IS NOT NULL                       
GROUP BY BUKRS, AUGDT

UNION ALL

-- FORECAST (Expected Unpaid Vendor Invoices)
SELECT 
    'AP_Expected_Unpaid' AS Status,
    BUKRS                AS Company_Code,
    CASE 
        WHEN ZBD3T > 0 THEN (ZFBDT + ZBD3T)
        WHEN ZBD2T > 0 THEN (ZFBDT + ZBD2T)
        ELSE (ZFBDT + ZBD1T) 
    END                  AS Cash_Flow_Date,     
    SUM(DMBTR)           AS Amount_Local_Currency
FROM BSIK
WHERE SHKZG = 'H'
  AND ZLSPR = ''         -- Exclude Blocked Invoices                      
GROUP BY BUKRS, 
    CASE 
        WHEN ZBD3T > 0 THEN (ZFBDT + ZBD3T)
        WHEN ZBD2T > 0 THEN (ZFBDT + ZBD2T)
        ELSE (ZFBDT + ZBD1T) 
    END;
```

---

## Dataset 4: Vendor Payments Made from Bank (Bank Outflows)
This dataset captures actual cash leaving the bank accounts to pay vendors (or other liabilities) from external statements (EBS) or internal bank accounts (IHC). It looks purely at outgoing funds (-).

**Source Tables:** `FEBKO` & `FEBEP` (EBS), `BKKIT` (IHC)
*   **EBS Indicator:** `FEBEP-EPVOZ` = '-' (Outgoing cash)
*   **IHC Indicator:** `BKKIT-A_S_H` = 'D' (Debit/Outgoing)

### SQL Pseudocode
```sql
-- EXTERNAL BANK STATEMENTS (Outgoing)
SELECT 
    'Bank_Outflow_EBS'  AS Status,
    A.BUKRS             AS Company_Code,
    A.WAERS             AS Company_Currency,
    B.VALUT             AS Cash_Flow_Date,  -- Value Date
    SUM(B.KWERT)        AS Amount_Local_Currency
FROM FEBKO A
JOIN FEBEP B ON A.KUKEY = B.KUKEY
WHERE B.EPVOZ = '-'                         -- Outgoing Payment
GROUP BY A.BUKRS, A.WAERS, B.VALUT

UNION ALL

-- IN-HOUSE CASH (Outgoing)
SELECT 
    'Bank_Outflow_IHC'  AS Status,
    BKKRS               AS Company_Code,    -- IHC Bank Area mapped to CoCode
    LCUR                AS Company_Currency, 
    VALUT               AS Cash_Flow_Date,
    SUM(LCURAM)         AS Amount_Local_Currency
FROM BKKIT
WHERE A_S_H = 'D'                           -- Debit
GROUP BY BKKRS, LCUR, VALUT;
```

## 5. Forecasting & Algorithm Selection Methodology

Once the four datasets are extracted and aggregated, the time series forecasting and model selection process is executed independently for both the Accounts Receivable (AR) and Accounts Payable (AP) cash flows.

### 5.1 Variable Assignment & Data Testing
Before forecasting, data tests (e.g., stationarity, seasonality, causality) are performed on the respective datasets:
*   **AR Forecasting Process:**
    *   **Endogenous Variable:** Customer Payments Received in Bank (Dataset 2).
    *   **Exogenous Variable:** Customer Invoices (Dataset 1).
*   **AP Forecasting Process:**
    *   **Endogenous Variable:** Vendor Payments Made from Bank (Dataset 4).
    *   **Exogenous Variable:** Vendor Invoices (Dataset 3).

### 5.2 Execution & Rolling Backtesting
For both the AR and AP processes, for each of the algorithm, perform data tests, if test confirm fit for purpose - run algorithm :
#,Algorithm,Data Tests,Minimum observations
1,ARIMAX,"Stationarity, ACF/PACF, trend, exogenous relationship, lag effects, residual autocorrelation",50+
2,SARIMAX,ARIMAX tests + stable seasonality and seasonal period,100+
3,VARMAX,"Multiple-series correlation, stationarity, cross-series dependency, Granger causality, lag structure",100+
4,Dynamic Regression,"Target–exogenous relationships, lag effects, multicollinearity, residual autocorrelation",50+
5,State Space Models,"Latent trend/level, time-varying behavior, structural changes, missing observations",50+
6,Structural Time Series,"Trend, seasonality, cycles, level shifts, structural breaks",100+
7,Prophet,"Trend, seasonality, changepoints, holidays/events, irregular/missing dates",50+
8,Linear Regression,"Linear relationship, multicollinearity, residuals, heteroscedasticity, outliers",50+
9,Polynomial Regression,"Nonlinear/curved relationship, polynomial degree, overfitting",50+
10,Ridge Regression,Multicollinearity and high-dimensional correlated predictors,50+
11,Lasso Regression,Sparse relationships and usefulness/stability of feature selection,50+
12,ElasticNet,Sparse + correlated predictors; L1/L2 regularization balance,50+
13,Random Forest Regressor,"Nonlinear relationships, feature interactions, lag features, categorical encoding",200+
14,Extra Trees Regressor,Nonlinear/high-dimensional relationships and noisy features,200+
15,XGBoost,"Nonlinear effects, interactions, lag/rolling features, exogenous variables",200+
16,LightGBM,"Nonlinear/high-dimensional relationships, missing values, categorical features",500+
17,CatBoost,"Nonlinear relationships, categorical variables, high-cardinality dimensions",200+
18,SVR,"Nonlinear relationship, scaling, outliers, dimensionality, kernel suitability",100+
19,KNN Regression,"Local similarity, scaling, dimensionality, noise",200+
20,RNN,Sequential dependency and useful sequence-history length,500+
21,LSTM,Short/long temporal dependencies and sequence length,500+
22,GRU,Short/long temporal dependencies and sequence length,500+
23,TCN,Local/multi-scale temporal dependencies and receptive field,500+
24,WaveNet,Very long sequential dependencies and multi-scale temporal patterns,"1,000+"
25,DeepAR,Shared patterns across multiple related time series; probabilistic forecasting,"1,000+ total"
26,DeepState,"Latent state, changing trend/seasonality, probabilistic forecasting","1,000+ total"
27,DeepVAR,Cross-series dependencies and multivariate probabilistic forecasting,"1,000+ total"
28,TFT,"Static features, known-future/observed covariates, interactions, long/short dependencies","1,000+"
29,TiDE,"Long-horizon forecasting, nonlinear covariates, multivariate relationships","1,000+"
30,TSMixer,Multivariate temporal and cross-variable mixing,"1,000+"
31,TimesNet,Multiple periodicities and complex temporal/frequency patterns,"1,000+"
32,iTransformer,High-dimensional multivariate relationships and long dependencies,"1,000+"
33,Informer,Very long input sequences and long-horizon forecasting,"1,000+"
34,Autoformer,Trend/seasonal decomposition and long-horizon patterns,"1,000+"
35,FEDformer,"Frequency-domain patterns, seasonality and long-horizon forecasting","1,000+"
36,ETSformer,Structured trend/seasonal behavior and long-horizon forecasting,"1,000+"
37,N-BEATS,Trend/seasonality and long-horizon forecasting,500+
38,N-HiTS,Multi-scale temporal patterns and long-horizon forecasting,500+

each of test is logged data test results, evals including charts etc.

each of algorithm and their tests are implemented as seperate modules with inputs interfaces for data with endogenous variable and one or more exogenous variables in dataframe.

Use **rolling backtesting** so that every algorithm predicts the same historical periods that are already known.

### 5.3 Evaluation Metrics
For every algorithm in each rolling period, calculate the following six metrics, then combine the results across all periods to get an overall performance profile for each algorithm:
1.  **sMAPE**
2.  **WAPE**
3.  **MAE**
4.  **RMSE**
5.  **MASE**
6.  **Bias %**

### 5.4 Elimination & Ranking
Apply the following rules to filter and select the best algorithm:
1.  **Eliminate against Baseline (MASE):** Eliminate algorithms that do not beat the baseline, using MASE. If MASE is 1 or greater, the algorithm is not better than the chosen naive/seasonal-naive benchmark.
2.  **Eliminate Unacceptable Bias:** Eliminate algorithms with unacceptable bias. For example, if the business requires forecast bias to remain within ±2%, algorithms outside that range are removed.
3.  **Rank Remaining Algorithms:** Compare the remaining algorithms on each metric:
    *   Lowest sMAPE gets the best rank.
    *   Lowest WAPE gets the best rank.
    *   Lowest MAE gets the best rank.
    *   Lowest RMSE gets the best rank.
    *   Lowest MASE gets the best rank.
    *   Bias closest to zero gets the best rank.
4.  **Combine Ranks:** Combine those six ranks for each algorithm. The algorithm with the lowest combined rank becomes the statistical winner.
5.  **Consistency Check:** Finally, check whether that winner performs consistently across the different rolling periods. If it performs well only in a few periods but poorly in others, it should not automatically be accepted.

### 5.5 Logging and Final Output
For each process, log the results of all steps and formally declare the winning algorithm. 

The winner is the algorithm that consistently produces the best combination of accuracy and low bias across historical rolling forecast periods, after eliminating algorithms that fail the baseline or business-bias requirements.

## 6. Business Assumptions & Open Issues

Based on initial business feedback, the following design rules and assumptions have been locked in, alongside remaining open issues that require resolution.

### 6.1 Resolved Assumptions & Business Rules
*   **Rolling Periods:** The rolling backtest duration is set to **3 months** for each period.
*   **Currency:** It is assumed that all transactions occur in the Company Currency as represented in the document. No global exchange rate translation logic is required at this stage.
*   **Partial Payments:** The model assumes invoices are paid in full. Complexities regarding partial payments/residual items are out of scope.
*   **Payment Blocks & Disputes (AR & AP):** Any invoice marked with a blocking flag (including disputes) will be strictly **ignored** and excluded from the forecasted expected cash flows. 
*   **Missing Payment Method:** If the payment method is blank on the invoice item, it defaults to the Vendor/Customer Master Data. It is assumed the master data will always provide the default payment method.
*   **Data Extraction Frequency:** Data extraction will occur **daily** via a scheduled batch job.
*   **Algorithms:** The `Time series.csv` file acts as the definitive manifest of forecasting algorithms to be run against the data.



## conclusion -
declare winner for this data.
Every thing has to configurable.