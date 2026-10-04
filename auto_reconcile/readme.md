# Auto Reconcile

## 1. Overview
The `auto_reconcile` project is an automated financial reconciliation tool designed specifically for SAP open items (using `BSEG` and `BKPF` tables). It is built to automatically group and match open accounting entries (debits and credits) that belong together so they can be cleared, ultimately saving manual effort for finance teams.

## 2. Functionality
The application takes in SAP transactional data, split into two datasets: **history** (items already cleared by human accountants) and **open items** (items currently pending). 

Its main functions are:
1. **Learning**: Analyzing historical data to learn the valid patterns accountants use to clear items.
2. **Solving**: Finding groups of open items that sum to zero (or within tolerance) while strictly adhering to both hard business rules and the learned patterns.
3. **Validating**: Running an independent check on proposed groups to ensure no constraints were violated.
4. **Reporting**: Outputting proposed clearing groups (`proposed_groups.csv`), items that require human review (`review_queue.csv`), and running a "time-travel" backtest to compare its proposals against what SAP actually did in the future.

## 3. Design of Project
The project architecture is composed of two primary modules working in sequence:
*   **Data Preparation (`source_data.py`)**: Responsible for extracting data (via Pandas, CSV, or PySpark), standardizing field names, calculating transaction signatures (e.g., Document Type + Debit/Credit indicator), and partitioning the data by constraints (Company Code, GL Account, Partner, Currency).
*   **Reconciliation Engine (`auto_reconcile.py`)**: 
    *   **Phase 1: Unsupervised Mining**: Learns from `.history`.
    *   **Phase 2: CSP Solver**: Applies a branch-and-bound optimization to `.open_items` partitioned by partner and currency.
    *   **Validation & Output**: Evaluates the results using 18-21 specific accounting constraint checks.

## 4. Algorithms Used

### Unsupervised Learning (FP-Growth)
The project deliberately avoids hardcoded rules for matching different document types (e.g., matching an Invoice `RV` with a Payment `DZ`). Instead, it uses **Unsupervised Learning** via the `mlxtend` library.
*   **Frequent Itemset Mining**: It uses the **FP-Growth** (Frequent Pattern Growth) algorithm to identify recurring combinations of document signatures in the historical cleared data.
*   **Association Rules**: It generates rules linking debit signatures to credit signatures, calculating metrics like *support*, *confidence*, and *lift*. This creates a probability matrix that scores how likely a proposed group of documents is valid based on past accounting behavior.

### Constraint Satisfaction Problem (CSP) Solver
Matching open items is a variation of the Subset Sum problem, which is NP-hard. The project tackles this using a **CSP solver with Branch-and-Bound Backtracking**:
*   The solver groups data into isolated partitions (same partner, same currency).
*   It performs a **Depth-First Search (DFS)** recursively exploring the tree of possible subsets.
*   It prioritizes items by absolute amount (largest first) to close large gaps quickly.
*   It evaluates candidates by checking them against constraints. If a partial sum makes reaching a zero-net balance impossible, or if the document types don't match historical patterns, the branch is **pruned** (bound), and the algorithm backtracks to try a different combination.
*   Groups are ranked using a scoring function based on historical support, date gaps, and exact reference matches.

## 5. Constraints for Backtracking
The backtracking algorithm rigorously applies two sets of constraints to prune the search space:

**Hard Constraints (Always Enforced):**
*   **Zero Net Balance (1/17)**: The sum of items in a proposed group must be zero (or within a minor currency tolerance).
*   **Basic Accounting (4-9)**: Grouped items must belong to the same company code, GL account, business partner, and currency. They must contain at least one debit and one credit.
*   **Reversal Pairs (16)**: Reversal documents (`stblg`) must be forced into the same group as their origin documents.

**Optional/Learned Constraints (Pruning Flags 11-15):**
*   **Pattern Allowed (11)**: Prunes the tree early using `subset_ok()`. If the current combination of item signatures was never seen in history, the search branch terminates immediately.
*   **Group Size Bounded (12)**: The number of lines cannot exceed the historical maximum for that specific pattern.
*   **Date Window (14)**: The gap in days between the oldest and newest item in a group cannot exceed the historical gap observed for that pattern (plus a slack buffer).
*   **Date Order (13)**: The proposed clearing date must be strictly on or after every member's posting date.
*   **References (15)**: Target invoice references (`REBZG`) must be present within the group if specified.

## 6. Conclusion
The `auto_reconcile` project is an advanced, flexible system that solves a complex combinatorial problem in financial accounting. By combining unsupervised machine learning (to understand *how* humans reconcile) with an optimized, constraint-driven backtracking solver (to *execute* the reconciliation), it ensures accurate, compliant, and highly probable matches without relying on brittle, hardcoded business logic.