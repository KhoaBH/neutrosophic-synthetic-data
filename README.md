# A Parallelized Framework for Multi-Distribution Neutrosophic Synthetic Data on Apache Spark

Official code repository for the paper: **"A Parallelized Framework for Multi-Distribution Neutrosophic Synthetic Data on Apache Spark"**.

This repository contains the framework code (Spark and Sequential) and the scripts to reproduce all results, tables, and figures in the paper.

---

## Getting Started

1.  **Clone the repository:**

    ```bash
    git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
    cd your-repo-name
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    **Note:** This project requires you to have **Apache Spark** and **Java (JDK)** installed and correctly configured on your system for `pyspark` to function.

---

## Reproducing Results

Follow these steps to regenerate all Tables and Figures from the paper.

### 1. Performance Benchmark (Table I)

The results for Table I are generated from two separate scripts.

**a) Spark vs. Sequential Comparison:**
Run `framework.py` in `benchmark` mode.

```bash
# (You must edit the MODE = "benchmark" variable in the framework.py file)
python framework.py
```
