# A Parallelized Framework for Multi-Distribution Neutrosophic Synthetic Data on Apache Spark

Official code repository for the paper: **"A Parallelized Framework for Multi-Distribution Neutrosophic Synthetic Data on Apache Spark"**.

This repository contains the core framework code and analysis scripts required to reproduce the paper's results.

---

## Getting Started

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/KhoaBH/neutrosophic-synthetic-data.git](https://github.com/KhoaBH/neutrosophic-synthetic-data.git)
    cd neutrosophic-synthetic-data
    ```
## File Descriptions

* `neutrosophic_data_generation.py`
    * The main script for the neutrosophic framework.
    * Run in **`MODE = "generate"`** to create the synthetic `.csv` files for Step 2.
    * Run in **`MODE = "benchmark"`** to reproduce the Spark vs. Sequential performance results for Table I.

* `gan_benchmark.py`
    * A separate script to benchmark the CTGAN (train and sample) performance for Table I.
* `data/`
    * Data set use for experiment (e.g., `pima_diabetes.csv`, `winequality-white-scaled.csv`) here.
* `results/`
    * The output directory where the experiment scripts will save their figures and `.txt` summary reports.


---
