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

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    **Note:** You must have a full installation of **Apache Spark** and **Java (JDK)** on your system for the `pyspark` library to function.

---

## File Descriptions

* `framework.py`
    * The main script for the neutrosophic framework.
    * Run in **`MODE = "generate"`** to create the synthetic `.csv` files for Step 2.
    * Run in **`MODE = "benchmark"`** to reproduce the Spark vs. Sequential performance results for Table I.

* `gan_benchmark.py`
    * A separate script to benchmark the CTGAN (train and sample) performance for Table I.

* `experiments/run_utility_[dataset].py`
    * (e.g., `run_utility_pima.py`, `run_utility_wine.py`, `run_utility_shopper.py`)
    * These are the analysis scripts. They run the 3x10-Fold Cross-Validation on the generated data.
    * They load the data, run the models, and output the final result figures (Figs 1-3) and summary reports (`.txt`) containing the metrics for Tables II, III, and the p-values.

* `data/`
    * A placeholder directory. You must place the input datasets (e.g., `pima_diabetes.csv`, `winequality-white-scaled.csv`) here.

* `results/`
    * The output directory where the experiment scripts will save their figures and `.txt` summary reports.

* `requirements.txt`
    * A list of all Python packages required to run the code (`pandas`, `pyspark`, `sdv`, `scikit-learn`, etc.).

---

## Citation

If you find this work useful, please cite the paper:

```bibtex
@inproceedings{Nguyen2026,
  title={A Parallelized Framework for Multi-Distribution Neutrosophic Synthetic Data on Apache Spark},
  author={Tri Nguyen Ho Duy and Tri Nguyen and Khoa Lam Anh},
  booktitle={Proceedings of the International Conference on Machine Learning and Computational Intelligence (MLCI)},
  year={2026},
  address={Ho Chi Minh City, Vietnam}
}