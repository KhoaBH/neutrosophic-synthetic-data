Neutrosophic Synthetic Data Generation Framework on Apache Spark
Overview

This project implements a parallelized synthetic data generation framework based on Neutrosophic probability distributions.
The framework models data uncertainty through an indeterminacy parameter (IN) and supports large-scale parallel data generation on Apache Spark.

It is developed as part of the research project “A Parallelized Framework for Multi-Distribution Neutrosophic Synthetic Data on Apache Spark.”

Main Features

Generate synthetic data following multiple distributions: Erlang, Gaussian, Gamma, Uniform, Poisson, etc.

Integrate Neutrosophic logic to model uncertainty in distribution parameters.

Implemented on Apache Spark for large-scale parallel data generation.

Compare performance with Sequential Python and CTGAN (GAN-based) baselines.

Benchmark runtime, scalability, and statistical characteristics across distributions.

Technology Stack

Python

Apache Spark

NumPy / Pandas

CTGAN (for GAN-based baseline)

Citation

Nguyen H.D.T., Nguyen T., & Lam A.K. (2025).
A Parallelized Framework for Multi-Distribution Neutrosophic Synthetic Data on Apache Spark.
University of Information Technology (VNUHCM).

Would you like me to add a short “How to run” section (setup + example command) too? It’ll make the README complete for GitHub.
