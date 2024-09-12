# Benchmarking Dimensionality Reduction Techniques on Chemical Datasets

## Introduction
This repository contains the data and scripts required to reproduce the results presented in our paper on benchmarking dimensionality reduction techniques applied to chemical datasets.

## Repository Structure

- **src**: Contains the essential code for data preprocessing, dimensionality reduction, optimization, analysis, and visualization.
- **datasets**: Contains the datasets utilized in the study.
- **notebooks**: Includes Jupyter notebooks used for data analysis and visualization.
- **results**: Stores the optimized low-dimensional embeddings and all calculated metrics.
- **scripts**: Includes master scripts for data preparation, running benchmarks, and analyzing results.

## Datasets
The `datasets` directory houses the chemical datasets used throughout the study.

## Results
The `results` directory includes the optimized low-dimensional embeddings and all associated metrics.

## Notebooks
The `notebooks` directory contains Jupyter notebooks for data analysis, visualization, and further exploration of the study's findings.

## Code
### Core code
The `src/cdr_bench` directory contains various components for dimensionality reduction benchmarking:

- **`dr_methods/`** – Directory containing code for different dimensionality reduction methods.
- **`features/`** – Contains code for feature extraction and processing.
- **`io_utils/`** – Utility code for input/output operations.
- **`method_configs/`** – Configuration files for different benchmarking methods.
- **`optimization/`** – Code for optimization routines.
- **`scoring/`** – Contains code for scoring and evaluating methods.
- **`visualization/`** – Code for visualizing benchmarking results.



### Scripts

The `scripts` directory contains the master scripts for data preparation, running benchmarks, and analyzing results:

- **`run_optimization.py`** – Main script for running optimization processes.
- **`analyze_results.py`** – Script for automated result analysis.
- **`prepare_lolo.py`** – Script for splitting datasets in leave-one-library-out (LOLO) mode.

