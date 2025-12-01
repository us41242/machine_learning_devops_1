# NYC Airbnb Price Prediction - ML DevOps Pipeline

This project implements an end-to-end Machine Learning pipeline to predict Airbnb rental prices in New York City. It demonstrates MLOps best practices using **MLflow**, **Weights & Biases**, and **Hydra** for orchestration, tracking, and configuration management.

The pipeline is designed to be reproducible, modular, and robust against data quality issues.

## Project Overview

The goal of this project is to build a scalable and reproducible ML workflow that:
1.  Downloads and cleans raw data.
2.  Validates data quality using automated tests.
3.  Splits data into training, validation, and test sets.
4.  Trains a Random Forest model with feature engineering.
5.  Optimizes hyperparameters using Hydra.
6.  Verifies the final "production" model against a held-out test set.

## Architecture & Technologies

* **Language:** Python 3.10
* **Orchestration:** [Hydra](https://hydra.cc/) (Configuration & Multi-run)
* **Tracking:** [Weights & Biases (W&B)](https://wandb.ai/) (Artifacts & Experiment Tracking)
* **Workflow:** [MLflow](https://mlflow.org/) (Project packaging & Component execution)
* **Libraries:** Scikit-learn, Pandas, Pytest

## Pipeline Components

The pipeline consists of the following modular steps defined in `main.py`:

1.  **`download`**: Fetches the latest dataset from the source.
2.  **`basic_cleaning`**:
    * Removes outliers based on price thresholds.
    * **Feature:** Implements geospatial filtering (Lat/Lon) to remove invalid listings outside NYC boundaries.
3.  **`data_check`**:
    * Validates data schema and quality.
    * **Feature:** Custom `pytest` tests ensuring price ranges (10-350) and sufficient row counts.
4.  **`data_split`**: Segregates data into training/validation and test sets.
5.  **`train_random_forest`**:
    * Trains a Random Forest Regressor.
    * Pipeline includes `OneHotEncoder` for categorical variables and `SimpleImputer` for missing values.
    * Logs performance metrics (MAE, R2) to W&B.
6.  **`test_regression_model`**:
    * Loads the model tagged as `prod` from W&B.
    * Verifies performance against the unseen `test_data.csv`.

## Installation

1.  **Clone the repository:**
    ```bash
    git clone <your-repo-url>
    cd machine_learning_devops_1
    ```

2.  **Ensure Conda is installed.**

3.  **Configure Weights & Biases:**
    Ensure you are logged in to W&B in your terminal:
    ```bash
    wandb login
    ```

## Usage

### 1. Run the Entire Pipeline (Default)
    python main.py

This will trigger 4 distinct training runs. You can visualize the results in the W&B dashboard.

3. Release Process
Development: Code changes are tested locally.

Releases: Valid pipelines are tagged on GitHub (e.g., v1.0.1).

Production: The best performing model from the hyperparameter sweep is manually tagged as prod in the W&B dashboard, which triggers the final test_regression_model verification step.

Recent Updates (v1.0.1)
Fixed Data Corruption Issue: Added a geospatial filter in the basic_cleaning step to handle corrupted data (like sample2.csv) that contained coordinates outside of New York City.

Environment Standardization: Localized the test_regression_model component to enforce Python 3.10 compatibility, preventing pickle serialization errors between training and testing environments.