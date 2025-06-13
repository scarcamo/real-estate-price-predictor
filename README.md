# ML Experiment Pipeline

This project contains a machine learning experiment pipeline that trains and evaluates multiple models on different feature sets using cross-validation and hyperparameter optimization.

## Prerequisites

### Install uv (Python Package Manager)

uv is a fast Python package installer and resolver. Install it using one of the following methods:

**On macOS and Linux:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**On Windows:**
```bash
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

**Alternative installation methods:**
```bash
# Using pip
pip install uv

# Using homebrew (macOS)
brew install uv

# Using conda
conda install -c conda-forge uv
```

## Setup

### 1. Clone and Navigate to Project
```bash
git clone https://github.com/scarcamo/real-estate-price-predictor
cd real-estate-price-predictor
```

### 2. Create Virtual Environment and Install Dependencies
```bash
# Create virtual environment with python 3.12+
uv venv --python 3.12

# Activate virtual environment
# On macOS/Linux:
source .venv/bin/activate
# On Windows:
.venv\Scripts\activate

# Install dependencies
uv sync
```


### 3. Setup MLflow Tracking
```bash
# Start MLflow tracking server
mlflow ui 
```

## Running Experiments

### Main Experiment Pipeline

Run the complete experiment pipeline using:

```bash
# Using uv run (recommended)
uv run src/run_experiment.py

# Or if environment is activated
python src/run_experiment.py
```

This script will:
- Load configuration from config files
- Run experiments for each feature set defined in the config
- Train multiple models (LightGBM, XGBoost, XGBoostQuantile, RandomForest)
- Perform cross-validation and hyperparameter optimization with Optuna
- Log all results, parameters, and metrics to MLflow

### Configuration

The experiment can be configured through:
- Configuration files in the config directory
- Feature sets in the `feature_sets` directory
- Model parameters in the model factory

Key configuration parameters:
- `target_variable`: The target variable to predict
- `models_to_run`: List of models to train
- `feature_sets`: List of feature set files to use
- `cv_folds`: Number of cross-validation folds
- `optuna.n_trials`: Number of hyperparameter optimization trials

## Viewing Results

### MLflow UI
After running experiments, view results in the MLflow UI:
```bash
mlflow ui
```
Navigate to http://127.0.0.1:5000 to view:
- Experiment runs
- Model parameters
- Metrics and artifacts
- Model comparisons

## Creating Experiment Summary

### Generate Summary Report

After running experiments, create a comprehensive summary of the best performing models:

```bash
# Using uv run (recommended)
uv run make_summary.py

# Or if environment is activated
python make_summary.py
```

### What the Summary Script Does

The `make_summary.py` script provides comprehensive analysis by:

1. **Querying MLflow**: Connects to your MLflow tracking server and retrieves all experiments and runs
2. **Finding Best Runs**: Identifies the best performing run for each model-feature set combination
3. **Extracting Metrics**: Collects all relevant metrics including:
   - Train/test scores (R², MAE, MSE, RMSE)
   - Cross-validation results
   - Hyperparameter optimization details
   - Feature set information
4. **Intelligent Metric Selection**: Automatically detects the best available optimization metric (test_score, cv_score, test_r2, etc.)
5. **Comprehensive Analysis**: Groups results by model type and feature set for comparison

### Summary Output Files

The script generates multiple output formats in the `experiment_summaries/` directory:

1. **`experiment_summary_YYYYMMDD_HHMMSS.csv`**: 
   - Tabular format with best runs
   - Includes model names, feature sets, all metrics
   - Sorted by performance (best first)

2. **`best_models_report_YYYYMMDD_HHMMSS.json`**: 
   - Detailed JSON report with metadata
   - Complete run information including parameters
   - Summary statistics and experiment overview


## Workflow

### Complete Workflow

Here's the complete workflow from setup to analysis:

```bash
# 1. Setup environment
uv venv --python 3.12
source .venv/bin/activate
uv sync

# 2. Run experiments
uv run src/run_experiment.py

# 3. Start MLflow UI (optional, in separate terminal)
mlflow ui

# 4. Generate summary after experiments complete
uv run make_summary.py

# 5. View results
# - CSV file for spreadsheet analysis
# - jupyter noteboook summary.ipynb
```

