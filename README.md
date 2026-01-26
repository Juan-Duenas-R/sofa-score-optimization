# Data-Driven Recalibration of the SOFA Score Using Interpretable Machine Learning
[![Python 3.8+](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

This repository contains the implementation of an optimized SOFA (Sequential Organ Failure Assessment) score using integer weights and automatic bin selection. The methodology is designed to create clinically interpretable scoring systems while maintaining high predictive performance. The data (`data\synthetic_data.csv`) is not real, therefore, metrics do not show real performance.
## Key Features

- **Automatic bin selection** per feature using decision trees
- **Integer weight optimization** for clinical interpretability
- **Scale-invariant optimization** with targeted score standard deviation
- **Grid search** with cross-validation for hyperparameter tuning
- **Positive coefficient enforcement** for monotonic risk relationships

## Repository Structure

```
.
├── README.md
├── requirements.txt
├── LICENSE
├── docs
│   └── index.html
├── data/
│   └── synthetic_data.csv
├── models/
├── src/
│   ├── train.py
│   ├── evaluate.py
│   └── utils/
│       ├── binning.py
│       ├── optimization.py
│       └── scoring.py
└── results/
```

## Installation

### Prerequisites

- Python 3.10 or higher
- pip package manager

### Setup

1. Clone the repository:
```
git clone https://github.com/Juan-Duenas-R/sofa-score-optimization.git
cd sofa-score-optimization
```

2. Create a virtual environment (recommended):
```
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```
pip install -r requirements.txt
```
## Data
`synthetic_data.csv` contains mock patient records.
This file is for demonstration only. To apply the pipeline to real eICU data,
users must structure their data with identical column names and types.

## Usage

### Training a Model

```python
from src.train import train_sofa_model

# Train with default parameters
results = train_sofa_model(
    data_path='data/synthetic_data.csv',
    output_dir='models/'
)
```

### Evaluating a Model

```python
from src.evaluate import evaluate_model

# Load and evaluate
metrics = evaluate_model(
    model_path='models/sofa_model.pkl',
    test_data_path='data/synthetic_data.csv'
)
```

### Command Line Interface

```
# Train model
python -m src.train --data data/synthetic_data.parquet --output models/

# Evaluate model
python -m src.evaluate --model models/sofa_model.pkl --data data/test_data.parquet
```

## Methodology

### 1. Automatic Bin Selection

Each feature is discretized using decision tree-based binning with automatic selection of the optimal number of bins (k) per feature. The selection minimizes validation log-loss with a complexity penalty.

### 2. Integer Weight Optimization

The optimization process consists of three stages:

1. **Continuous optimization**: Initial weight optimization using L-BFGS-B
2. **Scale correction**: Weights are rescaled to achieve target score standard deviation
3. **Discrete refinement**: Local search in integer space using coordinate descent

### 3. Score Calibration

The final score is calibrated using logistic regression with:
- Standardized score as input
- L2 regularization
- Class-balanced sample weights
- Positive coefficient enforcement

## Configuration

Key hyperparameters in `config.taml`. For example.
Grid search parameters:
- `min_samples_leaf`: [15]
- `reg_lambda`: [0.01, 0.1, 1, 10, 100, 1000]
- `W_bound`: [100]

## Results

Detailed results are saved in `results/grid_search_results.csv` after training



## Web Calculator
A user-friendly web calculator is available for clinical implementation of the recalibrated SOFA score.

### Access the Calculator
**Live Demo**: https://juan-duenas-r.github.io/sofa-score-optimization/
### Features
- **Real-time calculation** of the recalibrated SOFA score
- **Mortality probability estimation** using the validated logistic regression model
- **Color-coded risk stratification**:
  - 🟢 Low Risk (< 20%)
  - 🟡 Moderate Risk (20-40%)
  - 🟠 High Risk (40-60%)
  - 🔴 Very High Risk (> 60%)
- **Responsive design** for desktop and mobile devices
- **No data storage** - all calculations performed locally in the browser
### Clinical Variables

The calculator implements the following variables based on the data-driven thresholds and weights described in the paper:

- PaO₂/FiO₂ Ratio (Respiratory function)
- Glasgow Coma Scale (Neurological function)
- Mean Blood Pressure (Cardiovascular function)
- Platelets (Coagulation)
- Creatinine (Renal function)
- Bilirubin (Hepatic function)
- Vasopressor support (Norepinephrine, Epinephrine, Dopamine, Dobutamine)


All thresholds and weights are implemented as specified in Table 2 of the manuscript.

### Disclaimer 
This calculator is intended for research and educational purposes. Clinical decisions should be made by qualified healthcare professionals considering all available patient information. The calculator provides probability estimates based on the eICU-CRD dataset and may require local validation before clinical implementation.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


## Contact

- **Author**: Juan Dueñas-Ruiz
- **Email**: juan.duenas@uva.es
- **Institution**: Universidad de Valladolid

[![Web Calculator](https://img.shields.io/badge/Web-Calculator-blue?style=for-the-badge&logo=html5)](https://juan-duenas-r.github.io/sofa-score-optimization/)
[![DOI](https://img.shields.io/badge/DOI-pending-orange?style=for-the-badge)](-)

[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)
