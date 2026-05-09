# California Housing Price Prediction — ML Model Comparison

A comprehensive comparison of classical machine learning and deep learning approaches for predicting California housing prices. This project goes beyond building a single model — it systematically evaluates how different architectures, regularization strategies, and hyperparameters affect prediction quality.

## Objective

Predict median house values across California districts using the [California Housing dataset](https://scikit-learn.org/stable/datasets/real_world.html#california-housing-dataset) from scikit-learn, and rigorously compare model performance using standard evaluation metrics.

## Dataset

| Property | Value |
|----------|-------|
| Source | `sklearn.datasets.fetch_california_housing` |
| Samples | 20,640 |
| Features | 8 |
| Target | Median house value (in $100,000s) |

**Features:** MedInc (median income), HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude

## Models Compared

### 1. Linear Regression (Baseline)
- Scikit-learn `LinearRegression`
- Establishes the baseline performance to beat

### 2. Neural Network — Shallow
- TensorFlow/Keras `Sequential` model
- 1 hidden layer
- Tests whether a simple neural network improves over linear regression

### 3. Neural Network — Deep
- Multiple hidden layers with increasing depth
- ReLU activation in hidden layers, linear output
- Tests the effect of added depth on prediction quality

### 4. Regularized Neural Network
- L2 regularization applied to dense layers
- Multiple λ values tested
- Shows how regularization affects generalization

## Evaluation Metrics

Each model is evaluated on both training and test sets using:

| Metric | What It Measures |
|--------|-----------------|
| **MSE** | Mean Squared Error — average squared prediction error |
| **RMSE** | Root MSE — error in the same units as house price |
| **MAE** | Mean Absolute Error — average absolute prediction error |
| **R²** | Coefficient of determination — variance explained by the model |

## Project Structure

```
california-housing-ml-comparison/
│
├── README.md
├── requirements.txt
│
├── notebooks/
│   ├── 01_eda.ipynb                    # Exploratory data analysis
│   ├── 02_linear_regression.ipynb      # Baseline model
│   ├── 03_neural_network.ipynb         # Neural network experiments
│   └── 04_comparison.ipynb             # Final comparison and results
│
├── src/
│   ├── data_preprocessing.py           # Scaling, train/test split
│   ├── evaluation.py                   # Metrics computation
│   └── visualization.py               # Plotting utilities
│
├── results/
│   ├── model_comparison.csv            # Metrics for all models
│   └── figures/                        # Training curves, comparison plots
│
└── report/
    └── project_report.md               # Detailed findings and analysis
```

## Key Findings

> _This section will be updated after experiments are complete._

- Linear regression baseline: R² = ___
- Best neural network: R² = ___
- Effect of regularization: ___
- Effect of depth: ___

## Setup

```bash
git clone https://github.com/yourusername/california-housing-ml-comparison.git
cd california-housing-ml-comparison
pip install -r requirements.txt
```

## Requirements

```
numpy
pandas
matplotlib
seaborn
scikit-learn
tensorflow
```

## What I Learned

- How evaluation metrics (MSE, RMSE, MAE, R²) reveal different aspects of model performance
- Why feature scaling is critical for neural networks but optional for linear regression
- How regularization prevents overfitting by penalizing large weights
- The tradeoff between model complexity (more layers/neurons) and generalization
- When a simple model is sufficient vs when added complexity helps

## Author

**Suman** — CS Major, Mathematics Minor at Texas State University

## License

MIT