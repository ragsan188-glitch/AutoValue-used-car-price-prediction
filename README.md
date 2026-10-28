# AUTOVALUE-Used Car Price Prediction

## Overview

This project develops and evaluates machine learning regression models to estimate the selling price of used cars. It combines exploratory data analysis, data preprocessing, model training, performance evaluation, and Tableau visualization.

The objective is to understand the factors associated with used car prices and compare different regression algorithms to identify a suitable predictive model.

The project uses Python for data analysis and machine learning and Tableau for dashboard-based visualization.

## Objectives

- Explore and understand used car market data.
- Identify relationships between vehicle characteristics and selling prices.
- Clean and prepare data for machine learning.
- Train and compare multiple regression models.
- Evaluate model performance using appropriate regression metrics.
- Visualize relevant findings using Tableau.
- Identify limitations and potential improvements for future development.

## Project Files

| File | Description |
|---|---|
| `used_car_price_predictor.ipynb` | Jupyter notebook containing data analysis, preprocessing, model development, and evaluation. |
| `dashboard.twbx` | Tableau packaged workbook containing the project dashboard. |

## Technology Stack

- **Python** — Data analysis and model development.
- **Pandas and NumPy** — Data manipulation and numerical operations.
- **Matplotlib and Seaborn** — Exploratory data visualization.
- **Scikit-learn** — Regression models, model evaluation, and hyperparameter tuning.
- **TensorFlow/Keras** — Neural network experiments.
- **Jupyter Notebook** — Interactive development environment.
- **Tableau** — Dashboard creation and data visualization.
- **Git and GitHub** — Version control and project documentation.

## Machine Learning Workflow

### 1. Exploratory Data Analysis

The dataset contains 46,405 vehicle records. Exploratory analysis is used to examine the dataset structure, inspect variables, understand price distributions, and investigate relationships between vehicle attributes and prices.

### 2. Data Preprocessing

Data preprocessing prepares the dataset for model training. Depending on the feature and model requirements, this includes handling missing values, selecting relevant variables, encoding categorical features, and preparing training and testing datasets.

### 3. Model Development

The following regression approaches were evaluated:

- Linear Regression
- Decision Tree Regressor
- Tuned Decision Tree Regressor
- Random Forest Regressor
- Neural network experiments

### 4. Model Evaluation

Model performance is assessed using regression evaluation metrics, including Mean Absolute Error (MAE) and the coefficient of determination (R²).

MAE measures the average absolute difference between predicted and actual prices. R² measures how much of the variation in the target variable is explained by the model on the evaluated dataset.

## Model Performance

The following results were recorded in the notebook's model evaluation.

| Model | Test MAE | Test R² |
|---|---:|---:|
| Linear Regression | €2,841.90 | 0.8869 |
| Decision Tree Regressor | €2,228.06 | 0.9028 |
| Tuned Decision Tree Regressor | €2,200.40 | 0.9046 |
| Random Forest Regressor | €2,077.02 | 0.9149 |

*Note: These are recorded experimental results. They should be reproduced and verified before being treated as final benchmark results.*

## Key Findings and Insights

### Random Forest achieved the strongest recorded performance

Among the models listed above, Random Forest achieved the lowest test MAE and highest test R².

Its MAE of approximately €2,077 indicates that the average absolute prediction error on the evaluated test set was around €2,077. This is an average, not a guarantee that an individual prediction will fall within that amount of the actual selling price.

### Tree-based models outperformed Linear Regression

The Decision Tree and Random Forest models recorded lower MAE values than Linear Regression. This suggests that the tree-based approaches captured useful relationships in the dataset that the linear model did not represent as effectively.

### Hyperparameter tuning produced a modest improvement

The tuned Decision Tree reduced MAE from €2,228.06 to €2,200.40. The improvement is measurable but relatively small, so further tuning and cross-validation may be useful.

### Model complexity requires careful evaluation

A model can perform well on training data without generalizing equally well to unseen vehicles. Overfitting, data leakage, feature selection, and the reliability of the train-test split must be considered when interpreting model performance.

## Tableau Dashboard

The project includes a Tableau packaged workbook:

`dashboard.twbx`

The workbook can be opened using Tableau Desktop or a compatible Tableau application.

The dashboard is intended to complement the machine learning analysis by providing a visual way to explore the data. Its exact charts and findings should be interpreted from the workbook itself.

## How to Run the Project

### Prerequisites

- Python 3
- Jupyter Notebook or JupyterLab
- The dataset used by the notebook
- Tableau Desktop or a compatible application to open the dashboard

### Installation

Install the core Python libraries using pip:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

If the notebook contains TensorFlow/Keras experiments, install a compatible TensorFlow version separately if required by your environment.

### Run the Notebook

1. Clone or download this repository.
2. Obtain the dataset from its original source, subject to its licensing and redistribution terms.
3. Place the dataset in the location expected by the notebook, or update the data-loading path.
4. Start Jupyter Notebook:

```bash
jupyter notebook
```

5. Open `used_car_price_predictor.ipynb`.
6. Execute the cells in order and inspect the outputs.

The notebook may require adjustments to file paths or library versions depending on the local environment. The dataset is not included in this repository at present.

## Limitations

- The reported performance metrics are experimental and have not been independently reproduced here.
- The recorded Random Forest configuration uses only three trees and requires further investigation and tuning.
- Hyperparameter search scoring and metric labels should be checked for consistency before the results are considered final.
- Neural network performance depends on the architecture, preprocessing, training configuration, and evaluation procedure.
- Prediction accuracy may vary across vehicle categories, price ranges, and data distributions.
- Dataset quality, missing values, outliers, and possible data leakage can affect results.
- The model has not been deployed as a public prediction service.
- Predictions are estimates and should not be treated as guaranteed market prices.

## Future Improvements

- Build a reproducible preprocessing and prediction pipeline.
- Perform cross-validation and more systematic hyperparameter tuning.
- Compare additional ensemble methods, including gradient boosting.
- Analyze residuals and prediction errors across different price ranges.
- Investigate feature importance and model interpretability.
- Save the trained model and preprocessing pipeline for inference.
- Build a web application that accepts vehicle details and returns a predicted price.
- Integrate the prediction application with a documented deployment workflow.
- Improve testing, dependency management, and experiment tracking.

## Disclaimer

This project is intended for educational and portfolio purposes. Predictions depend on the dataset, selected features, and training procedure. They should not be interpreted as professional vehicle valuations or guaranteed selling prices.

## Author
RAGUL Y S

(Developed as a machine learning and data analytics portfolio project.)
