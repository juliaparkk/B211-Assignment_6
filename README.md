# B211-Assignment_6

# Best Performing Model:
Based on the metrics in model_metrics.csv, the best overall model is Ridge Regression with α = 0.1. It achieves the strongest balance of predictive accuracy and error minimization across the regression metrics utilized. Specifically, it has the highest R² score (0.4609) and the highest explained variance (0.4629) among all models, indicating that it captures more variance in diabetes progression than the other regressors. Its MSE (2856.49) is the lowest among all models, and its MSLE (0.1684) is also the best, indicating that its predictions remain proportionally close to the true values. While Linear Regression performs similarly, Ridge α=0.1 consistently edges it out across nearly every metric. Random Forest models show moderate performance but higher error values, and all SVR variants perform substantially worse, with low explained variance and high MAE/MSE. Therefore, Ridge Regression (α = 0.1) is the most effective model.

# Project Purpose
Diabetes is a chronic condition in which the body struggles to regulate blood glucose levels. Predicting the progression of diabetes can help clinicians identify high‑risk patients earlier and improve treatment planning. This project uses the built‑in Scikit‑Learn diabetes dataset to train and evaluate multiple machine learning regression models. The goal is to compare model performance using the regression metrics introduced in Information Infrastructure II — Lecture 10, including explained variance, max error, MAE, MSE, MSLE, median absolute error, MAPE, and R² score.

The project is implemented using a single class, DiabetesRegressionRunner, which encapsulates the entire machine learning workflow from data loading to model evaluation and CSV export.

# Class Design
The class is designed to be modular, readable, and easy to extend. It follows a structured machine‑learning pipeline:
1. Load dataset
2. Split into training and testing sets
3. Train multiple regression models
4. Generate predictions
5. Evaluate each model using lecture‑defined metrics
6. Store results
7. Export results to a CSV file using dynamic pathing
This object‑oriented design keeps all components organized and allows new models or metrics to be added easily.

# Class Attributes
Attribute: Description
test_size: Proportion of data used for testing (default 0.2).
random_state: Ensures reproducible results across runs.
X, y: Feature matrix and target vector from the diabetes dataset.
feature_names: Names of the dataset’s features.
X_train, X_test, y_train, y_test: Training and testing splits using train_size=0.8 as taught in lecture.
results: A dictionary storing all evaluation metrics for each model.

# Class Methods
__init__(...)
- Loads the diabetes dataset from Scikit‑Learn.
- Stores feature data, target values, and feature names.
- Splits the dataset into training and testing sets using the standard 80/20 split.
- Initializes an empty results dictionary.

run_models()
- Trains a diverse set of regression models, including:
Linear Models
- Linear Regression
- Ridge Regression (α = 0.1, 1.0, 10)
Random Forest Regressors
- 100 trees, depth 10
- 200 trees, depth 20
- 300 trees, unlimited depth
- 500 trees, depth 30
Support Vector Regressors
- RBF kernel with different C and gamma values
- Linear kernel variant
For each model, the method:
- Fits the model on the training data
- Generates predictions
- Computes all regression metrics:
  Explained Variance
  Max Error
  Mean Absolute Error (MAE)
  Mean Squared Error (MSE)
  Mean Squared Log Error (MSLE)
  Median Absolute Error
  Mean Absolute Percentage Error (MAPE)
  R² Score

All metrics are stored in self.results.
save_metrics_to_csv(filename="model_metrics.csv")
- Uses dynamic pathing with os.path.join() and os.path.dirname(__file__) to ensure portability.
- Automatically creates a /results folder if it does not exist.
- Converts the results dictionary into a Pandas DataFrame.
- Saves the metrics to a CSV file inside the results directory.
This ensures the project works on any machine without hard‑coded file paths.

print_results()
Prints all model metrics in a readable format for quick inspection.

# How to Run the Program
The script includes a __main__ block:

if __name__ == "__main__":
    runner = DiabetesRegressionRunner()
    runner.run_models()
    runner.print_results()
    runner.save_metrics_to_csv()

Running the file will:
1. Train all models
2. Print their metrics
3. Save the results to results/model_metrics.csv

Limitations
- No hyperparameter tuning: Models use manually chosen parameters rather than optimized ones. GridSearchCV or RandomizedSearchCV could improve performance.
- Small dataset: The diabetes dataset contains only 442 samples, which limits model generalization.
- SVR sensitivity: SVR models often require feature scaling (e.g., StandardScaler) for optimal performance, which was not applied here.
- Random Forest variability: Performance depends heavily on depth and number of trees.
- Linear models assume linearity, which may not fully capture the complexity of diabetes progression.
