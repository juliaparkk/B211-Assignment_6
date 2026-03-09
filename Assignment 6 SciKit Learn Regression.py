import os
import numpy as np
import pandas as pd
from sklearn import datasets, model_selection, metrics, linear_model, ensemble, svm

class DiabetesRegressionRunner:
    """
    Runs multiple regression models on the Scikit-Learn diabetes dataset.
    """

    def __init__(self, test_size=0.2, random_state=42):
        self.test_size = test_size
        self.random_state = random_state

        # Load dataset
        diabetes = datasets.load_diabetes()
        self.X = diabetes.data
        self.y = diabetes.target
        self.feature_names = diabetes.feature_names

        # Train/test split (lecture uses train_size=0.8)
        self.X_train, self.X_test, self.y_train, self.y_test = model_selection.train_test_split(
            self.X, self.y, train_size=0.8, random_state=self.random_state
        )

        # Store results
        self.results = {}

    def run_models(self):
        models = {
            # LinearRegression variations
            "LinearRegression": linear_model.LinearRegression(),
            "Ridge_alpha0.1": linear_model.Ridge(alpha=0.1),
            "Ridge_alpha1.0": linear_model.Ridge(alpha=1.0),
            "Ridge_alpha10": linear_model.Ridge(alpha=10.0),
            
            # RandomForestRegressor variations
            "RF_n100_d10": ensemble.RandomForestRegressor(
                n_estimators=100, max_depth=10, random_state=self.random_state
            ),
            "RF_n200_d20": ensemble.RandomForestRegressor(
                n_estimators=200, max_depth=20, random_state=self.random_state
            ),
            "RF_n300_dNone": ensemble.RandomForestRegressor(
                n_estimators=300, max_depth=None, random_state=self.random_state
            ),
            "RF_n500_d30": ensemble.RandomForestRegressor(
                n_estimators=500, max_depth=30, random_state=self.random_state
            ),
            
            # SVR variations
            "SVR_C1_g0.01": svm.SVR(kernel="rbf", C=1, gamma=0.01),
            "SVR_C10_g0.1": svm.SVR(kernel="rbf", C=10, gamma=0.1),
            "SVR_C100_g0.1": svm.SVR(kernel="rbf", C=100, gamma=0.1),
            "SVR_linear": svm.SVR(kernel="linear", C=10)
        }

        for name, model in models.items():
            model.fit(self.X_train, self.y_train)
            preds = model.predict(self.X_test)

            # Collect metrics from lecture
            self.results[name] = {
                "Explained Variance": metrics.explained_variance_score(self.y_test, preds),
                "Max Error": metrics.max_error(self.y_test, preds),
                "MAE": metrics.mean_absolute_error(self.y_test, preds),
                "MSE": metrics.mean_squared_error(self.y_test, preds),
                "MSLE": metrics.mean_squared_log_error(self.y_test, np.maximum(preds, 0)),
                "Median AE": metrics.median_absolute_error(self.y_test, preds),
                "MAPE": metrics.mean_absolute_percentage_error(self.y_test, preds),
                "R2 Score": metrics.r2_score(self.y_test, preds)
            }

        return self.results

    def save_metrics_to_csv(self, filename="model_metrics.csv"):
        """
        Saves the model metrics to a CSV file using dynamic pathing
        with os.path.join and os.path.dirname(__file__).
        """

        # Directory where this script is located
        dirname = os.path.dirname(__file__)

        # Create a results folder
        results_dir = os.path.join(dirname, "results")
        os.makedirs(results_dir, exist_ok=True)

        # Convert results dictionary to DataFrame
        df = pd.DataFrame(self.results).T  # models as rows

        # Build full file path
        filepath = os.path.join(results_dir, filename)

        # Save CSV
        df.to_csv(filepath, index=True)

        print(f"Metrics saved to: {filepath}")

    def print_results(self):
        for model, metrics_dict in self.results.items():
            print(f"\n===== {model} =====")
            for metric_name, value in metrics_dict.items():
                print(f"{metric_name}: {value:.4f}")

if __name__ == "__main__":
    runner = DiabetesRegressionRunner()
    runner.run_models()
    runner.print_results()
    runner.save_metrics_to_csv()
