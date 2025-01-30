import h2o
import pandas as pd
from h2o.automl import H2OAutoML
import os

class H2OForecaster:
    def __init__(self, data, time_col='date', target_col='target'):
        """
        Initialize the H2OForecaster class.

        :param data: Pandas DataFrame containing the dataset.
        :param time_col: Name of the time column (default: 'date').
        :param target_col: Name of the target column (default: 'target').
        """
        self.data = data
        self.time_col = time_col
        self.target_col = target_col
        self.train_data = None
        self.test_data = None
        self.aml = None
        self.best_model = None
        self.predictions = None
        self.province=None

    def initialize_h2o(self):
        """Initialize the H2O cluster."""
        h2o.init()

    def preprocess_data(self):
        """
        Preprocess the data:
        1. Sort by the time column.
        2. Split into training and testing sets.
        """
        # Sort by the time column
        self.data = self.data.sort_values(by=self.time_col, ascending=True)

        # Split into training (80%) and testing (20%) sets
        train_size = int(len(self.data) * 0.8)
        self.train_data = self.data.iloc[:train_size]
        self.test_data = self.data.iloc[train_size:]

        print("Training Data:")
        print(self.train_data)

    def train_model(self):
        """
        Train an H2O AutoML model.
        """
        # Convert pandas DataFrames to H2O Frames
        h2o_df = h2o.H2OFrame(self.train_data)
        h2o_test = h2o.H2OFrame(self.test_data)

        # Define predictors and response
        response = self.target_col
        predictors = [col for col in h2o_df.columns if col != self.time_col and col != self.target_col]

        # Run H2O AutoML
        self.aml = H2OAutoML(max_models=10, seed=1,exclude_algos=["StackedEnsemble"])
        self.aml.train(y=response, x=predictors, training_frame=h2o_df)

        # Display the leaderboard
        lb = self.aml.leaderboard
        print(lb)
        lb_df = lb.as_data_frame()

        # Save the leaderboard DataFrame to a CSV file
        
        lb_df.to_csv(f"src/AutoML/leaderboard/{self.province}_leaderboard.csv", index=False)
        # Get the best model
        self.best_model = self.aml.leader

        # Make predictions
        self.predictions = self.best_model.predict(h2o_test)
        
        self.predictions= self.predictions.as_data_frame()

        
    def shutdown_h2o(self):
        """Shutdown the H2O cluster."""
        h2o.shutdown()

    def run(self):
        """Run the entire workflow."""
        self.initialize_h2o()
        self.preprocess_data()
        self.train_model()
        self.shutdown_h2o()