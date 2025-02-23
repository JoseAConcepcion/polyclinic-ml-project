import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error,mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
color_pal = sns.color_palette()  
plt.style.use( 'fivethirtyeight' ) 

def split_data(df,n=0.8):
    '''
    Split the data set into training data set and testing data set.

    Args:
        - df (csv): CSV file with date set.
        - n (float): Size of training data set relative to the data set.
    
    Return:
        - train_data (csv): Training data set.
        - test_data (csv): Testing data set.
    '''

    train_size = int(len(df) * n)
    train_data = df.iloc[:train_size]
    test_data = df.iloc[train_size:]

    return train_data,test_data


def create_features(df, target_cols, lags=7,wind=7):
    '''
    Generate additional features for a time series dataset.

    Args:
        - df (csv): CSV file with data set.
        - target_cols (list): List of strings containing the target columns of the dataset
        - lags (int): Maximum lag.
        - wind (int): Window size for rolling window statistics.

    Return:
        - df (csv): Dataset with all new features added.
    '''

    df['year'] = df['Date'].dt.year
    df['month'] = df['Date'].dt.month
    df['day'] = df['Date'].dt.day
    df['dayofweek'] = df['Date'].dt.dayofweek
    df['dayofyear'] = df['Date'].dt.dayofyear
    df['weekofyear'] = df['Date'].dt.isocalendar().week.astype(int)
    df['quarter'] = df['Date'].dt.quarter

    
    for col in target_cols:
        for lag in range(1, lags + 1):
            df[f'{col}_lag_{lag}'] = df[col].shift(lag)
        df[f'{col}_rolling_mean_{wind}'] = df[col].rolling(window=wind).mean()
        df[f'{col}_rolling_std_{wind}'] = df[col].rolling(window=wind).std()

    
    return df.dropna()


def train_and_evaluate(train_data, test_data, target_col, features):
    '''
    Train a regression model and evaluate its performance.

    This function trains an XGBoost regression model using the provided training data,
    evaluates it on the test data, and calculates various error metrics. It also rounds
    the predicted values to the nearest integer.

    Args:
        train_data (pd.DataFrame): DataFrame containing the training dataset.
        test_data (pd.DataFrame): DataFrame containing the testing dataset.
        target_col (str): The name of the target column to predict.
        features (list of str): List of feature column names used for training.

    Returns:
        tuple: A tuple containing:
            - model (XGBRegressor): The trained XGBoost regression model.
            - y_pred (np.ndarray): Array of rounded predictions for the test set.
            - rmse (float): Root Mean Squared Error of the predictions.
            - mse (float): Mean Squared Error of the predictions.
            - mae (float): Mean Absolute Error of the predictions.
    '''
    X_train = train_data[features]
    y_train = train_data[target_col]
    X_test = test_data[features]
    y_test = test_data[target_col]

    model = XGBRegressor(
        objective='reg:squarederror', 
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=7,
        min_child_weight=3,
        gamma=0.1, 
        random_state=42)
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    y_pred = model.predict(X_test)
    y_pred = np.round(y_pred)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    return model, y_pred, rmse, mse, mae


def train_and_evaluate_multiple_targets(train_data, test_data, target_cols, date_col='Date'):
    """
    Train and evaluate models for multiple target columns.

    This function trains a regression model for each target column in the dataset, 
    using common features derived from the training dataset. The models are stored 
    in a dictionary, along with their respective predictions.

    Args:
        train_data (pd.DataFrame): DataFrame containing the training dataset.
        test_data (pd.DataFrame): DataFrame containing the testing dataset.
        target_cols (list of str): List of target column names to predict.
        date_col (str): Column name to exclude from features. Default is 'Date'.

    Returns:
        tuple: A tuple containing:
            - models (dict): A dictionary where keys are target column names, 
              and values are the trained models (XGBRegressor).
            - predictions (dict): A dictionary where keys are target column names, 
              and values are arrays of rounded predictions for the test set.
    """
    # Define common features excluding the date column and target columns
    features = [col for col in train_data.columns if col not in [date_col] + target_cols]

    # Initialize dictionaries for models and predictions
    models = {}
    predictions = {}

    # Train and evaluate a model for each target column
    for target_col in target_cols:
        print(f'Entrenando modelo para: {target_col}')
        model, y_pred, rmse, mse, mae = train_and_evaluate(train_data, test_data, target_col, features)
        print(f'{target_col} - RMSE: {rmse}, MSE: {mse}, MAE: {mae}')
        models[target_col] = model
        predictions[target_col] = y_pred

    return models, predictions


def plot_model_performance(test_data, target_cols, predictions, date_col='Date', linestyle_test='-',linestyle_predictions='--'):
    """
    Plot the model performance for multiple target columns.

    This function creates a plot for each target column, comparing the actual 
    values with the predicted values over time (based on the Date column).

    Args:
        test_data (pd.DataFrame): DataFrame containing the testing dataset, including the 'Date' column.
        target_cols (list of str): List of target column names to plot.
        predictions (dict): Dictionary where keys are target column names, 
                             and values are arrays of predicted values.
        date_col (str): The name of the column containing the date information. Default is 'Date'.

    Returns:
        None: The function generates and displays plots for each target column.
    """
    # Graficar los resultados para cada columna objetivo
    for target_col in target_cols:
        plt.figure(figsize=(10, 6))
        plt.plot(test_data[date_col], test_data[target_col], 'o', label=f'{target_col} - Valores reales', linestyle=linestyle_test)
        plt.plot(test_data[date_col], predictions[target_col], 'x', label=f'{target_col} - Predicciones', linestyle=linestyle_predictions)
        plt.xlabel('Fecha')
        plt.ylabel(target_col)
        plt.title(f'Rendimiento del modelo para {target_col}')
        plt.legend()
        plt.show()