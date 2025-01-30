import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

import os

import matplotlib.pyplot as plt

path = os.path.abspath(os.path.join(os.path.dirname(__file__),"../../data/Anexos_7/training/total_hospitalized_augmented_training_data.csv"))

df = pd.read_csv(path)

df['Date'] = pd.to_datetime(df['Date'])

df = df.sort_values(by='Date')

provinces_df=[]
for column in [column for column in df.columns if column!='Date']:
    
    scaler = MinMaxScaler()
    df[column] = scaler.fit_transform(df[[column]])

    def create_sequences(data, timesteps):
        X, y = [], []
        for i in range(len(data) - timesteps):
            X.append(data[i:i+timesteps])
            y.append(data[i+timesteps])
        return np.array(X), np.array(y)

    timesteps = 7  # Tamaño de las secuencias, cuántos días hacia atrás se usarán para predecir el siguiente
    # esto es un hiperparámetro que se puede ajustar, creo que tiene problemas por los datos no periodicos


    X, y = create_sequences(df[column].values, timesteps)


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

    feature_columns = [f'feature_{i+1}' for i in range(X.shape[1])]  # Create column names
    df_X = pd.DataFrame(X, columns=feature_columns)
    df_X['target'] = y
    df_X['date']=df['Date']
    
    provinces_df.append(df_X)
