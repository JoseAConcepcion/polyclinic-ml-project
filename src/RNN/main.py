import pandas as pd
import numpy as np
import sys
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, root_mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import torch
import os
import logging as LOG
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt

LOG.basicConfig(
    level=LOG.INFO,  
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        LOG.StreamHandler(sys.stdout)
    ]
)

class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleRNN, self).__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    def forward(self, x):
        out, _ = self.rnn(x)
        out = self.fc(out[:, -1, :])  
        return out
    
    
def create_sequences(data, timesteps):
    """ 
    data: Datos a transformar en secuencias
    timesteps: Tamaño de las secuencias, cuántos días se usarán para predecir el siguiente

    Returns:
    X: Datos de entrada (Estos son los datos que se usarán para predecir)
    y: Datos de salida (Estos son los datos que se quieren predecir)
    """
    X, y = [], []
    for i in range(len(data) - timesteps):
        X.append(data[i:i+timesteps])
        y.append(data[i+timesteps])
    return np.array(X), np.array(y)
    
  
def RNN(df, timesteps, target_column, test_size=0.1, random_state=42, rnn_input_size=1, rnn_hidden_size=16, rnn_output_size=1, epochs=200):
    """ 
    Args:
        - df: DataFrame con los datos
        - timesteps: Tamaño de las secuencias, cuántos días se usarán para predecir el siguiente
        - target_column: Columna a predecir
        - test_size: Porcentaje de datos a usar para test
        - random_state: Semilla para reproducibilidad
        - rnn_input_size: Tamaño de la entrada de la red recurrente
        - rnn_hidden_size: Tamaño de las capas ocultas de la red recurrente
        - rnn_output_size: Tamaño de la salida de la red recurrente
        - epochs: Cantidad de épocas para entrenar la red
    
    Returns:
        - y_test_inv: Datos reales
        - y_pred_inv: Datos predichos
    """
    scaler = MinMaxScaler()
    df[target_column] = scaler.fit_transform(df[[target_column]])

    data = df[target_column].values
    
    X, Y = create_sequences(data, timesteps)
    
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=random_state)
    
    X_train = torch.tensor(X_train, dtype=torch.float32).unsqueeze(-1)
    X_test = torch.tensor(X_test, dtype=torch.float32).unsqueeze(-1)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)
    
    LOG.info("Cantidad de datos a entrenar: %s", X_train.shape)
    LOG.info("Cantidad de datos a testear: %s", X_test.shape)
    
    model = SimpleRNN(rnn_input_size, rnn_hidden_size, rnn_output_size)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        outputs = model(X_train)
        loss = criterion(outputs.squeeze(), y_train)
        
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 20 == 0:
            LOG.info(f"Época {epoch + 1}/{epochs}, Pérdida: {loss.item():.4f}")
            
    model.eval()
    
    with torch.no_grad():
        y_pred = model(X_test).squeeze()
        test_loss = criterion(y_pred, y_test)
        LOG.info(f"Pérdida en el conjunto de prueba: {test_loss.item():.4f}")
        
    y_test_inv = scaler.inverse_transform(y_test.numpy().reshape(-1, 1))
    y_pred_inv = scaler.inverse_transform(y_pred.numpy().reshape(-1, 1))
    
    return y_test_inv, y_pred_inv
    
def plot_results(y_test_inv, y_pred_inv):
    plt.figure(figsize=(10, 6))
    plt.plot(y_test_inv, label="Real", marker='o')
    plt.plot(y_pred_inv, label="Predicción", marker='x')
    plt.legend()
    plt.title("Predicción vs Real")
    plt.show()
    
def metrics(y_test_inv, y_pred_inv):
    mse = mean_squared_error(y_test_inv, y_pred_inv)
    rmse = root_mean_squared_error(y_test_inv, y_pred_inv)
    r2 = r2_score(y_test_inv, y_pred_inv)
    
    LOG.info("Error cuadrático medio: %s", mse)
    LOG.info("RMSE: %s", rmse)
    LOG.info("R2 Score: %s", r2)
    
    return mse, rmse, r2


def main():
    path = os.path.abspath(os.path.join(os.path.dirname(__file__),"training_data.csv"))
    df = pd.read_csv(path)

    df['Date'] = pd.to_datetime(df['Date'])

    df = df.sort_values(by='Date')

    df.set_index('Date', inplace=True)
    
    y_test_inv, y_pred_inv = RNN(df, timesteps=10, target_column='Pinar del Rio')
    
    plot_results(y_test_inv, y_pred_inv)
    
    metrics(y_test_inv, y_pred_inv)
    

if __name__ == "__main__":
    main()