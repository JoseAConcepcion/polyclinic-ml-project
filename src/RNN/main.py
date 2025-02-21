import pandas as pd
import numpy as np
import sys
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import torch
import os
import logging as LOG
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
from scipy.stats import ttest_rel

log_directory = os.path.join(os.path.dirname(__file__), "logs")

LOG.basicConfig(
    level=LOG.INFO,  
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        LOG.FileHandler(os.path.join(log_directory, "rnn_logs.txt"), mode="w")  # Guarda logs en un archivo
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
    X, y = [], []
    for i in range(len(data) - timesteps):
        X.append(data[i:i+timesteps])
        y.append(data[i+timesteps])
    return np.array(X), np.array(y)


def RNN(df, timesteps, target_column, test_size=0.3, random_state=42, 
        rnn_input_size=1, rnn_hidden_size=16, rnn_output_size=1, epochs=200):
    
    scaler = MinMaxScaler()
    df[target_column] = scaler.fit_transform(df[[target_column]])
    data = df[target_column].values
    X, Y = create_sequences(data, timesteps)
    
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=random_state)
    
    X_train = torch.tensor(X_train, dtype=torch.float32).unsqueeze(-1)
    X_test = torch.tensor(X_test, dtype=torch.float32).unsqueeze(-1)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)
    
    LOG.info(f"Cantidad de datos a entrenar: {X_train.shape}")
    LOG.info(f"Cantidad de datos a testear: {X_test.shape}")
    
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
    
    return y_test_inv, y_pred_inv, test_loss.item()


def metrics(y_test_inv, y_pred_inv):
    mse = mean_squared_error(y_test_inv, y_pred_inv)
    r2 = r2_score(y_test_inv, y_pred_inv)
    LOG.info(f"MSE: {mse:.6f}")
    LOG.info(f"R2 Score: {r2:.6f}")
    return mse, r2

def main():
    path = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                        "../../data/Anexos_7/full/total_incomes_augmented_full_data.csv"))
    df = pd.read_csv(path)
    
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values(by='Date')
    df.set_index('Date', inplace=True)
    
    torch.manual_seed(42)
    np.random.seed(42)


    num_runs = 30
    errors = []

    for i in range(num_runs):
        LOG.info(f"Ejecutando la RNN - Iteración {i + 1}")
        _, _, mse = RNN(df, timesteps=10, target_column='Pinar del Rio')
        errors.append(mse)
        
    
    base_error = errors[0]  # Compare with first iteration
    t_stat, p_value = ttest_rel(errors[1:], [base_error] * (num_runs - 1))

    LOG.info(f"T-statistic: {t_stat:.6f}")
    LOG.info(f"P-value: {p_value:.6f}")

    if p_value < 0.05:
        LOG.info("Los resultados de la RNN varían significativamente entre ejecuciones (p < 0.05).")
    else:
        LOG.info("No hay suficiente evidencia para decir que los resultados de la RNN cambian significativamente.")


if __name__ == "__main__":
    main()
