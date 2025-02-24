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
from scipy.stats import ttest_rel, ttest_1samp

LOG.basicConfig(
    level=LOG.INFO,  
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        LOG.StreamHandler(sys.stdout)
    ]
)

class DataNoiser:
    def __init__(self, noise_factor=0.1):
        self.noise_factor = noise_factor
    def add_noise(self, data):
        noise = torch.randn_like(data) * self.noise_factor
        return data + noise

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
    
  
def RNN(df, timesteps, target_column, test_size=0.3, random_state=42, rnn_input_size=1, rnn_hidden_size=16, rnn_output_size=1, epochs=200, logs=False, return_model=False):
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
    new_df = df.copy()
    new_df[target_column] = scaler.fit_transform(new_df[[target_column]])

    data = new_df[target_column].values
    
    X, Y = create_sequences(data, timesteps)
    
    # X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=random_state)
    split_idx = int(len(X) * (1 - test_size))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = Y[:split_idx], Y[split_idx:]
    
    test_dates = df.index[timesteps + split_idx +1 : timesteps + split_idx + len(X_test)]
    
    X_train = torch.tensor(X_train, dtype=torch.float32).unsqueeze(-1)
    X_test = torch.tensor(X_test, dtype=torch.float32).unsqueeze(-1)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)
    
    if logs:
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
        
        if (epoch + 1) % 20 == 0 and logs:
            LOG.info(f"Época {epoch + 1}/{epochs}, Pérdida: {loss.item():.4f}")
            

    # torch.save(model.state_dict(), './data/results/model_weights.pth')
    
    model.eval()
    
    with torch.no_grad():
        y_pred = model(X_test).squeeze()
        test_loss = criterion(y_pred, y_test)
        if logs:
            LOG.info(f"Pérdida en el conjunto de prueba: {test_loss.item():.4f}")
        
    y_test_inv = scaler.inverse_transform(y_test.numpy().reshape(-1, 1))
    y_pred_inv = scaler.inverse_transform(y_pred.numpy().reshape(-1, 1))
    
    y_test_inv = y_test_inv[:-1]
    y_pred_inv = y_pred_inv[1:]
    
    if return_model:
        return y_test_inv, y_pred_inv, test_dates, test_loss.item(), model, scaler, X_test, y_test
    else:
        return y_test_inv, y_pred_inv, test_dates, test_loss.item()
    
def plot_results(y_test_inv, y_pred_inv, test_dates,tag="Comparación: Valores Reales vs Predicciones"):
    plt.figure(figsize=(12, 6))
    plt.plot(test_dates, y_test_inv, label="Real", marker='o', linestyle='-')
    plt.plot(test_dates, y_pred_inv, label="Predicción", marker='x', linestyle='--')
    plt.xlabel("Fecha")
    plt.ylabel("Valor")
    plt.xticks(rotation=45)
    plt.legend()
    plt.title(tag)
    plt.tight_layout()
    plt.show()
    
def metrics(y_test_inv, y_pred_inv):
    mse = mean_squared_error(y_test_inv, y_pred_inv)
    rmse = root_mean_squared_error(y_test_inv, y_pred_inv)
    r2 = r2_score(y_test_inv, y_pred_inv)
    
    LOG.info("Error cuadrático medio: %s", mse)
    LOG.info("RMSE: %s", rmse)
    LOG.info("R2 Score: %s", r2)
    
    return mse, rmse, r2


def execute(df, target, show_plot=False, show_metrics=False, show_statistics=False, check_noise=False, num_runs=30, noise_level=0.1):

    df['Date'] = pd.to_datetime(df['Date'])

    df = df.sort_values(by='Date')

    df.set_index('Date', inplace=True)
    
    y_test_inv, y_pred_inv, test_dates, test_loss, model, scaler, X_test, y_test = RNN(df, timesteps=10, target_column=target, logs=not show_statistics, return_model=True)


    if check_noise:
        
        
        y_test_noisy_inv, y_pred_noisy_inv, test_loss_noisy, X_test_noisy = evaluate_with_noise(model, X_test, y_test, scaler, noise_level)
        
        LOG.info("\n--- Resultados con Datos Ruidosos ---")
        LOG.info(f"Pérdida original: {test_loss:.4f} | Pérdida con ruido: {test_loss_noisy:.4f}")
        
        if show_metrics:
            LOG.info("\nMétricas Originales:")
            metrics(y_test_inv, y_pred_inv)
            LOG.info("\nMétricas con Ruido:")
            metrics(y_test_noisy_inv, y_pred_noisy_inv)
            
        if show_plot:
            plot_results(y_test_noisy_inv, y_pred_noisy_inv, test_dates)
            
        if show_statistics:
            errors = []
            errors_noisy = []

            for i in range(num_runs):
                LOG.info(f"Ejecutando la RNN - Iteración {i + 1}")
                y_test_inv, y_pred_inv, test_dates, loss = RNN(df, timesteps=10, target_column=target, logs=not show_statistics)
                mse, rmse, r2 = metrics(y_test_inv, y_pred_inv)
                
                errors.append((loss, mse, rmse, r2))
                
                y_test_inv_n, y_pred_noisy_inv, loss_noisy,_ = evaluate_with_noise(model, X_test, y_test, scaler, noise_level)
                mse_noisy, rmse_noisy, r2_noisy = metrics(y_test_inv_n, y_pred_noisy_inv)
                
                errors_noisy.append((loss_noisy, mse_noisy, rmse_noisy, r2_noisy))

            LOG.info("\n--- Estadísticas de Error ---")
            LOG.info("Sin Ruido:")
            statistics(errors, num_runs)
            LOG.info("\nCon Ruido:")
            statistics(errors_noisy, num_runs)
            
        return y_test_inv, y_pred_inv, test_dates, test_loss

    else:
        
        if show_plot:
            plot_results(y_test_inv, y_pred_inv, test_dates)

        if show_metrics:
            metrics(y_test_inv, y_pred_inv)
        if show_statistics:
            errors = []

            for i in range(num_runs):
                LOG.info(f"Ejecutando la RNN - Iteración {i + 1}")
                y_test_inv_new, y_pred_inv_new, test_dates, loss = RNN(df, timesteps=10, target_column=target)
                mse, rmse, r2 = metrics(y_test_inv_new, y_pred_inv_new)
                errors.append((loss, mse, rmse, r2))

            statistics(errors, num_runs)
    
    return y_test_inv, y_pred_inv, test_dates, test_loss

def statistics(errors, num_runs):
    
    losses = [error[0] for error in errors]
    mses = [error[1] for error in errors]
    rmses = [error[2] for error in errors]
    r2s = [error[3] for error in errors]
    
    LOG.info(f"Error cuadrático medio promedio: {np.mean(mses):.6f} ± {np.std(mses):.6f}")
    LOG.info(f"RMSE promedio: {np.mean(rmses):.6f} ± {np.std(rmses):.6f}")
    LOG.info(f"R2 Score promedio: {np.mean(r2s):.6f} ± {np.std(r2s):.6f}")

    base_error = mses[0]
    t_stat, p_value = ttest_1samp(mses, popmean=base_error)
    
    LOG.info(f"T-statistic: {t_stat:.6f}")
    LOG.info(f"P-value: {p_value:.6f}")
    

    if p_value < 0.05:
        LOG.info("Los resultados de la RNN varían significativamente entre ejecuciones (p < 0.05).")
    else:
        LOG.info("No hay suficiente evidencia para decir que los resultados de la RNN cambian significativamente.")

    return t_stat, p_value
    
def evaluate_with_noise(model, X_test, y_test, scaler, noise_level=0.1):
    noiser = DataNoiser(noise_level)
    X_test_noisy = noiser.add_noise(X_test)
    
    model.eval()
    with torch.no_grad():
        y_pred_noisy = model(X_test_noisy).squeeze()
        test_loss = nn.MSELoss()(y_pred_noisy, y_test)
    
    y_test_inv = scaler.inverse_transform(y_test.numpy().reshape(-1, 1))
    y_pred_noisy_inv = scaler.inverse_transform(y_pred_noisy.numpy().reshape(-1, 1))
    
    y_test_inv = y_test_inv[:-1]
    y_pred_noisy_inv = y_pred_noisy_inv[1:]
    
    return y_test_inv, y_pred_noisy_inv, test_loss.item(), X_test_noisy


if __name__ == "__main__":
    path = os.path.abspath(os.path.join(os.path.dirname(__file__),"../../data/Anexos_7/full/total_incomes_augmented_full_data.csv"))
    df = pd.read_csv(path)
        
    y_test_inv, y_pred_inv, test_dates, _ = execute(df=df, target='Pinar del Rio', show_plot=False, show_metrics=True, show_statistics=False, num_runs=50, check_noise=True , noise_level=0.05)
    
        
    # for target in df.columns[1:17]:
    #     print(f"Predicting for {target}")