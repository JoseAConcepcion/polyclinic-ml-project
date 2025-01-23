import numpy as np

def mse(y_true, y_pred):
    """
    Calcula el Error Cuadrático Medio (MSE).

    El MSE mide el promedio de los cuadrados de las diferencias
    entre los valores reales (y_true) y los valores predichos (y_pred).
    Un MSE bajo indica un mejor ajuste del modelo.
    """
    return np.mean((y_true - y_pred) ** 2)

def rmse(y_true, y_pred):
    """
    Calcula la Raíz del Error Cuadrático Medio (RMSE).

    El RMSE es la raíz cuadrada del MSE y se expresa en la misma escala
    que la variable objetivo. Penaliza más los errores grandes
    al igual que el MSE, pero facilita la interpretación por compartir escala.
    """
    return np.sqrt(mse(y_true, y_pred))

def mae(y_true, y_pred):
    """
    Calcula el Error Absoluto Medio (MAE).

    El MAE mide el promedio de las diferencias absolutas
    entre los valores reales y los valores predichos.
    Es más robusto frente a valores atípicos que el MSE o el RMSE.
    """
    return np.mean(np.abs(y_true - y_pred))

def mape(y_true, y_pred):
    """
    Calcula el Error Absoluto Porcentual Medio (MAPE).

    El MAPE mide el error promedio en términos porcentuales
    respecto a los valores reales. Es especialmente útil para
    comparar el desempeño en problemas con diferentes escalas,
    pero debe usarse con precaución cuando hay valores cercanos a cero
    en y_true.
    """
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
