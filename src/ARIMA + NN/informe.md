### 1. Modelo ARIMA

1. **Definición**
   Un modelo **ARIMA(\(p,d,q\))** (AutoRegressive Integrated Moving Average) se compone de:
   - Parte **AR** (autoregresiva): usa valores pasados de la serie.
   - Parte **I** (integrada): diferencia la serie para hacerla estacionaria.
   - Parte **MA** (media móvil): incorpora términos pasados del error.

2. **Fórmula ARIMA**
   De forma general, un ARIMA(\(p,d,q\)) puede expresarse como:
   \[
     (1 - \phi_1 L - \cdots - \phi_p L^p)(1 - L)^d \, X_t
       = (1 + \theta_1 L + \cdots + \theta_q L^q)\,\varepsilon_t,
   \]
   donde:
   - \(X_t\) es la serie,
   - \(L\) es el operador “lag” (desplaza la serie en el tiempo),
   - \(\phi_i\) y \(\theta_j\) son coeficientes a estimar,
   - \(d\) es el orden de diferenciación requerido para la estacionariedad,
   - \(\varepsilon_t\) es un ruido blanco con media cero.

3. **Ajuste en el código**
   - Se utiliza la función `auto_arima(...)` para estimar automáticamente \(p\), \(d\), \(q\), ajustándose a la serie de **entrenamiento**.
   - Se obtiene una predicción **\( \hat{X}_t^{(ARIMA)} \)** para el periodo de prueba y un vector de **residuos** en entrenamiento.

---

### 2. Residuos y Red Neuronal

1. **Cálculo de Residuos**
   Tras ajustar ARIMA, se define:
   \[
     e_t = X_t - \hat{X}_t^{(ARIMA)},
   \]
   donde \(X_t\) es el valor real en el tiempo \(t\) y \(\hat{X}_t^{(ARIMA)}\) la predicción lineal del ARIMA. Este residuo \(e_t\) contiene la información no lineal no capturada por ARIMA.

2. **Red Neuronal para Residuos**
   - La **red neuronal** recibe como entrada los lags de la serie (o a veces de los propios residuos).
   - El objetivo es predecir \( e_t \) para cada \(t\). Si la red acierta bien estos residuos, significa que está captando patrones no lineales adicionales.

3. **Fórmula de la Red**
   Para un MLP (Multi-Layer Perceptron) con una capa oculta, puede representarse como:
   \[
     \hat{e}_t = f(W_2 \, f(W_1 \, \mathbf{z}_t + \mathbf{b}_1) + \mathbf{b}_2),
   \]
   donde:
   - \(\mathbf{z}_t\) son los lags de la serie como entrada,
   - \(W_1, W_2\) y \(\mathbf{b}_1, \mathbf{b}_2\) son parámetros de la red,
   - \(f\) es la función de activación, p. ej. Tanh o ReLU.

4. **Entrenamiento en el código**
   - Se normalizan la serie y los residuos, se generan lags para la parte neuronal, y se entrena un MLP para predecir \(e_t\).
   - Se usa un optimizador (Adam) con un criterio de error (MSE) para que la red aprenda a “ajustar” los residuos.

---

### 3. Predicción Final del Modelo Híbrido

El pronóstico del modelo híbrido en el tiempo \( t \) (en test) se define como la suma de la parte lineal ARIMA y la parte no lineal aprendida por la NN:
\[
  \hat{X}_t^{(\mathrm{Híbrido})}
    = \hat{X}_t^{(\mathrm{ARIMA})} + \hat{e}_t^{(\mathrm{NN})}.
\]
Así, la red neuronal actúa como **corrector** de los errores que ARIMA deja sin capturar.

---

### 4. Guía de Desarrollo en el Código

1. **Lectura de Datos:**
   - Se separa un CSV para entrenamiento y otro para prueba.
   - Se extrae una columna de interés (univariante).

2. **Entrenamiento ARIMA:**
   - Se aplica `auto_arima(...)` sobre los datos de entrenamiento.
   - Se predice y se obtienen los residuos en la parte de entrenamiento.

3. **Creación de Lags y Normalización:**
   - Se normalizan serie y residuos.
   - Se generan lags (ventanas temporales) para la red neuronal.

4. **Entrenamiento de la Red Neuronal (MLP):**
   - Se definen capas ocultas (generalmente 1–2 capas con Tanh).
   - Se entrena con backpropagation para aproximar los \( e_t \).

5. **Combinación de Salidas y Evaluación:**
   - En test, se calcula \(\hat{X}^{(ARIMA)}\), se predice \(\hat{e}^{(NN)}\) y se suman.
   - Se calculan métricas (RMSE, MAE, MAPE).
   - Se analiza la ACF/PACF de los errores finales y se ejecuta la prueba Diebold–Mariano para comparar con ARIMA.

---

## Informe de Resultados Obtenidos

A partir de las métricas reportadas:

```
=== MÉTRICAS EN TEST ===
           RMSE      MAE      MAPE
ARIMA:   37.4318  32.5673  190.85%
Híbrido: 36.3927  31.7844  198.56%

=== Diebold-Mariano Test (ARIMA vs. Híbrido) ===
DM statistic = 1.1777, p-value = 0.2389
La diferencia NO es estadísticamente significativa (p >= 0.05).
```

Podemos extraer las siguientes observaciones:

1. **Rendimiento Numérico:**
   - El modelo híbrido logra **ligeramente** menores valores de RMSE y MAE, lo que sugiere una mejora marginal en el error global de predicción.
   - Sin embargo, el **MAPE** del híbrido es algo mayor, lo que puede indicar que en puntos concretos (especialmente cuando los valores reales son bajos), el híbrido no reduce el error relativo.

2. **Prueba Diebold–Mariano:**
   - El estadístico DM = 1.1777 y un p-valor de ~0.24 indican que la diferencia no es estadísticamente significativa a un nivel de confianza del 95%. Esto implica que, aunque haya una leve mejora en RMSE y MAE, **no** se puede confirmar con evidencia estadística que el híbrido supere a ARIMA de manera concluyente.

3. **Comportamiento de los Errores (ACF/PACF):**
   - En la representación gráfica, se observa que no hay autocorrelaciones muy marcadas en los residuos finales. Esto **es un buen indicio** de que el modelo (lineal + no lineal) está captando la mayoría de la dinámica temporal.
   - No obstante, algunos rezagos se mantienen dentro del rango de confianza pero no revelan un patrón fuertemente sistemático.

4. **Interpretación Final:**
   - El **ligero descenso** en RMSE y MAE del híbrido indica que la red neuronal está corrigiendo algo de la parte no lineal que ARIMA no captó.
   - El **MAPE mayor** sugiere que hay situaciones donde la predicción híbrida no reduce el error proporcional. Esto puede pasar con valores muy pequeños en la serie, que inflan drásticamente el porcentaje.
   - La **falta de significancia estadística** (p!≥0.05) implica que, con los datos actuales, no hay evidencia robusta de una mejora sustancial. Sin embargo, **no** empeora el pronóstico, y en algunos casos se acerca más a los valores reales.
   - En la práctica, puede ser un predictor “aceptable” si se valora una pequeña reducción del error medio y se toleran variaciones altas en MAPE. Se sugiere seguir ajustando la parte neuronal (lags, regularización, etc.), o aumentar la muestra para detectar mejoras más contundentes.
