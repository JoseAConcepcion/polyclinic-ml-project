# RNN

Las Redes Neuronales Recurrentes están diseñadas para procesar datos secuenciales como series temporales, en nuestro caso aplicamos RNN para predecir la cantidad de ingresos por día de enfermos con Fiebre indeterminada o Sospecha de dengue en todos los hospitales de una provincia dada.

## Dataset

Ejemplo del dataset para una provincia:
![alt text](image.png)

Se entrenó el modelo con un total de 107 datos de días secuenciales y se testeo con 27 datos.

## Porqué RNN

La idea de utilizar RNN es identificar patrones en los ingresos, como picos relacionados con ciertas épocas del año, días de la semana u otros factores temporales. Una RNN con múltiples capas pudiera detectar relacionas a largo y corto plazo.

# Resultados

En la imagen se observan la cantidad de ingresos por días predecidos vs los reales, la principal dificultad se observa en la predicción de los picos de ingresos.
![alt text](image-1.png)

Durante el entrenamiento se probaron diferentes valores del hipérparámetro `n_steps` obteniendo valores muy similares en los resultados, finalmente se seleccionó un valor de 7. (7 días anteriores para predecir el siguiente)
Se obtuvo un Error Cuadrático Medio de 4009 y su raíz cuadrada es aproximadamente 63. Lo que
indica que el modelo tiene un error promedio de 63 ingresos por día. Dado que los valores de los datos están entre 0 y 300 el error es considerable.

# Dificultades

Dado que los intervalos no son regulares en el tiempo las relaciones temporales no son captadas adecuadamente por la RNN. Además el conjunto de datos es pequeño para entrenar correctamente el modelo.

# Conclusiones

Para mayor precisión en la predicción se intentará aumentar el tamaño del dataset y lograr una regularidad en los intervalos de tiempo.

# Recomendaciones futuras

EL trabajo en esta tarea fue básica para la familiarización con el flujo de trabajo y el pipeline de desarrollo. Se deben incorporar otras métricas y además combinar con otros métodos cómo se plantea en algunos papers.

---

# Nueva ejecución con datos ampliados

![alt text](image-3.png)

Como se observa en la imagen, los resultados mejoraron SIGNIFICATIVAMENTE, las métricas obtenidas fueron:

- Error cuadrático medio: 214.1036
- RMSE: 14.632279
- R2 Score: 0.9390216343195826

Verificamos si está ocurriendo overfitting viendo el comportamiento de las diferentes métricas y aplicando K-Fold Cross Validation.

## K-Fold Cross Validation

Fold 1/5
Fold 1: RMSE=19.5407, R2=0.8623
Fold 2/5
Fold 2: RMSE=23.9539, R2=0.8360
Fold 3/5
Fold 3: RMSE=37.7033, R2=0.6905
Fold 4/5
Fold 4: RMSE=21.7118, R2=0.8510
Fold 5/5
Fold 5: RMSE=32.2624, R2=0.6255

Resultados de Validación Cruzada:
RMSE Promedio: 27.0344
R2 Promedio: 0.7731

## Conclusión

El modelo no parece estar sobreajustado pues las métricas obtenidas en la validación cruzada son similares a las obtenidas anteriormente. Parece indicar que el modelo "aprendió" bien las relaciones temporales en los datos.
