"""
TC3006C.102: Inteligencia artificial avanzada para la ciencia de datos I

Entregable: Implementación de una técnica de aprendizaje de máquina sin el uso de un framework

Por: Fernando Bustos Monsiváis - A00829931

Profesor: MSC Jesús Adrián Rodríguez Rocha

Fecha de creación: Viernes 23 de agosto de 2024 a las 11:35 a.m.
Fecha de última modificación: Domingo 25 de agosto de 2024 a las 04:05 p.m.
"""

# Importar la librería pandas para la manipulación de datos
import pandas as pd

# Leer el archivo CSV
df = pd.read_csv('weatherHistory.csv')

# Dividir los datos en training set y test set
df_train = df[:700]
df_test = df[700:1000]

# Definir feature (X) y label (y) para training set y test set
X_train = df_train["Apparent Temperature (C)"]
y_train = df_train["Temperature (C)"]

X_test = df_test["Apparent Temperature (C)"]
y_test = df_test["Temperature (C)"]

# Definir el tamaño del learning rate (alpha) y el número de epochs
alpha = 0.000001
epochs = 12000

# Actualiza los parámetros w (pendiente) y b (intersección) usando gradient descent
def update_w_and_b(X, y, w, b, alpha):
    dl_dw = 0.0 # Gradiente con respecto a w, inicializado en 0.0
    dl_db = 0.0 # Gradiente con respecto a b, inicializado en 0.0
    N = len(X) # Número de muestras
    for i in range(N):
        # Calcular la derivada parcial de la función de costo (MSE) con respecto a w y b
        dl_dw += -2 * X[i] * (y[i] - (w * X[i] + b))
        dl_db += -2 * (y[i] - (w * X[i] + b))
    # Actualizar los parámetros w y b
    w = w - (1/float(N)) * dl_dw * alpha
    b = b - (1/float(N)) * dl_db * alpha
    return w, b

# Entrena el modelo usando gradient descent
def train(X, y, w, b, alpha, epochs):
    print('Training progress:')
    for e in range(epochs):
        # Llamar a update_w_and_b para ajustar w y b
        w, b = update_w_and_b(X, y, w, b, alpha)
        if e % 1000 == 0: # Mostrar el progreso cada 1000 epochs
            avg_loss_ = avg_loss(X, y, w, b) # Calcular la pérdida con la función de costo (MSE)
            print(f"Epoch: {e:<7} | Loss: {avg_loss_:<12.8f} | w: {w:<8.4f} | b: {b:<10.4f}")
    return w, b

# Cálculo de la función de costo (Mean Squared Error (MSE))
def avg_loss(X, y, w, b):
    N = len(X) # Número de muestras
    total_error = 0.0 # Inicializar el error total
    for i in range(N):
        # Sumar el error cuadrático para cada muestra
        total_error += (y[i] - (w * X[i] + b))**2
    return total_error / float(N)

# Realiza una predicción usando la función del modelo
def predict(x, w, b):
    return w * x + b

# Entrenar el modelo
w, b = train(X=X_train, y=y_train, w=0.0, b=0.0, alpha=alpha, epochs=epochs)

# Imprimir la función del modelo
print(f"\nHypothesis function: y = {round(w, 4)}x + {round(b, 4)}")

# Predecir los valores del test set
y_pred = [predict(x, w, b) for x in X_test]

# Imprimir los primeros 15 valores predichos y los valores reales del test set
pred_values = ' '.join(f"{round(val):<4}" for val in y_pred[:15])
actual_values = ' '.join(f"{round(val):<4}" for val in y_test[:15])
print(f"\nPrimeros 15 valores predichos:    {pred_values}\nPrimeros 15 valores del test set: {actual_values}")

# Calcular la precisión del modelo
correct_predictions = sum(1 for actual, pred in zip(y_test, y_pred) if round(actual) == round(pred))
accuracy = correct_predictions / len(y_test)

print(f"\nPrecisión del modelo: {accuracy * 100:.2f}%")

"""
Output:

Training progress:
Epoch: 0       | Loss: 182.90108126 | w: 0.0004   | b: 0.0000    
Epoch: 1000    | Loss: 90.55505980  | w: 0.3019   | b: 0.0215    
Epoch: 2000    | Loss: 45.19606207  | w: 0.5133   | b: 0.0367    
Epoch: 3000    | Loss: 22.91625241  | w: 0.6614   | b: 0.0476    
Epoch: 4000    | Loss: 11.97252843  | w: 0.7651   | b: 0.0555    
Epoch: 5000    | Loss: 6.59688770   | w: 0.8379   | b: 0.0612    
Epoch: 6000    | Loss: 3.95619033   | w: 0.8888   | b: 0.0654    
Epoch: 7000    | Loss: 2.65884854   | w: 0.9245   | b: 0.0686    
Epoch: 8000    | Loss: 2.02133923   | w: 0.9495   | b: 0.0711    
Epoch: 9000    | Loss: 1.70792806   | w: 0.9670   | b: 0.0730    
Epoch: 10000   | Loss: 1.55370829   | w: 0.9793   | b: 0.0746    
Epoch: 11000   | Loss: 1.47768069   | w: 0.9878   | b: 0.0759    

Hypothesis function: y = 0.9938x + 0.0771

Primeros 15 valores predichos:    2    0    -1   1    4    10   12   13   14   15   16   17   17   17   17  
Primeros 15 valores del test set: 2    2    2    4    6    10   12   13   14   15   16   17   17   17   17  

Precisión del modelo: 87.00%
"""

"""
Reporte:

En este programa, se aplica un modelo de regresión lineal simple para predecir la temperatura real utilizando la 
temperatura aparente como única característica (feature), sin el uso de un framework de aprendizaje de máquina. 
Este proceso se realizó con un dataset de Kaggle que contiene datos de temperaturas en Szeged, Hungría, desde 
2006 hasta 2016.

El dataset utilizado, 'weatherHistory.csv', contiene varias features relacionadas con las condiciones 
meteorológicas. De estas, se seleccionó la temperatura aparente como feature 'X', y la temperatura real como el 
objetivo a predecir (label o target) 'y'. El conjunto de datos se dividió en un training set (primeras 700 
instancias) para entrenar el modelo, y un test set (instancias 701-1000) para evaluar su desempeño.

El objetivo del modelo es encontrar la relación lineal entre feature y label, lo que se expresa mediante la 
hypothesis function o modelo: y = wx + b, donde 'w' es la pendiente y 'b' es la intersección con el eje 'y'. Este 
modelo se entrena para minimizar la función de costo, que en este caso es el Mean Squared Error (MSE), una medida 
que cuantifica el error promedio al cuadrado entre las predicciones del modelo y los valores reales.

Para ajustar los parámetros 'w' y 'b', se utilizó el algoritmo de gradient descent, que actualiza estos 
parámetros iterativamente para minimizar el MSE. Se inicializaron 'w' y 'b' en 0, y se realizaron 12 000 
iteraciones (epochs) con un learning rate (alpha) de 0.000001. Durante cada iteración, se calcula la derivada de 
la función de costo con respecto a 'w' y 'b', y se actualizan estos parámetros en dirección opuesta al gradiente 
para reducir el error.

La precisión del modelo se calculó con el número de predicciones correctas sobre el total de instancias del test 
set. En este caso, se obtuvo una precisión del 87.00%, lo que indica que el modelo es capaz de predecir la 
temperatura real a partir de la temperatura aparente con un buen nivel de precisión. También se imprimieron los 
primeros 15 valores predichos y los valores reales del test set para visualizar la comparación.

El valor del learning rate y el número de epochs son hiperparámetros clave en el rendimiento del modelo. Se 
seleccionaron estos valores mediante pruebas y ajustes, asegurando un balance entre el tiempo de entrenamiento y 
la precisión del modelo.

Referncias:
1. Creative Commons (CC). (2024). Atribución/Reconocimiento-NoComercial-CompartirIgual 4.0 Internacional. 
Recuperado el lunes 09 de septiembre de 2024, de https://creativecommons.org/licenses/by-nc-sa/4.0/deed.es

2. Budincsevity, N. (2016). Weather in Szeged 2006-2016. Recuperado el 09 de septiembre de 2024, de Kaggle: 
https://www.kaggle.com/datasets/budincsevity/szeged-weather?select=weatherHistory.csv
"""
