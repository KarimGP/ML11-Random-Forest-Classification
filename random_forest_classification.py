# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 12:01:53 2024

@author: KGP
"""

# Random forest Classification

# Cómo importar las librerías
import numpy as np # contiene las herrarmientas matemáticas para hacer los algoritmos de machine learning
import matplotlib.pyplot as plt #pyplot es la sublibrería enfocada a los gráficos, dibujos
import pandas as pd #librería para la carga de datos, manipular, etc

# Importar el dataset
dataset = pd.read_csv('Social_Network_Ads.csv')
x = dataset.iloc[:, [2,3]].values 
y = dataset.iloc[:, 4].values
# iloc sirve para localizar por posición las variables, en este caso independientes
# hemos indicado entre los cochetes, coge todas las filas [:(todas las filas), :-1(todas las columnas excepto la última]
# .values significa que quiero sacar solo los valores del dataframe no las posiciones


# Dividir el dataset en conjunto de entrenamiento y conjunto de testing
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state = 0) # random_state podría coger cualquier número, es el número para poder reproducir el algoritmo

# Como es un algoritmo que no está basado en distancias no realizamos el escalado de variables como parte del preprocesado del siguiente código
#pero podríamos activarlo si lo que queremos es que la representación quede más fina (más puntitos pintados).

"""
# Escalado de variables. Siguiente código COMENTADO porque se usa mucho pero no siempre
from sklearn.preprocessing import StandardScaler # Utilizarlo para saber que valores debe escalar apropiadamente y luego hacer el cambio
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test) #hacemos transform sin "fit" para que haga la transformación con los datos del transform de entrenamiento
"""

# Ajustar el clasificador en el Conjunto de Entrenamiento
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=10, criterion="entropy", random_state=0 ) # criterion = "índice de gini es un índice que mide la dispersión de los datos, mientras que la entropía mide el caos, la discrepancia cuando los datos están juntos o separados". Buscaremos minimizar la entropía para homogeneizar y que la suma de entropías sea 0, en este caso
classifier.fit(x_train, y_train)


# Predicción de los resultados con el Conjunto de Testing
y_pred = classifier.predict(x_test)

# Elaborar una matriz de confusión. Mide que tan bien ha evaluado el algoritmo la clasificación de los usuarios (en este caso) en compra o no compra con respecto a cómo estaban etiquetados
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# Representación gráfica de los resultados del algoritmo en el conjunto de entrenamiento. Veremos que regiones el algoritmo decide si compra o no compra
from matplotlib.colors import ListedColormap
X_set, y_set = x_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 1),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 50))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Random Forest (Conjunto de entrenamiento)')
plt.xlabel('Edad')
plt.ylabel('Sueldo Estimado')
plt.legend()
plt.show() 

# Representación gráfica de los resultados del algoritmo en el conjunto de test. Veremos que regiones el algoritmo decide si compra o no compra
from matplotlib.colors import ListedColormap
X_set, y_set = x_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 1),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 50))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Random forest (Conjunto de Test)')
plt.xlabel('Edad')
plt.ylabel('Sueldo Estimado')
plt.legend()
plt.show() 