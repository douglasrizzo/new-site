---
layout: post
title: Classificação da base de dados Iris com a menor quantidade de dados e rede neural possível
categories: colab portugues keras neural-networks python classification tutorial pca
---

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1cA81_uuJlS9IDItBqfHOeIPWB9yCp59I?usp=sharing)

Este notebook continua uma série na qual foram introduzidas a base de dados Iris e o método de treinamento e avaliação de redes neurais [[1]](https://colab.research.google.com/drive/13r1TY_BF1AXOMH8ufDEct3E8g243sONM#scrollTo=MLs97GxwgNrs) e o PCA [[2]](https://colab.research.google.com/drive/15vsVfH-qlJP2YBCdgS8V8uc8jjkRYCFs#scrollTo=LOpZfT9qCa3K).

Aqui, vamos enfrentar o desafio de realizar a classificação da base de dados Iris na menor rede neural possível.

# Importando pacotes e criando funções

As funções abaixo aplicam o PCA numa base de dados e avaliam uma rede neural, retornando as medidas para serem exibidas em gráficos posteriormente.

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt

import tensorflow.keras as keras
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

def apply_PCA(X, n_components):
  pca = PCA(n_components=n_components)
  new_X = pca.fit_transform(iris_X)
  print('Tamanho da base de dados antes: ', X.shape)
  print('Tamanho da base de dados depois: ', new_X.shape)
  print('Porcentagem da variância explicada por cada coluna:', pca.explained_variance_ratio_)
  print('Variância acumulada nas colunas remanescentes:', sum(pca.explained_variance_ratio_))

  return new_X

def evaluate_model(model, X, y):
  X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size = 0.2,
                                                    random_state=123)
  
  y_train_onehot = keras.utils.to_categorical(y_train, num_classes = 3)
  y_test_onehot = keras.utils.to_categorical(y_test, num_classes = 3)

  model.save_weights('weights.hdf5')
  history = model.fit(X_train, y_train_onehot, validation_split=.1, epochs=500, verbose=0)

  v1 = history.history['categorical_accuracy']
  v2 = history.history['loss']
  v3 = history.history['val_categorical_accuracy']
  v4 = history.history['val_loss']

  model.load_weights('weights.hdf5')
  history = model.fit(X_train, y_train_onehot, epochs=150, verbose=0)
  
  v5 = history.history['categorical_accuracy']
  v6 = history.history['loss']
  
  y_pred = model.predict(X_test)
  y_pred_onehot = y_pred.argmax(axis=1)

  print(classification_report(y_test, y_pred_onehot))
  print(confusion_matrix(y_test, y_pred_onehot))
  
  return v1, v2, v3, v4, v5, v6
```

## Preparando os dados

Usaremos o pacote *scikit-learn* para carregar a base de dados Iris. Vamos gerar versões com o maior e menor número possível de componentes principais (4 e 1, respectivamente).

```python
iris_X, iris_y = load_iris(return_X_y=True)
iris_X_large = apply_PCA(iris_X, 4)
iris_X_tiny = apply_PCA(iris_X, 1)
```

    Tamanho da base de dados antes:  (150, 4)
    Tamanho da base de dados depois:  (150, 4)
    Porcentagem da variância explicada por cada coluna: [0.92461872 0.05306648 0.01710261 0.00521218]
    Variância acumulada nas colunas remanescentes: 1.0
    Tamanho da base de dados antes:  (150, 4)
    Tamanho da base de dados depois:  (150, 1)
    Porcentagem da variância explicada por cada coluna: [0.92461872]
    Variância acumulada nas colunas remanescentes: 0.9246187232017271

As funções abaixo vão criar duas rede neurais distintas. A primeira foi utilizada nos tutoriais anterior e possui 4 entradas, 4 camadas e 213 pesos treináveis. A segunda rede possui aproximadamente 10% do tamanho da primeira, 1 entrada, 3 camadas e 23 pesos treináveis.

```python
def create_large_model(input_dim):
  model = Sequential()
  model.add(Dense(10, activation='tanh', input_dim=input_dim))
  model.add(Dense(8,activation='tanh'))
  model.add(Dense(6,activation='tanh'))
  model.add(Dense(3,activation='softmax'))
  model.compile('adam','categorical_crossentropy', metrics=['categorical_accuracy'])

  return model

def create_tiny_model(input_dim):
  model = Sequential()
  model.add(Dense(3, activation='tanh', input_dim=input_dim))
  model.add(Dense(2,activation='tanh'))
  model.add(Dense(3,activation='softmax'))
  model.compile('adam','categorical_crossentropy', metrics=['categorical_accuracy'])

  return model

large_net = create_large_model(iris_X_large.shape[1])
tiny_net = create_tiny_model(iris_X_tiny.shape[1])

large_net.summary()
tiny_net.summary()
```

    Model: "sequential"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    dense (Dense)                (None, 10)                50        
    _________________________________________________________________
    dense_1 (Dense)              (None, 8)                 88        
    _________________________________________________________________
    dense_2 (Dense)              (None, 6)                 54        
    _________________________________________________________________
    dense_3 (Dense)              (None, 3)                 21        
    =================================================================
    Total params: 213
    Trainable params: 213
    Non-trainable params: 0
    _________________________________________________________________
    Model: "sequential_1"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    dense_4 (Dense)              (None, 3)                 6         
    _________________________________________________________________
    dense_5 (Dense)              (None, 2)                 8         
    _________________________________________________________________
    dense_6 (Dense)              (None, 3)                 9         
    =================================================================
    Total params: 23
    Trainable params: 23
    Non-trainable params: 0
    _________________________________________________________________

Vamos utilizar nossa função `evaluate_model()` para avaliar as duas redes. A rede maior sera treinada na base com 4 componentes principais e a rede menor, na base com 1 componente principal.

```python
acc1_large, l1_large, acc_v_large, lv_large, acc2_large, l2_large = evaluate_model(large_net, iris_X_large, iris_y)
acc1_tiny, l1_tiny, acc_v_tiny, lv_tiny, acc2_tiny, l2_tiny = evaluate_model(tiny_net, iris_X_tiny, iris_y)
```

                  precision    recall  f1-score   support
    
               0       1.00      1.00      1.00        13
               1       0.75      1.00      0.86         6
               2       1.00      0.82      0.90        11
    
        accuracy                           0.93        30
       macro avg       0.92      0.94      0.92        30
    weighted avg       0.95      0.93      0.93        30
    
    [[13  0  0]
     [ 0  6  0]
     [ 0  2  9]]
                  precision    recall  f1-score   support
    
               0       1.00      1.00      1.00        13
               1       1.00      0.50      0.67         6
               2       0.79      1.00      0.88        11
    
        accuracy                           0.90        30
       macro avg       0.93      0.83      0.85        30
    weighted avg       0.92      0.90      0.89        30
    
    [[13  0  0]
     [ 0  3  3]
     [ 0  0 11]]

Os valores podem mudar aleatoriamente, porém é possível perceber através dos valores de precisão, recall, F1-score e pela matriz de confusão que o desempenho de ambas as redes é comparável e, muitas vezes, idêntico.

Vamos gerar gráficos das 6 medidas coletadas dos dois modelos.

```python
fig, axes = plt.subplots(3,2, False,figsize=(20, 15), squeeze=True)

axes[0][0].plot(acc1_large)
axes[0][0].plot(acc1_tiny)
axes[0][0].set_title('Acurácia Treino')

axes[0][1].plot(l1_large)
axes[0][1].plot(l1_tiny)
axes[0][1].set_title('Erro Treino')

axes[1][0].plot(acc_v_large)
axes[1][0].plot(acc_v_tiny)
axes[1][0].set_title('Acurácia Val.')

axes[1][1].plot(lv_large)
axes[1][1].plot(lv_tiny)
axes[1][1].set_title('Erro Val.')

axes[2][0].plot(acc2_large)
axes[2][0].plot(acc2_tiny)
axes[2][0].set_title('Acurácia total')

axes[2][1].plot(l2_large)
axes[2][1].plot(l2_tiny)
axes[2][1].set_title('Erro total')

legenda = ['Grande', 'Pequena']
xlabel = 'Épocas'

axes[0][0].legend(legenda)
axes[1][0].legend(legenda)
axes[2][0].legend(legenda)
axes[0][1].legend(legenda)
axes[1][1].legend(legenda)
axes[2][1].legend(legenda)
axes[0][0].set_xlabel(xlabel)
axes[1][0].set_xlabel(xlabel)
axes[2][0].set_xlabel(xlabel)
axes[0][1].set_xlabel(xlabel)
axes[1][1].set_xlabel(xlabel)
axes[2][1].set_xlabel(xlabel)

plt.show()
```

![png](../images/output_10_0.png)

Ambas as redes foram treinadas por 500 épocas e avaliadas no conjunto de validação para constatar a presença de *overfitting* (4 primeiros gráficos). Depois, foram treinadas novamente por 150 épocas na totalidade dos dados de treinamento (2 últimos gráficos).

O gráfico de erro no conjunto de treinamento (topo, direita) demonstra que a rede menor leva consideravelmente mais tempo para aprender. Seu erro diminui lentamente.

O gráfico de erro no conjunto de validação (centro, direita) demonstra que a rede menor é menos afetada por *overfitting*. Seu erro estabiliza, mas não parece aumentar, ao contrário da rede maior, que é capaz de se especializar aos dados de treinamento.

# Conclusão

Este notebook demonstrou que uma rede neural consideravelmente menor que a utilizada até o momento é capaz de realizar a classificação das flores na base de dados Iris, utilizando 1/4 dos valores de entrada, 3/4 do número de camadas e aproximadamente 1/10 dos pesos treináveis.
