---
layout: post
title: Classificação da base de dados Iris utilizando um perceptron multi-camadas em Keras
categories: colab portugues keras neural-networks python classification tutorial
---

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/13r1TY_BF1AXOMH8ufDEct3E8g243sONM?usp=sharing)

Este notebook exemplifica o treinamento de uma perceptron multi-camadas na classificação da base de dados Iris. O notebook utiliza o pacote *scikit-learn* para carregamento e separação da base de dados em treinamento e teste, o pacote *keras* para criação e treinamento da rede neural e o *matplotlib* para a geração de gráficos.

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

import matplotlib.pyplot as plt

import tensorflow.keras as keras
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD,Adam
```

## Preparando a base de dados

A base de dados Iris contém 4 medidas de 150 pétalas de flores de 3 espécies distintas (50 pétalas de cada espécie). Ela foi criada em 1936 por Ronald Fisher [[link]](https://en.wikipedia.org/wiki/Iris_flower_data_set).

Neste notebook, uma rede neural será utilizada para descobrir a qual das 3 espécies de flor cada pétala pertence, dadas as 4 medidas da pétala. Em outras palavras, a rede neural será um classificador treinado para prever em qual classe (de 3) um vetor de 4 valores pertence.

![Iris setosa](https://upload.wikimedia.org/wikipedia/commons/thumb/5/56/Kosaciec_szczecinkowaty_Iris_setosa.jpg/180px-Kosaciec_szczecinkowaty_Iris_setosa.jpg)
![Iris versicolor](https://upload.wikimedia.org/wikipedia/commons/thumb/4/41/Iris_versicolor_3.jpg/320px-Iris_versicolor_3.jpg)
![Iris virginica](https://upload.wikimedia.org/wikipedia/commons/thumb/9/9f/Iris_virginica.jpg/295px-Iris_virginica.jpg)

Usaremos o pacote *scikit-learn* para carregar a base de dados e separá-la entre conjuntos de treinamento e de teste.


```python
iris_X, iris_y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(iris_X,
                                                    iris_y,
                                                    test_size = 0.2,
                                                    random_state=123)

print('Qtd. itens no conj. de treinamento:', len(y_train))
print('Classes:', y_train)
print('\nQtd. itens no conj. de teste:', len(y_test))
print('Classes:', y_test)
print('\nExemplo de medida de uma flor:', X_train[0])
```

    Qtd. itens no conj. de treinamento: 120
    Classes: [2 2 0 0 1 1 2 0 0 1 1 0 2 2 2 2 2 1 0 0 2 0 0 1 1 1 1 2 1 2 0 2 1 0 0 2 1
     2 2 0 1 1 2 0 2 1 1 0 2 2 0 0 1 1 2 0 0 1 0 1 2 0 2 0 0 1 0 0 1 2 1 1 1 0
     0 1 2 0 0 1 1 1 2 1 1 1 2 0 0 1 2 2 2 2 0 1 0 1 1 0 1 2 1 2 2 0 1 0 2 2 1
     1 2 2 1 0 1 1 2 2]
    
    Qtd. itens no conj. de teste: 30
    Classes: [1 2 2 1 0 2 1 0 0 1 2 0 1 2 2 2 0 0 1 0 0 2 0 2 0 0 0 2 2 0]
    
    Exemplo de medida de uma flor: [7.4 2.8 6.1 1.9]


Vamos categorizar as classes de flores (0, 1 e 2) utilizando *one-hot encoding*, uma espécie de categorização de dados que torna o aprendizado de classe linearmente independente para a rede neural.


```python
y_train_onehot = keras.utils.to_categorical(y_train, num_classes = 3)
y_test_onehot = keras.utils.to_categorical(y_test, num_classes = 3)

# imprime os 5 primeiros valores para exemplificar
print("### Antes ###")
print(y_train[:5])
print(y_test[:5])

print("\n### Depois ###")
print(y_train_onehot[:5])
print(y_test_onehot[:5])
```

    ### Antes ###
    [2 2 0 0 1]
    [1 2 2 1 0]
    
    ### Depois ###
    [[0. 0. 1.]
     [0. 0. 1.]
     [1. 0. 0.]
     [1. 0. 0.]
     [0. 1. 0.]]
    [[0. 1. 0.]
     [0. 0. 1.]
     [0. 0. 1.]
     [0. 1. 0.]
     [1. 0. 0.]]


## Declarando a topologia da rede neural

Vamos construir uma rede neural utilizando Keras. Aqui, configuramos camadas, neurônios por camada, funções de ativação, otimizador e função de erro.

Após construir a rede neural, utilizamos uma função utilitária do Keras para exibir a topologia do modelo. Neste caso, a rede receberá como entrada vetores de tamanho 4 (correspondente à quantidade de medidas de nossas pétalas e utilizará 4 camadas densas para realizar a classificação.

Repare que o tamanho da saída de uma camada equivale ao tamanho dos valores de entrada da camada seguinte.

A última camada da rede tem como saída vetores de tamanho 3, correspondentes à quantidade de espécies de flores que desejamos classificar.

Vamos criar uma função para conseguir recriar o mesmo modelo no futuro.


```python
def create_model():
  model = Sequential()
  model.add(Dense(10, activation='tanh', input_dim=4))
  model.add(Dense(8,activation='tanh'))
  model.add(Dense(6,activation='tanh'))
  model.add(Dense(3,activation='softmax'))
  model.compile('adam','categorical_crossentropy', metrics=['categorical_accuracy'])

  return model

model=create_model()

keras.utils.plot_model(
    model,
    show_shapes=True,
    show_layer_names=True,
    rankdir="LR",
    expand_nested=True,
    dpi=96
)
```

![png](../images/output_7_0.png)



## Treinando a rede

Vamos treinar nossa rede neural. Ela aprenderá a realizar a classificação dos dados de treinamento (`X_train`) para as classes categorizadas (`y_train_onehot`) por um número de épocas pré-determinado.

O Keras permite armazenar o progresso do treinamento em uma variável, a qual será utilizada no futuro.

Repare a diminuição da função de erro e aumento da acurácia (porcentagem de classificações corretas durante o treino) ao longo das épocas.

É possível executar a célula abaixo repetidas vezes para que a mesma rede seja treinada por mais épocas.

Também é possível separar uma parcela do conjunto de treinamento para *validação*, a avaliação em tempo real da rede neural em um conjunto de dados o qual não é utilizado para seu treinamento.


```python
# history = model.fit(X_train, y_train_onehot, epochs=500)
history = model.fit(X_train, y_train_onehot, validation_split=.1, epochs=500)
```

    Epoch 1/500
    4/4 [==============================] - 0s 43ms/step - loss: 1.1317 - categorical_accuracy: 0.3333 - val_loss: 1.3079 - val_categorical_accuracy: 0.0833
    Epoch 2/500
    4/4 [==============================] - 0s 5ms/step - loss: 1.1113 - categorical_accuracy: 0.3333 - val_loss: 1.2650 - val_categorical_accuracy: 0.0833
    Epoch 3/500
    4/4 [==============================] - 0s 5ms/step - loss: 1.0922 - categorical_accuracy: 0.3333 - val_loss: 1.2253 - val_categorical_accuracy: 0.0833
    Epoch 4/500
    4/4 [==============================] - 0s 5ms/step - loss: 1.0736 - categorical_accuracy: 0.3426 - val_loss: 1.1882 - val_categorical_accuracy: 0.1667
    Epoch 5/500
    4/4 [==============================] - 0s 6ms/step - loss: 1.0576 - categorical_accuracy: 0.5463 - val_loss: 1.1543 - val_categorical_accuracy: 0.5000
    Epoch 6/500
    4/4 [==============================] - 0s 5ms/step - loss: 1.0394 - categorical_accuracy: 0.6944 - val_loss: 1.1237 - val_categorical_accuracy: 0.5000
    Epoch 7/500
    4/4 [==============================] - 0s 6ms/step - loss: 1.0228 - categorical_accuracy: 0.6944 - val_loss: 1.0926 - val_categorical_accuracy: 0.5000
    Epoch 8/500
    4/4 [==============================] - 0s 6ms/step - loss: 1.0059 - categorical_accuracy: 0.6944 - val_loss: 1.0622 - val_categorical_accuracy: 0.5000
    Epoch 9/500
    4/4 [==============================] - 0s 5ms/step - loss: 0.9860 - categorical_accuracy: 0.6852 - val_loss: 1.0361 - val_categorical_accuracy: 0.5000
    Epoch 10/500
    4/4 [==============================] - 0s 5ms/step - loss: 0.9675 - categorical_accuracy: 0.6852 - val_loss: 1.0084 - val_categorical_accuracy: 0.5000
    Epoch 11/500
    4/4 [==============================] - 0s 5ms/step - loss: 0.9480 - categorical_accuracy: 0.6852 - val_loss: 0.9814 - val_categorical_accuracy: 0.5000
    Epoch 12/500
    4/4 [==============================] - 0s 6ms/step - loss: 0.9268 - categorical_accuracy: 0.6852 - val_loss: 0.9555 - val_categorical_accuracy: 0.5000
    Epoch 13/500
    4/4 [==============================] - 0s 5ms/step - loss: 0.9054 - categorical_accuracy: 0.6944 - val_loss: 0.9331 - val_categorical_accuracy: 0.5000
    Epoch 14/500
    4/4 [==============================] - 0s 6ms/step - loss: 0.8835 - categorical_accuracy: 0.7222 - val_loss: 0.9133 - val_categorical_accuracy: 0.5000
    Epoch 15/500
    4/4 [==============================] - 0s 5ms/step - loss: 0.8601 - categorical_accuracy: 0.7685 - val_loss: 0.8972 - val_categorical_accuracy: 0.5000
    Epoch 16/500
    4/4 [==============================] - 0s 6ms/step - loss: 0.8383 - categorical_accuracy: 0.7685 - val_loss: 0.8830 - val_categorical_accuracy: 0.5833
    Epoch 17/500
    4/4 [==============================] - 0s 5ms/step - loss: 0.8154 - categorical_accuracy: 0.8241 - val_loss: 0.8680 - val_categorical_accuracy: 0.5000
    Epoch 18/500
    4/4 [==============================] - 0s 7ms/step - loss: 0.7946 - categorical_accuracy: 0.8333 - val_loss: 0.8551 - val_categorical_accuracy: 0.6667
    Epoch 19/500
    4/4 [==============================] - 0s 6ms/step - loss: 0.7741 - categorical_accuracy: 0.8333 - val_loss: 0.8425 - val_categorical_accuracy: 0.7500
    Epoch 20/500
    4/4 [==============================] - 0s 6ms/step - loss: 0.7557 - categorical_accuracy: 0.8426 - val_loss: 0.8312 - val_categorical_accuracy: 0.8333
    Epoch 21/500
    4/4 [==============================] - 0s 5ms/step - loss: 0.7373 - categorical_accuracy: 0.8426 - val_loss: 0.8186 - val_categorical_accuracy: 0.8333
    Epoch 22/500
    4/4 [==============================] - 0s 6ms/step - loss: 0.7211 - categorical_accuracy: 0.8333 - val_loss: 0.8088 - val_categorical_accuracy: 0.8333
    Epoch 23/500
    4/4 [==============================] - 0s 5ms/step - loss: 0.7057 - categorical_accuracy: 0.8611 - val_loss: 0.7998 - val_categorical_accuracy: 0.8333
    Epoch 24/500
    4/4 [==============================] - 0s 5ms/step - loss: 0.6908 - categorical_accuracy: 0.8981 - val_loss: 0.7904 - val_categorical_accuracy: 0.8333
    Epoch 25/500
    4/4 [==============================] - 0s 5ms/step - loss: 0.6767 - categorical_accuracy: 0.9074 - val_loss: 0.7819 - val_categorical_accuracy: 0.8333
    Epoch 26/500
    4/4 [==============================] - 0s 5ms/step - loss: 0.6631 - categorical_accuracy: 0.9074 - val_loss: 0.7735 - val_categorical_accuracy: 0.8333
    Epoch 27/500
    4/4 [==============================] - 0s 5ms/step - loss: 0.6507 - categorical_accuracy: 0.9352 - val_loss: 0.7655 - val_categorical_accuracy: 0.6667
    Epoch 28/500
    4/4 [==============================] - 0s 5ms/step - loss: 0.6390 - categorical_accuracy: 0.9259 - val_loss: 0.7588 - val_categorical_accuracy: 0.5833
    Epoch 29/500
    4/4 [==============================] - 0s 6ms/step - loss: 0.6282 - categorical_accuracy: 0.8519 - val_loss: 0.7533 - val_categorical_accuracy: 0.5833
    Epoch 30/500
    4/4 [==============================] - 0s 5ms/step - loss: 0.6170 - categorical_accuracy: 0.7963 - val_loss: 0.7477 - val_categorical_accuracy: 0.5000
    Epoch 31/500
    4/4 [==============================] - 0s 5ms/step - loss: 0.6071 - categorical_accuracy: 0.7500 - val_loss: 0.7438 - val_categorical_accuracy: 0.5000
    Epoch 32/500
    4/4 [==============================] - 0s 6ms/step - loss: 0.5979 - categorical_accuracy: 0.7222 - val_loss: 0.7407 - val_categorical_accuracy: 0.5000
    Epoch 33/500
    4/4 [==============================] - 0s 6ms/step - loss: 0.5894 - categorical_accuracy: 0.7037 - val_loss: 0.7391 - val_categorical_accuracy: 0.5000
    Epoch 34/500
    4/4 [==============================] - 0s 5ms/step - loss: 0.5817 - categorical_accuracy: 0.7037 - val_loss: 0.7342 - val_categorical_accuracy: 0.5000
    Epoch 35/500
    4/4 [==============================] - 0s 5ms/step - loss: 0.5742 - categorical_accuracy: 0.7037 - val_loss: 0.7270 - val_categorical_accuracy: 0.5000
    Epoch 36/500
    4/4 [==============================] - 0s 5ms/step - loss: 0.5663 - categorical_accuracy: 0.7037 - val_loss: 0.7220 - val_categorical_accuracy: 0.5000
    Epoch 37/500
    4/4 [==============================] - 0s 6ms/step - loss: 0.5591 - categorical_accuracy: 0.7037 - val_loss: 0.7179 - val_categorical_accuracy: 0.5000
    Epoch 38/500
    4/4 [==============================] - 0s 6ms/step - loss: 0.5521 - categorical_accuracy: 0.7037 - val_loss: 0.7100 - val_categorical_accuracy: 0.5000
    Epoch 39/500
    4/4 [==============================] - 0s 5ms/step - loss: 0.5453 - categorical_accuracy: 0.7315 - val_loss: 0.7021 - val_categorical_accuracy: 0.5000
    Epoch 40/500
    4/4 [==============================] - 0s 6ms/step - loss: 0.5388 - categorical_accuracy: 0.7500 - val_loss: 0.6941 - val_categorical_accuracy: 0.5000
    Epoch 41/500
    4/4 [==============================] - 0s 5ms/step - loss: 0.5332 - categorical_accuracy: 0.7593 - val_loss: 0.6890 - val_categorical_accuracy: 0.5000
    Epoch 42/500
    4/4 [==============================] - 0s 5ms/step - loss: 0.5273 - categorical_accuracy: 0.7685 - val_loss: 0.6857 - val_categorical_accuracy: 0.5000
    Epoch 43/500
    4/4 [==============================] - 0s 5ms/step - loss: 0.5218 - categorical_accuracy: 0.7593 - val_loss: 0.6815 - val_categorical_accuracy: 0.5000
    Epoch 44/500
    4/4 [==============================] - 0s 5ms/step - loss: 0.5155 - categorical_accuracy: 0.7593 - val_loss: 0.6813 - val_categorical_accuracy: 0.5000
    Epoch 45/500
    4/4 [==============================] - 0s 6ms/step - loss: 0.5109 - categorical_accuracy: 0.7407 - val_loss: 0.6830 - val_categorical_accuracy: 0.5000
    Epoch 46/500
    4/4 [==============================] - 0s 6ms/step - loss: 0.5066 - categorical_accuracy: 0.7222 - val_loss: 0.6840 - val_categorical_accuracy: 0.5000
    Epoch 47/500
    4/4 [==============================] - 0s 6ms/step - loss: 0.5024 - categorical_accuracy: 0.7222 - val_loss: 0.6796 - val_categorical_accuracy: 0.5000
    Epoch 48/500
    4/4 [==============================] - 0s 6ms/step - loss: 0.4972 - categorical_accuracy: 0.7222 - val_loss: 0.6725 - val_categorical_accuracy: 0.5000
    Epoch 49/500
    4/4 [==============================] - 0s 6ms/step - loss: 0.4943 - categorical_accuracy: 0.7407 - val_loss: 0.6600 - val_categorical_accuracy: 0.5000
    Epoch 50/500
    4/4 [==============================] - 0s 6ms/step - loss: 0.4866 - categorical_accuracy: 0.7778 - val_loss: 0.6542 - val_categorical_accuracy: 0.5000
    Epoch 51/500
    4/4 [==============================] - 0s 6ms/step - loss: 0.4823 - categorical_accuracy: 0.8056 - val_loss: 0.6494 - val_categorical_accuracy: 0.5000
    Epoch 52/500
    4/4 [==============================] - 0s 5ms/step - loss: 0.4778 - categorical_accuracy: 0.8333 - val_loss: 0.6462 - val_categorical_accuracy: 0.5000
    Epoch 53/500
    4/4 [==============================] - 0s 5ms/step - loss: 0.4734 - categorical_accuracy: 0.8241 - val_loss: 0.6430 - val_categorical_accuracy: 0.5000
    Epoch 54/500
    4/4 [==============================] - 0s 6ms/step - loss: 0.4686 - categorical_accuracy: 0.8519 - val_loss: 0.6356 - val_categorical_accuracy: 0.6667
    Epoch 55/500
    4/4 [==============================] - 0s 6ms/step - loss: 0.4658 - categorical_accuracy: 0.9074 - val_loss: 0.6276 - val_categorical_accuracy: 0.6667
    Epoch 56/500
    4/4 [==============================] - 0s 5ms/step - loss: 0.4606 - categorical_accuracy: 0.9259 - val_loss: 0.6294 - val_categorical_accuracy: 0.6667
    Epoch 57/500
    4/4 [==============================] - 0s 5ms/step - loss: 0.4557 - categorical_accuracy: 0.8796 - val_loss: 0.6266 - val_categorical_accuracy: 0.6667
    Epoch 58/500
    4/4 [==============================] - 0s 5ms/step - loss: 0.4518 - categorical_accuracy: 0.8796 - val_loss: 0.6231 - val_categorical_accuracy: 0.6667
    Epoch 59/500
    4/4 [==============================] - 0s 5ms/step - loss: 0.4472 - categorical_accuracy: 0.8796 - val_loss: 0.6235 - val_categorical_accuracy: 0.6667
    Epoch 60/500
    4/4 [==============================] - 0s 6ms/step - loss: 0.4435 - categorical_accuracy: 0.8519 - val_loss: 0.6207 - val_categorical_accuracy: 0.6667
    Epoch 61/500
    4/4 [==============================] - 0s 6ms/step - loss: 0.4391 - categorical_accuracy: 0.8704 - val_loss: 0.6131 - val_categorical_accuracy: 0.6667
    Epoch 62/500
    4/4 [==============================] - 0s 7ms/step - loss: 0.4345 - categorical_accuracy: 0.8981 - val_loss: 0.6035 - val_categorical_accuracy: 0.6667
    Epoch 63/500
    4/4 [==============================] - 0s 6ms/step - loss: 0.4302 - categorical_accuracy: 0.9444 - val_loss: 0.5951 - val_categorical_accuracy: 0.6667
    Epoch 64/500
    4/4 [==============================] - 0s 5ms/step - loss: 0.4255 - categorical_accuracy: 0.9630 - val_loss: 0.5921 - val_categorical_accuracy: 0.6667
    Epoch 65/500
    4/4 [==============================] - 0s 6ms/step - loss: 0.4209 - categorical_accuracy: 0.9537 - val_loss: 0.5875 - val_categorical_accuracy: 0.6667
    Epoch 66/500
    4/4 [==============================] - 0s 5ms/step - loss: 0.4169 - categorical_accuracy: 0.9537 - val_loss: 0.5841 - val_categorical_accuracy: 0.6667
    Epoch 67/500
    4/4 [==============================] - 0s 5ms/step - loss: 0.4116 - categorical_accuracy: 0.9722 - val_loss: 0.5753 - val_categorical_accuracy: 0.6667
    Epoch 68/500
    4/4 [==============================] - 0s 5ms/step - loss: 0.4074 - categorical_accuracy: 0.9722 - val_loss: 0.5687 - val_categorical_accuracy: 0.6667
    Epoch 69/500
    4/4 [==============================] - 0s 5ms/step - loss: 0.4031 - categorical_accuracy: 0.9722 - val_loss: 0.5664 - val_categorical_accuracy: 0.6667
    Epoch 70/500
    4/4 [==============================] - 0s 5ms/step - loss: 0.3977 - categorical_accuracy: 0.9722 - val_loss: 0.5656 - val_categorical_accuracy: 0.6667
    Epoch 71/500
    4/4 [==============================] - 0s 5ms/step - loss: 0.3928 - categorical_accuracy: 0.9722 - val_loss: 0.5674 - val_categorical_accuracy: 0.7500
    Epoch 72/500
    4/4 [==============================] - 0s 5ms/step - loss: 0.3891 - categorical_accuracy: 0.9352 - val_loss: 0.5693 - val_categorical_accuracy: 0.6667
    Epoch 73/500
    4/4 [==============================] - 0s 5ms/step - loss: 0.3844 - categorical_accuracy: 0.9444 - val_loss: 0.5588 - val_categorical_accuracy: 0.7500
    Epoch 74/500
    4/4 [==============================] - 0s 5ms/step - loss: 0.3788 - categorical_accuracy: 0.9722 - val_loss: 0.5491 - val_categorical_accuracy: 0.6667
    Epoch 75/500
    4/4 [==============================] - 0s 6ms/step - loss: 0.3740 - categorical_accuracy: 0.9722 - val_loss: 0.5430 - val_categorical_accuracy: 0.6667
    Epoch 76/500
    4/4 [==============================] - 0s 5ms/step - loss: 0.3687 - categorical_accuracy: 0.9722 - val_loss: 0.5357 - val_categorical_accuracy: 0.6667
    Epoch 77/500
    4/4 [==============================] - 0s 6ms/step - loss: 0.3644 - categorical_accuracy: 0.9722 - val_loss: 0.5294 - val_categorical_accuracy: 0.6667
    Epoch 78/500
    4/4 [==============================] - 0s 5ms/step - loss: 0.3606 - categorical_accuracy: 0.9722 - val_loss: 0.5352 - val_categorical_accuracy: 0.6667
    Epoch 79/500
    4/4 [==============================] - 0s 6ms/step - loss: 0.3540 - categorical_accuracy: 0.9722 - val_loss: 0.5227 - val_categorical_accuracy: 0.6667
    Epoch 80/500
    4/4 [==============================] - 0s 6ms/step - loss: 0.3485 - categorical_accuracy: 0.9722 - val_loss: 0.5116 - val_categorical_accuracy: 0.8333
    Epoch 81/500
    4/4 [==============================] - 0s 5ms/step - loss: 0.3442 - categorical_accuracy: 0.9815 - val_loss: 0.5060 - val_categorical_accuracy: 0.8333
    Epoch 82/500
    4/4 [==============================] - 0s 5ms/step - loss: 0.3379 - categorical_accuracy: 0.9722 - val_loss: 0.5097 - val_categorical_accuracy: 0.7500
    Epoch 83/500
    4/4 [==============================] - 0s 5ms/step - loss: 0.3343 - categorical_accuracy: 0.9722 - val_loss: 0.5212 - val_categorical_accuracy: 0.7500
    Epoch 84/500
    4/4 [==============================] - 0s 5ms/step - loss: 0.3309 - categorical_accuracy: 0.9722 - val_loss: 0.5057 - val_categorical_accuracy: 0.6667
    Epoch 85/500
    4/4 [==============================] - 0s 6ms/step - loss: 0.3235 - categorical_accuracy: 0.9722 - val_loss: 0.4863 - val_categorical_accuracy: 0.8333
    Epoch 86/500
    4/4 [==============================] - 0s 5ms/step - loss: 0.3182 - categorical_accuracy: 0.9815 - val_loss: 0.4751 - val_categorical_accuracy: 0.9167
    Epoch 87/500
    4/4 [==============================] - 0s 5ms/step - loss: 0.3162 - categorical_accuracy: 0.9722 - val_loss: 0.4759 - val_categorical_accuracy: 0.9167
    Epoch 88/500
    4/4 [==============================] - 0s 5ms/step - loss: 0.3079 - categorical_accuracy: 0.9815 - val_loss: 0.4638 - val_categorical_accuracy: 0.9167
    Epoch 89/500
    4/4 [==============================] - 0s 7ms/step - loss: 0.3038 - categorical_accuracy: 0.9907 - val_loss: 0.4601 - val_categorical_accuracy: 0.9167
    Epoch 90/500
    4/4 [==============================] - 0s 5ms/step - loss: 0.2981 - categorical_accuracy: 0.9907 - val_loss: 0.4602 - val_categorical_accuracy: 0.9167
    Epoch 91/500
    4/4 [==============================] - 0s 5ms/step - loss: 0.2939 - categorical_accuracy: 0.9722 - val_loss: 0.4661 - val_categorical_accuracy: 0.7500
    Epoch 92/500
    4/4 [==============================] - 0s 5ms/step - loss: 0.2886 - categorical_accuracy: 0.9722 - val_loss: 0.4559 - val_categorical_accuracy: 0.9167
    Epoch 93/500
    4/4 [==============================] - 0s 6ms/step - loss: 0.2830 - categorical_accuracy: 0.9815 - val_loss: 0.4437 - val_categorical_accuracy: 0.9167
    Epoch 94/500
    4/4 [==============================] - 0s 5ms/step - loss: 0.2799 - categorical_accuracy: 0.9907 - val_loss: 0.4293 - val_categorical_accuracy: 0.9167
    Epoch 95/500
    4/4 [==============================] - 0s 5ms/step - loss: 0.2744 - categorical_accuracy: 0.9907 - val_loss: 0.4331 - val_categorical_accuracy: 0.9167
    Epoch 96/500
    4/4 [==============================] - 0s 5ms/step - loss: 0.2680 - categorical_accuracy: 0.9907 - val_loss: 0.4477 - val_categorical_accuracy: 0.7500
    Epoch 97/500
    4/4 [==============================] - 0s 6ms/step - loss: 0.2658 - categorical_accuracy: 0.9722 - val_loss: 0.4446 - val_categorical_accuracy: 0.7500
    Epoch 98/500
    4/4 [==============================] - 0s 5ms/step - loss: 0.2625 - categorical_accuracy: 0.9722 - val_loss: 0.4396 - val_categorical_accuracy: 0.7500
    Epoch 99/500
    4/4 [==============================] - 0s 5ms/step - loss: 0.2566 - categorical_accuracy: 0.9722 - val_loss: 0.4164 - val_categorical_accuracy: 0.9167
    Epoch 100/500
    4/4 [==============================] - 0s 5ms/step - loss: 0.2500 - categorical_accuracy: 0.9907 - val_loss: 0.4038 - val_categorical_accuracy: 0.9167
    Epoch 101/500
    4/4 [==============================] - 0s 5ms/step - loss: 0.2494 - categorical_accuracy: 0.9815 - val_loss: 0.3935 - val_categorical_accuracy: 0.9167
    Epoch 102/500
    4/4 [==============================] - 0s 5ms/step - loss: 0.2444 - categorical_accuracy: 0.9815 - val_loss: 0.4030 - val_categorical_accuracy: 0.9167
    Epoch 103/500
    4/4 [==============================] - 0s 5ms/step - loss: 0.2381 - categorical_accuracy: 0.9815 - val_loss: 0.4146 - val_categorical_accuracy: 0.8333
    Epoch 104/500
    4/4 [==============================] - 0s 5ms/step - loss: 0.2354 - categorical_accuracy: 0.9722 - val_loss: 0.4104 - val_categorical_accuracy: 0.9167
    Epoch 105/500
    4/4 [==============================] - 0s 5ms/step - loss: 0.2315 - categorical_accuracy: 0.9815 - val_loss: 0.3966 - val_categorical_accuracy: 0.9167
    Epoch 106/500
    4/4 [==============================] - 0s 5ms/step - loss: 0.2255 - categorical_accuracy: 0.9907 - val_loss: 0.3759 - val_categorical_accuracy: 0.9167
    Epoch 107/500
    4/4 [==============================] - 0s 5ms/step - loss: 0.2232 - categorical_accuracy: 0.9815 - val_loss: 0.3724 - val_categorical_accuracy: 0.9167
    Epoch 108/500
    4/4 [==============================] - 0s 5ms/step - loss: 0.2185 - categorical_accuracy: 0.9815 - val_loss: 0.3735 - val_categorical_accuracy: 0.9167
    Epoch 109/500
    4/4 [==============================] - 0s 5ms/step - loss: 0.2160 - categorical_accuracy: 0.9907 - val_loss: 0.3866 - val_categorical_accuracy: 0.9167
    Epoch 110/500
    4/4 [==============================] - 0s 5ms/step - loss: 0.2118 - categorical_accuracy: 0.9815 - val_loss: 0.3810 - val_categorical_accuracy: 0.9167
    Epoch 111/500
    4/4 [==============================] - 0s 5ms/step - loss: 0.2081 - categorical_accuracy: 0.9815 - val_loss: 0.3647 - val_categorical_accuracy: 0.9167
    Epoch 112/500
    4/4 [==============================] - 0s 6ms/step - loss: 0.2034 - categorical_accuracy: 0.9907 - val_loss: 0.3601 - val_categorical_accuracy: 0.9167
    Epoch 113/500
    4/4 [==============================] - 0s 5ms/step - loss: 0.2005 - categorical_accuracy: 0.9907 - val_loss: 0.3699 - val_categorical_accuracy: 0.9167
    Epoch 114/500
    4/4 [==============================] - 0s 5ms/step - loss: 0.1967 - categorical_accuracy: 0.9907 - val_loss: 0.3621 - val_categorical_accuracy: 0.9167
    Epoch 115/500
    4/4 [==============================] - 0s 5ms/step - loss: 0.1932 - categorical_accuracy: 0.9907 - val_loss: 0.3550 - val_categorical_accuracy: 0.9167
    Epoch 116/500
    4/4 [==============================] - 0s 5ms/step - loss: 0.1888 - categorical_accuracy: 0.9907 - val_loss: 0.3446 - val_categorical_accuracy: 0.9167
    Epoch 117/500
    4/4 [==============================] - 0s 5ms/step - loss: 0.1876 - categorical_accuracy: 0.9907 - val_loss: 0.3396 - val_categorical_accuracy: 0.9167
    Epoch 118/500
    4/4 [==============================] - 0s 5ms/step - loss: 0.1834 - categorical_accuracy: 0.9907 - val_loss: 0.3387 - val_categorical_accuracy: 0.9167
    Epoch 119/500
    4/4 [==============================] - 0s 5ms/step - loss: 0.1814 - categorical_accuracy: 0.9907 - val_loss: 0.3433 - val_categorical_accuracy: 0.9167
    Epoch 120/500
    4/4 [==============================] - 0s 5ms/step - loss: 0.1775 - categorical_accuracy: 0.9907 - val_loss: 0.3320 - val_categorical_accuracy: 0.9167
    Epoch 121/500
    4/4 [==============================] - 0s 6ms/step - loss: 0.1746 - categorical_accuracy: 0.9907 - val_loss: 0.3295 - val_categorical_accuracy: 0.9167
    Epoch 122/500
    4/4 [==============================] - 0s 5ms/step - loss: 0.1709 - categorical_accuracy: 0.9907 - val_loss: 0.3290 - val_categorical_accuracy: 0.9167
    Epoch 123/500
    4/4 [==============================] - 0s 5ms/step - loss: 0.1681 - categorical_accuracy: 0.9907 - val_loss: 0.3311 - val_categorical_accuracy: 0.9167
    Epoch 124/500
    4/4 [==============================] - 0s 5ms/step - loss: 0.1659 - categorical_accuracy: 0.9907 - val_loss: 0.3371 - val_categorical_accuracy: 0.9167
    Epoch 125/500
    4/4 [==============================] - 0s 5ms/step - loss: 0.1630 - categorical_accuracy: 0.9907 - val_loss: 0.3243 - val_categorical_accuracy: 0.9167
    Epoch 126/500
    4/4 [==============================] - 0s 8ms/step - loss: 0.1587 - categorical_accuracy: 0.9907 - val_loss: 0.3140 - val_categorical_accuracy: 0.9167
    Epoch 127/500
    4/4 [==============================] - 0s 6ms/step - loss: 0.1591 - categorical_accuracy: 0.9815 - val_loss: 0.3091 - val_categorical_accuracy: 0.9167
    Epoch 128/500
    4/4 [==============================] - 0s 6ms/step - loss: 0.1569 - categorical_accuracy: 0.9907 - val_loss: 0.3095 - val_categorical_accuracy: 0.9167
    Epoch 129/500
    4/4 [==============================] - 0s 5ms/step - loss: 0.1525 - categorical_accuracy: 0.9907 - val_loss: 0.3071 - val_categorical_accuracy: 0.9167
    Epoch 130/500
    4/4 [==============================] - 0s 5ms/step - loss: 0.1500 - categorical_accuracy: 0.9907 - val_loss: 0.3099 - val_categorical_accuracy: 0.9167
    Epoch 131/500
    4/4 [==============================] - 0s 5ms/step - loss: 0.1473 - categorical_accuracy: 0.9907 - val_loss: 0.3097 - val_categorical_accuracy: 0.9167
    Epoch 132/500
    4/4 [==============================] - 0s 5ms/step - loss: 0.1450 - categorical_accuracy: 0.9907 - val_loss: 0.3078 - val_categorical_accuracy: 0.9167
    Epoch 133/500
    4/4 [==============================] - 0s 5ms/step - loss: 0.1432 - categorical_accuracy: 0.9907 - val_loss: 0.3019 - val_categorical_accuracy: 0.9167
    Epoch 134/500
    4/4 [==============================] - 0s 5ms/step - loss: 0.1403 - categorical_accuracy: 0.9907 - val_loss: 0.2978 - val_categorical_accuracy: 0.9167
    Epoch 135/500
    4/4 [==============================] - 0s 5ms/step - loss: 0.1389 - categorical_accuracy: 0.9907 - val_loss: 0.2943 - val_categorical_accuracy: 0.9167
    Epoch 136/500
    4/4 [==============================] - 0s 5ms/step - loss: 0.1362 - categorical_accuracy: 0.9907 - val_loss: 0.2938 - val_categorical_accuracy: 0.9167
    Epoch 137/500
    4/4 [==============================] - 0s 6ms/step - loss: 0.1346 - categorical_accuracy: 0.9907 - val_loss: 0.2999 - val_categorical_accuracy: 0.9167
    Epoch 138/500
    4/4 [==============================] - 0s 6ms/step - loss: 0.1330 - categorical_accuracy: 0.9907 - val_loss: 0.2998 - val_categorical_accuracy: 0.9167
    Epoch 139/500
    4/4 [==============================] - 0s 6ms/step - loss: 0.1302 - categorical_accuracy: 0.9907 - val_loss: 0.2886 - val_categorical_accuracy: 0.9167
    Epoch 140/500
    4/4 [==============================] - 0s 7ms/step - loss: 0.1288 - categorical_accuracy: 0.9907 - val_loss: 0.2833 - val_categorical_accuracy: 0.9167
    Epoch 141/500
    4/4 [==============================] - 0s 5ms/step - loss: 0.1271 - categorical_accuracy: 0.9907 - val_loss: 0.2840 - val_categorical_accuracy: 0.9167
    Epoch 142/500
    4/4 [==============================] - 0s 5ms/step - loss: 0.1233 - categorical_accuracy: 0.9907 - val_loss: 0.2935 - val_categorical_accuracy: 0.9167
    Epoch 143/500
    4/4 [==============================] - 0s 6ms/step - loss: 0.1248 - categorical_accuracy: 0.9907 - val_loss: 0.2964 - val_categorical_accuracy: 0.9167
    Epoch 144/500
    4/4 [==============================] - 0s 5ms/step - loss: 0.1230 - categorical_accuracy: 0.9907 - val_loss: 0.2877 - val_categorical_accuracy: 0.9167
    Epoch 145/500
    4/4 [==============================] - 0s 5ms/step - loss: 0.1188 - categorical_accuracy: 0.9907 - val_loss: 0.2761 - val_categorical_accuracy: 0.9167
    Epoch 146/500
    4/4 [==============================] - 0s 5ms/step - loss: 0.1182 - categorical_accuracy: 0.9907 - val_loss: 0.2743 - val_categorical_accuracy: 0.9167
    Epoch 147/500
    4/4 [==============================] - 0s 5ms/step - loss: 0.1176 - categorical_accuracy: 0.9815 - val_loss: 0.2734 - val_categorical_accuracy: 0.9167
    Epoch 148/500
    4/4 [==============================] - 0s 5ms/step - loss: 0.1139 - categorical_accuracy: 0.9907 - val_loss: 0.2772 - val_categorical_accuracy: 0.9167
    Epoch 149/500
    4/4 [==============================] - 0s 5ms/step - loss: 0.1139 - categorical_accuracy: 0.9907 - val_loss: 0.2865 - val_categorical_accuracy: 0.9167
    Epoch 150/500
    4/4 [==============================] - 0s 5ms/step - loss: 0.1160 - categorical_accuracy: 0.9907 - val_loss: 0.2719 - val_categorical_accuracy: 0.9167
    Epoch 151/500
    4/4 [==============================] - 0s 5ms/step - loss: 0.1090 - categorical_accuracy: 0.9907 - val_loss: 0.2707 - val_categorical_accuracy: 0.9167
    Epoch 152/500
    4/4 [==============================] - 0s 5ms/step - loss: 0.1075 - categorical_accuracy: 0.9907 - val_loss: 0.2701 - val_categorical_accuracy: 0.9167
    Epoch 153/500
    4/4 [==============================] - 0s 5ms/step - loss: 0.1064 - categorical_accuracy: 0.9907 - val_loss: 0.2678 - val_categorical_accuracy: 0.9167
    Epoch 154/500
    4/4 [==============================] - 0s 5ms/step - loss: 0.1044 - categorical_accuracy: 0.9907 - val_loss: 0.2643 - val_categorical_accuracy: 0.9167
    Epoch 155/500
    4/4 [==============================] - 0s 5ms/step - loss: 0.1045 - categorical_accuracy: 0.9907 - val_loss: 0.2633 - val_categorical_accuracy: 0.9167
    Epoch 156/500
    4/4 [==============================] - 0s 5ms/step - loss: 0.1024 - categorical_accuracy: 0.9907 - val_loss: 0.2632 - val_categorical_accuracy: 0.9167
    Epoch 157/500
    4/4 [==============================] - 0s 5ms/step - loss: 0.1002 - categorical_accuracy: 0.9907 - val_loss: 0.2665 - val_categorical_accuracy: 0.9167
    Epoch 158/500
    4/4 [==============================] - 0s 5ms/step - loss: 0.1015 - categorical_accuracy: 0.9907 - val_loss: 0.2707 - val_categorical_accuracy: 0.9167
    Epoch 159/500
    4/4 [==============================] - 0s 5ms/step - loss: 0.0994 - categorical_accuracy: 0.9907 - val_loss: 0.2611 - val_categorical_accuracy: 0.9167
    Epoch 160/500
    4/4 [==============================] - 0s 5ms/step - loss: 0.0968 - categorical_accuracy: 0.9907 - val_loss: 0.2586 - val_categorical_accuracy: 0.9167
    Epoch 161/500
    4/4 [==============================] - 0s 5ms/step - loss: 0.0969 - categorical_accuracy: 0.9907 - val_loss: 0.2585 - val_categorical_accuracy: 0.9167
    Epoch 162/500
    4/4 [==============================] - 0s 7ms/step - loss: 0.0942 - categorical_accuracy: 0.9907 - val_loss: 0.2569 - val_categorical_accuracy: 0.9167
    Epoch 163/500
    4/4 [==============================] - 0s 5ms/step - loss: 0.0937 - categorical_accuracy: 0.9907 - val_loss: 0.2562 - val_categorical_accuracy: 0.9167
    Epoch 164/500
    4/4 [==============================] - 0s 5ms/step - loss: 0.0923 - categorical_accuracy: 0.9907 - val_loss: 0.2589 - val_categorical_accuracy: 0.9167
    Epoch 165/500
    4/4 [==============================] - 0s 5ms/step - loss: 0.0913 - categorical_accuracy: 0.9907 - val_loss: 0.2569 - val_categorical_accuracy: 0.9167
    Epoch 166/500
    4/4 [==============================] - 0s 5ms/step - loss: 0.0901 - categorical_accuracy: 0.9907 - val_loss: 0.2541 - val_categorical_accuracy: 0.9167
    Epoch 167/500
    4/4 [==============================] - 0s 5ms/step - loss: 0.0885 - categorical_accuracy: 0.9907 - val_loss: 0.2531 - val_categorical_accuracy: 0.9167
    Epoch 168/500
    4/4 [==============================] - 0s 5ms/step - loss: 0.0895 - categorical_accuracy: 0.9907 - val_loss: 0.2522 - val_categorical_accuracy: 0.9167
    Epoch 169/500
    4/4 [==============================] - 0s 5ms/step - loss: 0.0856 - categorical_accuracy: 0.9907 - val_loss: 0.2537 - val_categorical_accuracy: 0.9167
    Epoch 170/500
    4/4 [==============================] - 0s 5ms/step - loss: 0.0867 - categorical_accuracy: 0.9907 - val_loss: 0.2651 - val_categorical_accuracy: 0.9167
    Epoch 171/500
    4/4 [==============================] - 0s 5ms/step - loss: 0.0885 - categorical_accuracy: 1.0000 - val_loss: 0.2563 - val_categorical_accuracy: 0.9167
    Epoch 172/500
    4/4 [==============================] - 0s 5ms/step - loss: 0.0854 - categorical_accuracy: 0.9907 - val_loss: 0.2496 - val_categorical_accuracy: 0.9167
    Epoch 173/500
    4/4 [==============================] - 0s 5ms/step - loss: 0.0849 - categorical_accuracy: 0.9907 - val_loss: 0.2496 - val_categorical_accuracy: 0.9167
    Epoch 174/500
    4/4 [==============================] - 0s 6ms/step - loss: 0.0843 - categorical_accuracy: 0.9907 - val_loss: 0.2474 - val_categorical_accuracy: 0.9167
    Epoch 175/500
    4/4 [==============================] - 0s 10ms/step - loss: 0.0810 - categorical_accuracy: 0.9907 - val_loss: 0.2469 - val_categorical_accuracy: 0.9167
    Epoch 176/500
    4/4 [==============================] - 0s 5ms/step - loss: 0.0806 - categorical_accuracy: 0.9907 - val_loss: 0.2459 - val_categorical_accuracy: 0.9167
    Epoch 177/500
    4/4 [==============================] - 0s 5ms/step - loss: 0.0785 - categorical_accuracy: 0.9907 - val_loss: 0.2459 - val_categorical_accuracy: 0.9167
    Epoch 178/500
    4/4 [==============================] - 0s 6ms/step - loss: 0.0779 - categorical_accuracy: 0.9907 - val_loss: 0.2469 - val_categorical_accuracy: 0.9167
    Epoch 179/500
    4/4 [==============================] - 0s 6ms/step - loss: 0.0777 - categorical_accuracy: 0.9907 - val_loss: 0.2463 - val_categorical_accuracy: 0.9167
    Epoch 180/500
    4/4 [==============================] - 0s 6ms/step - loss: 0.0763 - categorical_accuracy: 0.9907 - val_loss: 0.2427 - val_categorical_accuracy: 0.9167
    Epoch 181/500
    4/4 [==============================] - 0s 5ms/step - loss: 0.0741 - categorical_accuracy: 0.9907 - val_loss: 0.2431 - val_categorical_accuracy: 0.9167
    Epoch 182/500
    4/4 [==============================] - 0s 5ms/step - loss: 0.0750 - categorical_accuracy: 0.9907 - val_loss: 0.2448 - val_categorical_accuracy: 0.9167
    Epoch 183/500
    4/4 [==============================] - 0s 5ms/step - loss: 0.0752 - categorical_accuracy: 0.9907 - val_loss: 0.2419 - val_categorical_accuracy: 0.9167
    Epoch 184/500
    4/4 [==============================] - 0s 5ms/step - loss: 0.0729 - categorical_accuracy: 0.9907 - val_loss: 0.2406 - val_categorical_accuracy: 0.9167
    Epoch 185/500
    4/4 [==============================] - 0s 5ms/step - loss: 0.0714 - categorical_accuracy: 0.9907 - val_loss: 0.2407 - val_categorical_accuracy: 0.9167
    Epoch 186/500
    4/4 [==============================] - 0s 5ms/step - loss: 0.0709 - categorical_accuracy: 0.9907 - val_loss: 0.2408 - val_categorical_accuracy: 0.9167
    Epoch 187/500
    4/4 [==============================] - 0s 5ms/step - loss: 0.0704 - categorical_accuracy: 0.9907 - val_loss: 0.2395 - val_categorical_accuracy: 0.9167
    Epoch 188/500
    4/4 [==============================] - 0s 5ms/step - loss: 0.0694 - categorical_accuracy: 0.9907 - val_loss: 0.2386 - val_categorical_accuracy: 0.9167
    Epoch 189/500
    4/4 [==============================] - 0s 5ms/step - loss: 0.0684 - categorical_accuracy: 0.9907 - val_loss: 0.2382 - val_categorical_accuracy: 0.9167
    Epoch 190/500
    4/4 [==============================] - 0s 6ms/step - loss: 0.0677 - categorical_accuracy: 0.9907 - val_loss: 0.2382 - val_categorical_accuracy: 0.9167
    Epoch 191/500
    4/4 [==============================] - 0s 5ms/step - loss: 0.0672 - categorical_accuracy: 0.9907 - val_loss: 0.2383 - val_categorical_accuracy: 0.9167
    Epoch 192/500
    4/4 [==============================] - 0s 5ms/step - loss: 0.0666 - categorical_accuracy: 0.9907 - val_loss: 0.2400 - val_categorical_accuracy: 0.9167
    Epoch 193/500
    4/4 [==============================] - 0s 5ms/step - loss: 0.0681 - categorical_accuracy: 0.9907 - val_loss: 0.2409 - val_categorical_accuracy: 0.9167
    Epoch 194/500
    4/4 [==============================] - 0s 5ms/step - loss: 0.0660 - categorical_accuracy: 0.9907 - val_loss: 0.2369 - val_categorical_accuracy: 0.9167
    Epoch 195/500
    4/4 [==============================] - 0s 5ms/step - loss: 0.0639 - categorical_accuracy: 0.9907 - val_loss: 0.2368 - val_categorical_accuracy: 0.9167
    Epoch 196/500
    4/4 [==============================] - 0s 5ms/step - loss: 0.0641 - categorical_accuracy: 0.9907 - val_loss: 0.2365 - val_categorical_accuracy: 0.9167
    Epoch 197/500
    4/4 [==============================] - 0s 6ms/step - loss: 0.0639 - categorical_accuracy: 0.9907 - val_loss: 0.2361 - val_categorical_accuracy: 0.9167
    Epoch 198/500
    4/4 [==============================] - 0s 7ms/step - loss: 0.0625 - categorical_accuracy: 0.9907 - val_loss: 0.2373 - val_categorical_accuracy: 0.9167
    Epoch 199/500
    4/4 [==============================] - 0s 5ms/step - loss: 0.0624 - categorical_accuracy: 0.9907 - val_loss: 0.2366 - val_categorical_accuracy: 0.9167
    Epoch 200/500
    4/4 [==============================] - 0s 5ms/step - loss: 0.0622 - categorical_accuracy: 0.9907 - val_loss: 0.2343 - val_categorical_accuracy: 0.9167
    Epoch 201/500
    4/4 [==============================] - 0s 6ms/step - loss: 0.0610 - categorical_accuracy: 0.9907 - val_loss: 0.2338 - val_categorical_accuracy: 0.9167
    Epoch 202/500
    4/4 [==============================] - 0s 5ms/step - loss: 0.0604 - categorical_accuracy: 0.9907 - val_loss: 0.2340 - val_categorical_accuracy: 0.9167
    Epoch 203/500
    4/4 [==============================] - 0s 6ms/step - loss: 0.0599 - categorical_accuracy: 0.9907 - val_loss: 0.2334 - val_categorical_accuracy: 0.9167
    Epoch 204/500
    4/4 [==============================] - 0s 5ms/step - loss: 0.0588 - categorical_accuracy: 0.9907 - val_loss: 0.2342 - val_categorical_accuracy: 0.9167
    Epoch 205/500
    4/4 [==============================] - 0s 5ms/step - loss: 0.0584 - categorical_accuracy: 0.9907 - val_loss: 0.2352 - val_categorical_accuracy: 0.9167
    Epoch 206/500
    4/4 [==============================] - 0s 5ms/step - loss: 0.0586 - categorical_accuracy: 0.9907 - val_loss: 0.2354 - val_categorical_accuracy: 0.9167
    Epoch 207/500
    4/4 [==============================] - 0s 5ms/step - loss: 0.0575 - categorical_accuracy: 0.9907 - val_loss: 0.2324 - val_categorical_accuracy: 0.9167
    Epoch 208/500
    4/4 [==============================] - 0s 5ms/step - loss: 0.0558 - categorical_accuracy: 0.9907 - val_loss: 0.2314 - val_categorical_accuracy: 0.9167
    Epoch 209/500
    4/4 [==============================] - 0s 5ms/step - loss: 0.0569 - categorical_accuracy: 1.0000 - val_loss: 0.2322 - val_categorical_accuracy: 0.9167
    Epoch 210/500
    4/4 [==============================] - 0s 6ms/step - loss: 0.0563 - categorical_accuracy: 1.0000 - val_loss: 0.2304 - val_categorical_accuracy: 0.9167
    Epoch 211/500
    4/4 [==============================] - 0s 6ms/step - loss: 0.0552 - categorical_accuracy: 0.9907 - val_loss: 0.2335 - val_categorical_accuracy: 0.9167
    Epoch 212/500
    4/4 [==============================] - 0s 6ms/step - loss: 0.0559 - categorical_accuracy: 0.9907 - val_loss: 0.2337 - val_categorical_accuracy: 0.9167
    Epoch 213/500
    4/4 [==============================] - 0s 5ms/step - loss: 0.0551 - categorical_accuracy: 0.9907 - val_loss: 0.2299 - val_categorical_accuracy: 0.9167
    Epoch 214/500
    4/4 [==============================] - 0s 5ms/step - loss: 0.0534 - categorical_accuracy: 0.9907 - val_loss: 0.2300 - val_categorical_accuracy: 0.9167
    Epoch 215/500
    4/4 [==============================] - 0s 5ms/step - loss: 0.0529 - categorical_accuracy: 0.9907 - val_loss: 0.2302 - val_categorical_accuracy: 0.9167
    Epoch 216/500
    4/4 [==============================] - 0s 5ms/step - loss: 0.0531 - categorical_accuracy: 0.9907 - val_loss: 0.2292 - val_categorical_accuracy: 0.9167
    Epoch 217/500
    4/4 [==============================] - 0s 5ms/step - loss: 0.0518 - categorical_accuracy: 0.9907 - val_loss: 0.2305 - val_categorical_accuracy: 0.9167
    Epoch 218/500
    4/4 [==============================] - 0s 6ms/step - loss: 0.0518 - categorical_accuracy: 0.9907 - val_loss: 0.2317 - val_categorical_accuracy: 0.9167
    Epoch 219/500
    4/4 [==============================] - 0s 5ms/step - loss: 0.0516 - categorical_accuracy: 0.9907 - val_loss: 0.2297 - val_categorical_accuracy: 0.9167
    Epoch 220/500
    4/4 [==============================] - 0s 6ms/step - loss: 0.0503 - categorical_accuracy: 0.9907 - val_loss: 0.2314 - val_categorical_accuracy: 0.9167
    Epoch 221/500
    4/4 [==============================] - 0s 5ms/step - loss: 0.0503 - categorical_accuracy: 0.9907 - val_loss: 0.2317 - val_categorical_accuracy: 0.9167
    Epoch 222/500
    4/4 [==============================] - 0s 5ms/step - loss: 0.0498 - categorical_accuracy: 0.9907 - val_loss: 0.2303 - val_categorical_accuracy: 0.9167
    Epoch 223/500
    4/4 [==============================] - 0s 6ms/step - loss: 0.0493 - categorical_accuracy: 0.9907 - val_loss: 0.2291 - val_categorical_accuracy: 0.9167
    Epoch 224/500
    4/4 [==============================] - 0s 5ms/step - loss: 0.0491 - categorical_accuracy: 0.9907 - val_loss: 0.2293 - val_categorical_accuracy: 0.9167
    Epoch 225/500
    4/4 [==============================] - 0s 5ms/step - loss: 0.0483 - categorical_accuracy: 1.0000 - val_loss: 0.2288 - val_categorical_accuracy: 0.9167
    Epoch 226/500
    4/4 [==============================] - 0s 5ms/step - loss: 0.0501 - categorical_accuracy: 0.9907 - val_loss: 0.2289 - val_categorical_accuracy: 0.9167
    Epoch 227/500
    4/4 [==============================] - 0s 5ms/step - loss: 0.0479 - categorical_accuracy: 1.0000 - val_loss: 0.2330 - val_categorical_accuracy: 0.9167
    Epoch 228/500
    4/4 [==============================] - 0s 5ms/step - loss: 0.0502 - categorical_accuracy: 1.0000 - val_loss: 0.2291 - val_categorical_accuracy: 0.9167
    Epoch 229/500
    4/4 [==============================] - 0s 5ms/step - loss: 0.0465 - categorical_accuracy: 1.0000 - val_loss: 0.2342 - val_categorical_accuracy: 0.9167
    Epoch 230/500
    4/4 [==============================] - 0s 5ms/step - loss: 0.0485 - categorical_accuracy: 0.9907 - val_loss: 0.2393 - val_categorical_accuracy: 0.9167
    Epoch 231/500
    4/4 [==============================] - 0s 5ms/step - loss: 0.0466 - categorical_accuracy: 0.9907 - val_loss: 0.2275 - val_categorical_accuracy: 0.9167
    Epoch 232/500
    4/4 [==============================] - 0s 6ms/step - loss: 0.0456 - categorical_accuracy: 1.0000 - val_loss: 0.2310 - val_categorical_accuracy: 0.9167
    Epoch 233/500
    4/4 [==============================] - 0s 6ms/step - loss: 0.0478 - categorical_accuracy: 1.0000 - val_loss: 0.2274 - val_categorical_accuracy: 0.9167
    Epoch 234/500
    4/4 [==============================] - 0s 7ms/step - loss: 0.0447 - categorical_accuracy: 1.0000 - val_loss: 0.2280 - val_categorical_accuracy: 0.9167
    Epoch 235/500
    4/4 [==============================] - 0s 5ms/step - loss: 0.0434 - categorical_accuracy: 1.0000 - val_loss: 0.2325 - val_categorical_accuracy: 0.9167
    Epoch 236/500
    4/4 [==============================] - 0s 5ms/step - loss: 0.0452 - categorical_accuracy: 0.9907 - val_loss: 0.2369 - val_categorical_accuracy: 0.9167
    Epoch 237/500
    4/4 [==============================] - 0s 5ms/step - loss: 0.0445 - categorical_accuracy: 0.9907 - val_loss: 0.2274 - val_categorical_accuracy: 0.9167
    Epoch 238/500
    4/4 [==============================] - 0s 5ms/step - loss: 0.0427 - categorical_accuracy: 1.0000 - val_loss: 0.2262 - val_categorical_accuracy: 0.9167
    Epoch 239/500
    4/4 [==============================] - 0s 5ms/step - loss: 0.0429 - categorical_accuracy: 1.0000 - val_loss: 0.2262 - val_categorical_accuracy: 0.9167
    Epoch 240/500
    4/4 [==============================] - 0s 5ms/step - loss: 0.0420 - categorical_accuracy: 1.0000 - val_loss: 0.2289 - val_categorical_accuracy: 0.9167
    Epoch 241/500
    4/4 [==============================] - 0s 6ms/step - loss: 0.0439 - categorical_accuracy: 0.9907 - val_loss: 0.2415 - val_categorical_accuracy: 0.9167
    Epoch 242/500
    4/4 [==============================] - 0s 6ms/step - loss: 0.0443 - categorical_accuracy: 0.9907 - val_loss: 0.2357 - val_categorical_accuracy: 0.9167
    Epoch 243/500
    4/4 [==============================] - 0s 6ms/step - loss: 0.0414 - categorical_accuracy: 0.9907 - val_loss: 0.2268 - val_categorical_accuracy: 0.9167
    Epoch 244/500
    4/4 [==============================] - 0s 6ms/step - loss: 0.0409 - categorical_accuracy: 1.0000 - val_loss: 0.2253 - val_categorical_accuracy: 0.9167
    Epoch 245/500
    4/4 [==============================] - 0s 5ms/step - loss: 0.0406 - categorical_accuracy: 1.0000 - val_loss: 0.2265 - val_categorical_accuracy: 0.9167
    Epoch 246/500
    4/4 [==============================] - 0s 6ms/step - loss: 0.0401 - categorical_accuracy: 1.0000 - val_loss: 0.2323 - val_categorical_accuracy: 0.9167
    Epoch 247/500
    4/4 [==============================] - 0s 6ms/step - loss: 0.0404 - categorical_accuracy: 0.9907 - val_loss: 0.2320 - val_categorical_accuracy: 0.9167
    Epoch 248/500
    4/4 [==============================] - 0s 6ms/step - loss: 0.0401 - categorical_accuracy: 0.9907 - val_loss: 0.2276 - val_categorical_accuracy: 0.9167
    Epoch 249/500
    4/4 [==============================] - 0s 5ms/step - loss: 0.0390 - categorical_accuracy: 1.0000 - val_loss: 0.2262 - val_categorical_accuracy: 0.9167
    Epoch 250/500
    4/4 [==============================] - 0s 5ms/step - loss: 0.0394 - categorical_accuracy: 1.0000 - val_loss: 0.2264 - val_categorical_accuracy: 0.9167
    Epoch 251/500
    4/4 [==============================] - 0s 5ms/step - loss: 0.0389 - categorical_accuracy: 1.0000 - val_loss: 0.2243 - val_categorical_accuracy: 0.9167
    Epoch 252/500
    4/4 [==============================] - 0s 5ms/step - loss: 0.0387 - categorical_accuracy: 1.0000 - val_loss: 0.2253 - val_categorical_accuracy: 0.9167
    Epoch 253/500
    4/4 [==============================] - 0s 5ms/step - loss: 0.0391 - categorical_accuracy: 1.0000 - val_loss: 0.2282 - val_categorical_accuracy: 0.9167
    Epoch 254/500
    4/4 [==============================] - 0s 5ms/step - loss: 0.0380 - categorical_accuracy: 1.0000 - val_loss: 0.2247 - val_categorical_accuracy: 0.9167
    Epoch 255/500
    4/4 [==============================] - 0s 5ms/step - loss: 0.0391 - categorical_accuracy: 1.0000 - val_loss: 0.2245 - val_categorical_accuracy: 0.9167
    Epoch 256/500
    4/4 [==============================] - 0s 5ms/step - loss: 0.0378 - categorical_accuracy: 1.0000 - val_loss: 0.2256 - val_categorical_accuracy: 0.9167
    Epoch 257/500
    4/4 [==============================] - 0s 5ms/step - loss: 0.0377 - categorical_accuracy: 1.0000 - val_loss: 0.2327 - val_categorical_accuracy: 0.9167
    Epoch 258/500
    4/4 [==============================] - 0s 5ms/step - loss: 0.0371 - categorical_accuracy: 1.0000 - val_loss: 0.2320 - val_categorical_accuracy: 0.9167
    Epoch 259/500
    4/4 [==============================] - 0s 5ms/step - loss: 0.0368 - categorical_accuracy: 1.0000 - val_loss: 0.2283 - val_categorical_accuracy: 0.9167
    Epoch 260/500
    4/4 [==============================] - 0s 5ms/step - loss: 0.0371 - categorical_accuracy: 1.0000 - val_loss: 0.2317 - val_categorical_accuracy: 0.9167
    Epoch 261/500
    4/4 [==============================] - 0s 5ms/step - loss: 0.0364 - categorical_accuracy: 1.0000 - val_loss: 0.2253 - val_categorical_accuracy: 0.9167
    Epoch 262/500
    4/4 [==============================] - 0s 5ms/step - loss: 0.0356 - categorical_accuracy: 1.0000 - val_loss: 0.2256 - val_categorical_accuracy: 0.9167
    Epoch 263/500
    4/4 [==============================] - 0s 5ms/step - loss: 0.0349 - categorical_accuracy: 1.0000 - val_loss: 0.2302 - val_categorical_accuracy: 0.9167
    Epoch 264/500
    4/4 [==============================] - 0s 5ms/step - loss: 0.0351 - categorical_accuracy: 1.0000 - val_loss: 0.2375 - val_categorical_accuracy: 0.9167
    Epoch 265/500
    4/4 [==============================] - 0s 5ms/step - loss: 0.0355 - categorical_accuracy: 1.0000 - val_loss: 0.2300 - val_categorical_accuracy: 0.9167
    Epoch 266/500
    4/4 [==============================] - 0s 5ms/step - loss: 0.0342 - categorical_accuracy: 1.0000 - val_loss: 0.2279 - val_categorical_accuracy: 0.9167
    Epoch 267/500
    4/4 [==============================] - 0s 6ms/step - loss: 0.0339 - categorical_accuracy: 1.0000 - val_loss: 0.2274 - val_categorical_accuracy: 0.9167
    Epoch 268/500
    4/4 [==============================] - 0s 5ms/step - loss: 0.0340 - categorical_accuracy: 1.0000 - val_loss: 0.2319 - val_categorical_accuracy: 0.9167
    Epoch 269/500
    4/4 [==============================] - 0s 5ms/step - loss: 0.0340 - categorical_accuracy: 1.0000 - val_loss: 0.2325 - val_categorical_accuracy: 0.9167
    Epoch 270/500
    4/4 [==============================] - 0s 6ms/step - loss: 0.0330 - categorical_accuracy: 1.0000 - val_loss: 0.2263 - val_categorical_accuracy: 0.9167
    Epoch 271/500
    4/4 [==============================] - 0s 5ms/step - loss: 0.0333 - categorical_accuracy: 1.0000 - val_loss: 0.2258 - val_categorical_accuracy: 0.9167
    Epoch 272/500
    4/4 [==============================] - 0s 5ms/step - loss: 0.0327 - categorical_accuracy: 1.0000 - val_loss: 0.2323 - val_categorical_accuracy: 0.9167
    Epoch 273/500
    4/4 [==============================] - 0s 5ms/step - loss: 0.0329 - categorical_accuracy: 1.0000 - val_loss: 0.2396 - val_categorical_accuracy: 0.9167
    Epoch 274/500
    4/4 [==============================] - 0s 5ms/step - loss: 0.0335 - categorical_accuracy: 0.9907 - val_loss: 0.2355 - val_categorical_accuracy: 0.9167
    Epoch 275/500
    4/4 [==============================] - 0s 6ms/step - loss: 0.0319 - categorical_accuracy: 1.0000 - val_loss: 0.2284 - val_categorical_accuracy: 0.9167
    Epoch 276/500
    4/4 [==============================] - 0s 5ms/step - loss: 0.0316 - categorical_accuracy: 1.0000 - val_loss: 0.2259 - val_categorical_accuracy: 0.9167
    Epoch 277/500
    4/4 [==============================] - 0s 5ms/step - loss: 0.0319 - categorical_accuracy: 1.0000 - val_loss: 0.2255 - val_categorical_accuracy: 0.9167
    Epoch 278/500
    4/4 [==============================] - 0s 6ms/step - loss: 0.0319 - categorical_accuracy: 1.0000 - val_loss: 0.2261 - val_categorical_accuracy: 0.9167
    Epoch 279/500
    4/4 [==============================] - 0s 6ms/step - loss: 0.0311 - categorical_accuracy: 1.0000 - val_loss: 0.2307 - val_categorical_accuracy: 0.9167
    Epoch 280/500
    4/4 [==============================] - 0s 6ms/step - loss: 0.0317 - categorical_accuracy: 1.0000 - val_loss: 0.2335 - val_categorical_accuracy: 0.9167
    Epoch 281/500
    4/4 [==============================] - 0s 5ms/step - loss: 0.0306 - categorical_accuracy: 1.0000 - val_loss: 0.2273 - val_categorical_accuracy: 0.9167
    Epoch 282/500
    4/4 [==============================] - 0s 6ms/step - loss: 0.0308 - categorical_accuracy: 1.0000 - val_loss: 0.2266 - val_categorical_accuracy: 0.9167
    Epoch 283/500
    4/4 [==============================] - 0s 5ms/step - loss: 0.0306 - categorical_accuracy: 1.0000 - val_loss: 0.2316 - val_categorical_accuracy: 0.9167
    Epoch 284/500
    4/4 [==============================] - 0s 5ms/step - loss: 0.0299 - categorical_accuracy: 1.0000 - val_loss: 0.2311 - val_categorical_accuracy: 0.9167
    Epoch 285/500
    4/4 [==============================] - 0s 5ms/step - loss: 0.0296 - categorical_accuracy: 1.0000 - val_loss: 0.2316 - val_categorical_accuracy: 0.9167
    Epoch 286/500
    4/4 [==============================] - 0s 5ms/step - loss: 0.0297 - categorical_accuracy: 1.0000 - val_loss: 0.2322 - val_categorical_accuracy: 0.9167
    Epoch 287/500
    4/4 [==============================] - 0s 5ms/step - loss: 0.0291 - categorical_accuracy: 1.0000 - val_loss: 0.2296 - val_categorical_accuracy: 0.9167
    Epoch 288/500
    4/4 [==============================] - 0s 5ms/step - loss: 0.0287 - categorical_accuracy: 1.0000 - val_loss: 0.2265 - val_categorical_accuracy: 0.9167
    Epoch 289/500
    4/4 [==============================] - 0s 6ms/step - loss: 0.0302 - categorical_accuracy: 1.0000 - val_loss: 0.2261 - val_categorical_accuracy: 0.9167
    Epoch 290/500
    4/4 [==============================] - 0s 5ms/step - loss: 0.0293 - categorical_accuracy: 1.0000 - val_loss: 0.2297 - val_categorical_accuracy: 0.9167
    Epoch 291/500
    4/4 [==============================] - 0s 5ms/step - loss: 0.0278 - categorical_accuracy: 1.0000 - val_loss: 0.2387 - val_categorical_accuracy: 0.9167
    Epoch 292/500
    4/4 [==============================] - 0s 5ms/step - loss: 0.0295 - categorical_accuracy: 1.0000 - val_loss: 0.2415 - val_categorical_accuracy: 0.9167
    Epoch 293/500
    4/4 [==============================] - 0s 6ms/step - loss: 0.0289 - categorical_accuracy: 1.0000 - val_loss: 0.2331 - val_categorical_accuracy: 0.9167
    Epoch 294/500
    4/4 [==============================] - 0s 5ms/step - loss: 0.0280 - categorical_accuracy: 1.0000 - val_loss: 0.2307 - val_categorical_accuracy: 0.9167
    Epoch 295/500
    4/4 [==============================] - 0s 5ms/step - loss: 0.0275 - categorical_accuracy: 1.0000 - val_loss: 0.2351 - val_categorical_accuracy: 0.9167
    Epoch 296/500
    4/4 [==============================] - 0s 5ms/step - loss: 0.0282 - categorical_accuracy: 1.0000 - val_loss: 0.2313 - val_categorical_accuracy: 0.9167
    Epoch 297/500
    4/4 [==============================] - 0s 6ms/step - loss: 0.0278 - categorical_accuracy: 1.0000 - val_loss: 0.2383 - val_categorical_accuracy: 0.9167
    Epoch 298/500
    4/4 [==============================] - 0s 5ms/step - loss: 0.0270 - categorical_accuracy: 1.0000 - val_loss: 0.2325 - val_categorical_accuracy: 0.9167
    Epoch 299/500
    4/4 [==============================] - 0s 5ms/step - loss: 0.0270 - categorical_accuracy: 1.0000 - val_loss: 0.2295 - val_categorical_accuracy: 0.9167
    Epoch 300/500
    4/4 [==============================] - 0s 6ms/step - loss: 0.0267 - categorical_accuracy: 1.0000 - val_loss: 0.2324 - val_categorical_accuracy: 0.9167
    Epoch 301/500
    4/4 [==============================] - 0s 5ms/step - loss: 0.0264 - categorical_accuracy: 1.0000 - val_loss: 0.2313 - val_categorical_accuracy: 0.9167
    Epoch 302/500
    4/4 [==============================] - 0s 5ms/step - loss: 0.0264 - categorical_accuracy: 1.0000 - val_loss: 0.2322 - val_categorical_accuracy: 0.9167
    Epoch 303/500
    4/4 [==============================] - 0s 5ms/step - loss: 0.0258 - categorical_accuracy: 1.0000 - val_loss: 0.2357 - val_categorical_accuracy: 0.9167
    Epoch 304/500
    4/4 [==============================] - 0s 5ms/step - loss: 0.0267 - categorical_accuracy: 1.0000 - val_loss: 0.2404 - val_categorical_accuracy: 0.9167
    Epoch 305/500
    4/4 [==============================] - 0s 5ms/step - loss: 0.0260 - categorical_accuracy: 1.0000 - val_loss: 0.2333 - val_categorical_accuracy: 0.9167
    Epoch 306/500
    4/4 [==============================] - 0s 5ms/step - loss: 0.0256 - categorical_accuracy: 1.0000 - val_loss: 0.2295 - val_categorical_accuracy: 0.9167
    Epoch 307/500
    4/4 [==============================] - 0s 7ms/step - loss: 0.0258 - categorical_accuracy: 1.0000 - val_loss: 0.2314 - val_categorical_accuracy: 0.9167
    Epoch 308/500
    4/4 [==============================] - 0s 5ms/step - loss: 0.0247 - categorical_accuracy: 1.0000 - val_loss: 0.2408 - val_categorical_accuracy: 0.9167
    Epoch 309/500
    4/4 [==============================] - 0s 5ms/step - loss: 0.0255 - categorical_accuracy: 1.0000 - val_loss: 0.2431 - val_categorical_accuracy: 0.9167
    Epoch 310/500
    4/4 [==============================] - 0s 5ms/step - loss: 0.0249 - categorical_accuracy: 1.0000 - val_loss: 0.2351 - val_categorical_accuracy: 0.9167
    Epoch 311/500
    4/4 [==============================] - 0s 5ms/step - loss: 0.0245 - categorical_accuracy: 1.0000 - val_loss: 0.2319 - val_categorical_accuracy: 0.9167
    Epoch 312/500
    4/4 [==============================] - 0s 6ms/step - loss: 0.0256 - categorical_accuracy: 1.0000 - val_loss: 0.2315 - val_categorical_accuracy: 0.9167
    Epoch 313/500
    4/4 [==============================] - 0s 5ms/step - loss: 0.0244 - categorical_accuracy: 1.0000 - val_loss: 0.2384 - val_categorical_accuracy: 0.9167
    Epoch 314/500
    4/4 [==============================] - 0s 5ms/step - loss: 0.0243 - categorical_accuracy: 1.0000 - val_loss: 0.2430 - val_categorical_accuracy: 0.9167
    Epoch 315/500
    4/4 [==============================] - 0s 6ms/step - loss: 0.0244 - categorical_accuracy: 1.0000 - val_loss: 0.2406 - val_categorical_accuracy: 0.9167
    Epoch 316/500
    4/4 [==============================] - 0s 5ms/step - loss: 0.0238 - categorical_accuracy: 1.0000 - val_loss: 0.2354 - val_categorical_accuracy: 0.9167
    Epoch 317/500
    4/4 [==============================] - 0s 5ms/step - loss: 0.0238 - categorical_accuracy: 1.0000 - val_loss: 0.2346 - val_categorical_accuracy: 0.9167
    Epoch 318/500
    4/4 [==============================] - 0s 6ms/step - loss: 0.0241 - categorical_accuracy: 1.0000 - val_loss: 0.2408 - val_categorical_accuracy: 0.9167
    Epoch 319/500
    4/4 [==============================] - 0s 6ms/step - loss: 0.0245 - categorical_accuracy: 1.0000 - val_loss: 0.2340 - val_categorical_accuracy: 0.9167
    Epoch 320/500
    4/4 [==============================] - 0s 5ms/step - loss: 0.0236 - categorical_accuracy: 1.0000 - val_loss: 0.2375 - val_categorical_accuracy: 0.9167
    Epoch 321/500
    4/4 [==============================] - 0s 5ms/step - loss: 0.0230 - categorical_accuracy: 1.0000 - val_loss: 0.2382 - val_categorical_accuracy: 0.9167
    Epoch 322/500
    4/4 [==============================] - 0s 5ms/step - loss: 0.0229 - categorical_accuracy: 1.0000 - val_loss: 0.2364 - val_categorical_accuracy: 0.9167
    Epoch 323/500
    4/4 [==============================] - 0s 5ms/step - loss: 0.0227 - categorical_accuracy: 1.0000 - val_loss: 0.2350 - val_categorical_accuracy: 0.9167
    Epoch 324/500
    4/4 [==============================] - 0s 6ms/step - loss: 0.0229 - categorical_accuracy: 1.0000 - val_loss: 0.2377 - val_categorical_accuracy: 0.9167
    Epoch 325/500
    4/4 [==============================] - 0s 7ms/step - loss: 0.0224 - categorical_accuracy: 1.0000 - val_loss: 0.2355 - val_categorical_accuracy: 0.9167
    Epoch 326/500
    4/4 [==============================] - 0s 6ms/step - loss: 0.0224 - categorical_accuracy: 1.0000 - val_loss: 0.2336 - val_categorical_accuracy: 0.9167
    Epoch 327/500
    4/4 [==============================] - 0s 6ms/step - loss: 0.0228 - categorical_accuracy: 1.0000 - val_loss: 0.2338 - val_categorical_accuracy: 0.9167
    Epoch 328/500
    4/4 [==============================] - 0s 5ms/step - loss: 0.0226 - categorical_accuracy: 1.0000 - val_loss: 0.2386 - val_categorical_accuracy: 0.9167
    Epoch 329/500
    4/4 [==============================] - 0s 6ms/step - loss: 0.0222 - categorical_accuracy: 1.0000 - val_loss: 0.2400 - val_categorical_accuracy: 0.9167
    Epoch 330/500
    4/4 [==============================] - 0s 5ms/step - loss: 0.0215 - categorical_accuracy: 1.0000 - val_loss: 0.2355 - val_categorical_accuracy: 0.9167
    Epoch 331/500
    4/4 [==============================] - 0s 5ms/step - loss: 0.0221 - categorical_accuracy: 1.0000 - val_loss: 0.2365 - val_categorical_accuracy: 0.9167
    Epoch 332/500
    4/4 [==============================] - 0s 5ms/step - loss: 0.0240 - categorical_accuracy: 1.0000 - val_loss: 0.2352 - val_categorical_accuracy: 0.9167
    Epoch 333/500
    4/4 [==============================] - 0s 5ms/step - loss: 0.0214 - categorical_accuracy: 1.0000 - val_loss: 0.2607 - val_categorical_accuracy: 0.9167
    Epoch 334/500
    4/4 [==============================] - 0s 5ms/step - loss: 0.0239 - categorical_accuracy: 1.0000 - val_loss: 0.2615 - val_categorical_accuracy: 0.9167
    Epoch 335/500
    4/4 [==============================] - 0s 5ms/step - loss: 0.0236 - categorical_accuracy: 1.0000 - val_loss: 0.2411 - val_categorical_accuracy: 0.9167
    Epoch 336/500
    4/4 [==============================] - 0s 6ms/step - loss: 0.0207 - categorical_accuracy: 1.0000 - val_loss: 0.2364 - val_categorical_accuracy: 0.9167
    Epoch 337/500
    4/4 [==============================] - 0s 5ms/step - loss: 0.0210 - categorical_accuracy: 1.0000 - val_loss: 0.2374 - val_categorical_accuracy: 0.9167
    Epoch 338/500
    4/4 [==============================] - 0s 5ms/step - loss: 0.0212 - categorical_accuracy: 1.0000 - val_loss: 0.2420 - val_categorical_accuracy: 0.9167
    Epoch 339/500
    4/4 [==============================] - 0s 5ms/step - loss: 0.0209 - categorical_accuracy: 1.0000 - val_loss: 0.2382 - val_categorical_accuracy: 0.9167
    Epoch 340/500
    4/4 [==============================] - 0s 5ms/step - loss: 0.0206 - categorical_accuracy: 1.0000 - val_loss: 0.2408 - val_categorical_accuracy: 0.9167
    Epoch 341/500
    4/4 [==============================] - 0s 5ms/step - loss: 0.0202 - categorical_accuracy: 1.0000 - val_loss: 0.2448 - val_categorical_accuracy: 0.9167
    Epoch 342/500
    4/4 [==============================] - 0s 5ms/step - loss: 0.0208 - categorical_accuracy: 1.0000 - val_loss: 0.2475 - val_categorical_accuracy: 0.9167
    Epoch 343/500
    4/4 [==============================] - 0s 7ms/step - loss: 0.0211 - categorical_accuracy: 1.0000 - val_loss: 0.2359 - val_categorical_accuracy: 0.9167
    Epoch 344/500
    4/4 [==============================] - 0s 5ms/step - loss: 0.0210 - categorical_accuracy: 1.0000 - val_loss: 0.2363 - val_categorical_accuracy: 0.9167
    Epoch 345/500
    4/4 [==============================] - 0s 5ms/step - loss: 0.0202 - categorical_accuracy: 1.0000 - val_loss: 0.2438 - val_categorical_accuracy: 0.9167
    Epoch 346/500
    4/4 [==============================] - 0s 6ms/step - loss: 0.0194 - categorical_accuracy: 1.0000 - val_loss: 0.2490 - val_categorical_accuracy: 0.9167
    Epoch 347/500
    4/4 [==============================] - 0s 5ms/step - loss: 0.0200 - categorical_accuracy: 1.0000 - val_loss: 0.2499 - val_categorical_accuracy: 0.9167
    Epoch 348/500
    4/4 [==============================] - 0s 6ms/step - loss: 0.0197 - categorical_accuracy: 1.0000 - val_loss: 0.2401 - val_categorical_accuracy: 0.9167
    Epoch 349/500
    4/4 [==============================] - 0s 5ms/step - loss: 0.0195 - categorical_accuracy: 1.0000 - val_loss: 0.2417 - val_categorical_accuracy: 0.9167
    Epoch 350/500
    4/4 [==============================] - 0s 5ms/step - loss: 0.0204 - categorical_accuracy: 1.0000 - val_loss: 0.2578 - val_categorical_accuracy: 0.9167
    Epoch 351/500
    4/4 [==============================] - 0s 6ms/step - loss: 0.0203 - categorical_accuracy: 1.0000 - val_loss: 0.2543 - val_categorical_accuracy: 0.9167
    Epoch 352/500
    4/4 [==============================] - 0s 6ms/step - loss: 0.0193 - categorical_accuracy: 1.0000 - val_loss: 0.2445 - val_categorical_accuracy: 0.9167
    Epoch 353/500
    4/4 [==============================] - 0s 6ms/step - loss: 0.0188 - categorical_accuracy: 1.0000 - val_loss: 0.2421 - val_categorical_accuracy: 0.9167
    Epoch 354/500
    4/4 [==============================] - 0s 5ms/step - loss: 0.0189 - categorical_accuracy: 1.0000 - val_loss: 0.2457 - val_categorical_accuracy: 0.9167
    Epoch 355/500
    4/4 [==============================] - 0s 5ms/step - loss: 0.0185 - categorical_accuracy: 1.0000 - val_loss: 0.2435 - val_categorical_accuracy: 0.9167
    Epoch 356/500
    4/4 [==============================] - 0s 7ms/step - loss: 0.0184 - categorical_accuracy: 1.0000 - val_loss: 0.2417 - val_categorical_accuracy: 0.9167
    Epoch 357/500
    4/4 [==============================] - 0s 6ms/step - loss: 0.0190 - categorical_accuracy: 1.0000 - val_loss: 0.2458 - val_categorical_accuracy: 0.9167
    Epoch 358/500
    4/4 [==============================] - 0s 6ms/step - loss: 0.0182 - categorical_accuracy: 1.0000 - val_loss: 0.2474 - val_categorical_accuracy: 0.9167
    Epoch 359/500
    4/4 [==============================] - 0s 6ms/step - loss: 0.0179 - categorical_accuracy: 1.0000 - val_loss: 0.2540 - val_categorical_accuracy: 0.9167
    Epoch 360/500
    4/4 [==============================] - 0s 5ms/step - loss: 0.0185 - categorical_accuracy: 1.0000 - val_loss: 0.2563 - val_categorical_accuracy: 0.9167
    Epoch 361/500
    4/4 [==============================] - 0s 5ms/step - loss: 0.0182 - categorical_accuracy: 1.0000 - val_loss: 0.2450 - val_categorical_accuracy: 0.9167
    Epoch 362/500
    4/4 [==============================] - 0s 6ms/step - loss: 0.0182 - categorical_accuracy: 1.0000 - val_loss: 0.2455 - val_categorical_accuracy: 0.9167
    Epoch 363/500
    4/4 [==============================] - 0s 5ms/step - loss: 0.0189 - categorical_accuracy: 1.0000 - val_loss: 0.2417 - val_categorical_accuracy: 0.9167
    Epoch 364/500
    4/4 [==============================] - 0s 5ms/step - loss: 0.0190 - categorical_accuracy: 1.0000 - val_loss: 0.2452 - val_categorical_accuracy: 0.9167
    Epoch 365/500
    4/4 [==============================] - 0s 5ms/step - loss: 0.0184 - categorical_accuracy: 1.0000 - val_loss: 0.2576 - val_categorical_accuracy: 0.9167
    Epoch 366/500
    4/4 [==============================] - 0s 5ms/step - loss: 0.0179 - categorical_accuracy: 1.0000 - val_loss: 0.2589 - val_categorical_accuracy: 0.9167
    Epoch 367/500
    4/4 [==============================] - 0s 5ms/step - loss: 0.0173 - categorical_accuracy: 1.0000 - val_loss: 0.2512 - val_categorical_accuracy: 0.9167
    Epoch 368/500
    4/4 [==============================] - 0s 6ms/step - loss: 0.0176 - categorical_accuracy: 1.0000 - val_loss: 0.2433 - val_categorical_accuracy: 0.9167
    Epoch 369/500
    4/4 [==============================] - 0s 5ms/step - loss: 0.0183 - categorical_accuracy: 1.0000 - val_loss: 0.2446 - val_categorical_accuracy: 0.9167
    Epoch 370/500
    4/4 [==============================] - 0s 6ms/step - loss: 0.0169 - categorical_accuracy: 1.0000 - val_loss: 0.2573 - val_categorical_accuracy: 0.9167
    Epoch 371/500
    4/4 [==============================] - 0s 5ms/step - loss: 0.0192 - categorical_accuracy: 1.0000 - val_loss: 0.2603 - val_categorical_accuracy: 0.9167
    Epoch 372/500
    4/4 [==============================] - 0s 5ms/step - loss: 0.0182 - categorical_accuracy: 1.0000 - val_loss: 0.2435 - val_categorical_accuracy: 0.8333
    Epoch 373/500
    4/4 [==============================] - 0s 5ms/step - loss: 0.0175 - categorical_accuracy: 1.0000 - val_loss: 0.2475 - val_categorical_accuracy: 0.9167
    Epoch 374/500
    4/4 [==============================] - 0s 6ms/step - loss: 0.0160 - categorical_accuracy: 1.0000 - val_loss: 0.2695 - val_categorical_accuracy: 0.9167
    Epoch 375/500
    4/4 [==============================] - 0s 6ms/step - loss: 0.0196 - categorical_accuracy: 1.0000 - val_loss: 0.2759 - val_categorical_accuracy: 0.9167
    Epoch 376/500
    4/4 [==============================] - 0s 5ms/step - loss: 0.0173 - categorical_accuracy: 1.0000 - val_loss: 0.2550 - val_categorical_accuracy: 0.9167
    Epoch 377/500
    4/4 [==============================] - 0s 5ms/step - loss: 0.0151 - categorical_accuracy: 1.0000 - val_loss: 0.2437 - val_categorical_accuracy: 0.9167
    Epoch 378/500
    4/4 [==============================] - 0s 7ms/step - loss: 0.0174 - categorical_accuracy: 1.0000 - val_loss: 0.2443 - val_categorical_accuracy: 0.8333
    Epoch 379/500
    4/4 [==============================] - 0s 6ms/step - loss: 0.0183 - categorical_accuracy: 1.0000 - val_loss: 0.2452 - val_categorical_accuracy: 0.9167
    Epoch 380/500
    4/4 [==============================] - 0s 5ms/step - loss: 0.0159 - categorical_accuracy: 1.0000 - val_loss: 0.2540 - val_categorical_accuracy: 0.9167
    Epoch 381/500
    4/4 [==============================] - 0s 6ms/step - loss: 0.0165 - categorical_accuracy: 1.0000 - val_loss: 0.2746 - val_categorical_accuracy: 0.9167
    Epoch 382/500
    4/4 [==============================] - 0s 5ms/step - loss: 0.0174 - categorical_accuracy: 1.0000 - val_loss: 0.2719 - val_categorical_accuracy: 0.9167
    Epoch 383/500
    4/4 [==============================] - 0s 6ms/step - loss: 0.0166 - categorical_accuracy: 1.0000 - val_loss: 0.2556 - val_categorical_accuracy: 0.9167
    Epoch 384/500
    4/4 [==============================] - 0s 5ms/step - loss: 0.0154 - categorical_accuracy: 1.0000 - val_loss: 0.2502 - val_categorical_accuracy: 0.9167
    Epoch 385/500
    4/4 [==============================] - 0s 6ms/step - loss: 0.0157 - categorical_accuracy: 1.0000 - val_loss: 0.2504 - val_categorical_accuracy: 0.9167
    Epoch 386/500
    4/4 [==============================] - 0s 5ms/step - loss: 0.0153 - categorical_accuracy: 1.0000 - val_loss: 0.2578 - val_categorical_accuracy: 0.9167
    Epoch 387/500
    4/4 [==============================] - 0s 5ms/step - loss: 0.0155 - categorical_accuracy: 1.0000 - val_loss: 0.2628 - val_categorical_accuracy: 0.9167
    Epoch 388/500
    4/4 [==============================] - 0s 5ms/step - loss: 0.0154 - categorical_accuracy: 1.0000 - val_loss: 0.2535 - val_categorical_accuracy: 0.9167
    Epoch 389/500
    4/4 [==============================] - 0s 6ms/step - loss: 0.0152 - categorical_accuracy: 1.0000 - val_loss: 0.2520 - val_categorical_accuracy: 0.9167
    Epoch 390/500
    4/4 [==============================] - 0s 5ms/step - loss: 0.0153 - categorical_accuracy: 1.0000 - val_loss: 0.2551 - val_categorical_accuracy: 0.9167
    Epoch 391/500
    4/4 [==============================] - 0s 5ms/step - loss: 0.0151 - categorical_accuracy: 1.0000 - val_loss: 0.2531 - val_categorical_accuracy: 0.9167
    Epoch 392/500
    4/4 [==============================] - 0s 5ms/step - loss: 0.0150 - categorical_accuracy: 1.0000 - val_loss: 0.2547 - val_categorical_accuracy: 0.9167
    Epoch 393/500
    4/4 [==============================] - 0s 5ms/step - loss: 0.0147 - categorical_accuracy: 1.0000 - val_loss: 0.2605 - val_categorical_accuracy: 0.9167
    Epoch 394/500
    4/4 [==============================] - 0s 6ms/step - loss: 0.0150 - categorical_accuracy: 1.0000 - val_loss: 0.2655 - val_categorical_accuracy: 0.9167
    Epoch 395/500
    4/4 [==============================] - 0s 5ms/step - loss: 0.0149 - categorical_accuracy: 1.0000 - val_loss: 0.2611 - val_categorical_accuracy: 0.9167
    Epoch 396/500
    4/4 [==============================] - 0s 5ms/step - loss: 0.0146 - categorical_accuracy: 1.0000 - val_loss: 0.2580 - val_categorical_accuracy: 0.9167
    Epoch 397/500
    4/4 [==============================] - 0s 5ms/step - loss: 0.0147 - categorical_accuracy: 1.0000 - val_loss: 0.2555 - val_categorical_accuracy: 0.9167
    Epoch 398/500
    4/4 [==============================] - 0s 5ms/step - loss: 0.0152 - categorical_accuracy: 1.0000 - val_loss: 0.2525 - val_categorical_accuracy: 0.9167
    Epoch 399/500
    4/4 [==============================] - 0s 5ms/step - loss: 0.0145 - categorical_accuracy: 1.0000 - val_loss: 0.2575 - val_categorical_accuracy: 0.9167
    Epoch 400/500
    4/4 [==============================] - 0s 5ms/step - loss: 0.0142 - categorical_accuracy: 1.0000 - val_loss: 0.2642 - val_categorical_accuracy: 0.9167
    Epoch 401/500
    4/4 [==============================] - 0s 5ms/step - loss: 0.0150 - categorical_accuracy: 1.0000 - val_loss: 0.2710 - val_categorical_accuracy: 0.9167
    Epoch 402/500
    4/4 [==============================] - 0s 5ms/step - loss: 0.0147 - categorical_accuracy: 1.0000 - val_loss: 0.2632 - val_categorical_accuracy: 0.9167
    Epoch 403/500
    4/4 [==============================] - 0s 5ms/step - loss: 0.0152 - categorical_accuracy: 1.0000 - val_loss: 0.2512 - val_categorical_accuracy: 0.9167
    Epoch 404/500
    4/4 [==============================] - 0s 5ms/step - loss: 0.0151 - categorical_accuracy: 1.0000 - val_loss: 0.2620 - val_categorical_accuracy: 0.9167
    Epoch 405/500
    4/4 [==============================] - 0s 5ms/step - loss: 0.0140 - categorical_accuracy: 1.0000 - val_loss: 0.2720 - val_categorical_accuracy: 0.9167
    Epoch 406/500
    4/4 [==============================] - 0s 6ms/step - loss: 0.0142 - categorical_accuracy: 1.0000 - val_loss: 0.2712 - val_categorical_accuracy: 0.9167
    Epoch 407/500
    4/4 [==============================] - 0s 5ms/step - loss: 0.0139 - categorical_accuracy: 1.0000 - val_loss: 0.2637 - val_categorical_accuracy: 0.9167
    Epoch 408/500
    4/4 [==============================] - 0s 5ms/step - loss: 0.0138 - categorical_accuracy: 1.0000 - val_loss: 0.2534 - val_categorical_accuracy: 0.9167
    Epoch 409/500
    4/4 [==============================] - 0s 5ms/step - loss: 0.0142 - categorical_accuracy: 1.0000 - val_loss: 0.2545 - val_categorical_accuracy: 0.9167
    Epoch 410/500
    4/4 [==============================] - 0s 6ms/step - loss: 0.0139 - categorical_accuracy: 1.0000 - val_loss: 0.2618 - val_categorical_accuracy: 0.9167
    Epoch 411/500
    4/4 [==============================] - 0s 5ms/step - loss: 0.0133 - categorical_accuracy: 1.0000 - val_loss: 0.2716 - val_categorical_accuracy: 0.9167
    Epoch 412/500
    4/4 [==============================] - 0s 5ms/step - loss: 0.0142 - categorical_accuracy: 1.0000 - val_loss: 0.2736 - val_categorical_accuracy: 0.9167
    Epoch 413/500
    4/4 [==============================] - 0s 5ms/step - loss: 0.0135 - categorical_accuracy: 1.0000 - val_loss: 0.2590 - val_categorical_accuracy: 0.9167
    Epoch 414/500
    4/4 [==============================] - 0s 5ms/step - loss: 0.0137 - categorical_accuracy: 1.0000 - val_loss: 0.2578 - val_categorical_accuracy: 0.9167
    Epoch 415/500
    4/4 [==============================] - 0s 7ms/step - loss: 0.0129 - categorical_accuracy: 1.0000 - val_loss: 0.2713 - val_categorical_accuracy: 0.9167
    Epoch 416/500
    4/4 [==============================] - 0s 5ms/step - loss: 0.0147 - categorical_accuracy: 1.0000 - val_loss: 0.2926 - val_categorical_accuracy: 0.9167
    Epoch 417/500
    4/4 [==============================] - 0s 5ms/step - loss: 0.0151 - categorical_accuracy: 1.0000 - val_loss: 0.2761 - val_categorical_accuracy: 0.9167
    Epoch 418/500
    4/4 [==============================] - 0s 5ms/step - loss: 0.0136 - categorical_accuracy: 1.0000 - val_loss: 0.2537 - val_categorical_accuracy: 0.8333
    Epoch 419/500
    4/4 [==============================] - 0s 5ms/step - loss: 0.0163 - categorical_accuracy: 1.0000 - val_loss: 0.2539 - val_categorical_accuracy: 0.8333
    Epoch 420/500
    4/4 [==============================] - 0s 5ms/step - loss: 0.0131 - categorical_accuracy: 1.0000 - val_loss: 0.2972 - val_categorical_accuracy: 0.9167
    Epoch 421/500
    4/4 [==============================] - 0s 5ms/step - loss: 0.0183 - categorical_accuracy: 1.0000 - val_loss: 0.3033 - val_categorical_accuracy: 0.9167
    Epoch 422/500
    4/4 [==============================] - 0s 6ms/step - loss: 0.0137 - categorical_accuracy: 1.0000 - val_loss: 0.2595 - val_categorical_accuracy: 0.9167
    Epoch 423/500
    4/4 [==============================] - 0s 6ms/step - loss: 0.0143 - categorical_accuracy: 1.0000 - val_loss: 0.2562 - val_categorical_accuracy: 0.8333
    Epoch 424/500
    4/4 [==============================] - 0s 8ms/step - loss: 0.0162 - categorical_accuracy: 1.0000 - val_loss: 0.2534 - val_categorical_accuracy: 0.8333
    Epoch 425/500
    4/4 [==============================] - 0s 6ms/step - loss: 0.0139 - categorical_accuracy: 1.0000 - val_loss: 0.2662 - val_categorical_accuracy: 0.9167
    Epoch 426/500
    4/4 [==============================] - 0s 5ms/step - loss: 0.0127 - categorical_accuracy: 1.0000 - val_loss: 0.2772 - val_categorical_accuracy: 0.9167
    Epoch 427/500
    4/4 [==============================] - 0s 5ms/step - loss: 0.0127 - categorical_accuracy: 1.0000 - val_loss: 0.2699 - val_categorical_accuracy: 0.9167
    Epoch 428/500
    4/4 [==============================] - 0s 5ms/step - loss: 0.0129 - categorical_accuracy: 1.0000 - val_loss: 0.2623 - val_categorical_accuracy: 0.9167
    Epoch 429/500
    4/4 [==============================] - 0s 5ms/step - loss: 0.0121 - categorical_accuracy: 1.0000 - val_loss: 0.2729 - val_categorical_accuracy: 0.9167
    Epoch 430/500
    4/4 [==============================] - 0s 5ms/step - loss: 0.0123 - categorical_accuracy: 1.0000 - val_loss: 0.2748 - val_categorical_accuracy: 0.9167
    Epoch 431/500
    4/4 [==============================] - 0s 6ms/step - loss: 0.0122 - categorical_accuracy: 1.0000 - val_loss: 0.2647 - val_categorical_accuracy: 0.9167
    Epoch 432/500
    4/4 [==============================] - 0s 5ms/step - loss: 0.0125 - categorical_accuracy: 1.0000 - val_loss: 0.2573 - val_categorical_accuracy: 0.9167
    Epoch 433/500
    4/4 [==============================] - 0s 5ms/step - loss: 0.0125 - categorical_accuracy: 1.0000 - val_loss: 0.2619 - val_categorical_accuracy: 0.9167
    Epoch 434/500
    4/4 [==============================] - 0s 5ms/step - loss: 0.0116 - categorical_accuracy: 1.0000 - val_loss: 0.2852 - val_categorical_accuracy: 0.9167
    Epoch 435/500
    4/4 [==============================] - 0s 5ms/step - loss: 0.0127 - categorical_accuracy: 1.0000 - val_loss: 0.2880 - val_categorical_accuracy: 0.9167
    Epoch 436/500
    4/4 [==============================] - 0s 5ms/step - loss: 0.0128 - categorical_accuracy: 1.0000 - val_loss: 0.2755 - val_categorical_accuracy: 0.9167
    Epoch 437/500
    4/4 [==============================] - 0s 5ms/step - loss: 0.0120 - categorical_accuracy: 1.0000 - val_loss: 0.2621 - val_categorical_accuracy: 0.9167
    Epoch 438/500
    4/4 [==============================] - 0s 5ms/step - loss: 0.0118 - categorical_accuracy: 1.0000 - val_loss: 0.2602 - val_categorical_accuracy: 0.9167
    Epoch 439/500
    4/4 [==============================] - 0s 5ms/step - loss: 0.0119 - categorical_accuracy: 1.0000 - val_loss: 0.2632 - val_categorical_accuracy: 0.9167
    Epoch 440/500
    4/4 [==============================] - 0s 5ms/step - loss: 0.0117 - categorical_accuracy: 1.0000 - val_loss: 0.2660 - val_categorical_accuracy: 0.9167
    Epoch 441/500
    4/4 [==============================] - 0s 5ms/step - loss: 0.0122 - categorical_accuracy: 1.0000 - val_loss: 0.2850 - val_categorical_accuracy: 0.9167
    Epoch 442/500
    4/4 [==============================] - 0s 6ms/step - loss: 0.0121 - categorical_accuracy: 1.0000 - val_loss: 0.2818 - val_categorical_accuracy: 0.9167
    Epoch 443/500
    4/4 [==============================] - 0s 5ms/step - loss: 0.0116 - categorical_accuracy: 1.0000 - val_loss: 0.2701 - val_categorical_accuracy: 0.9167
    Epoch 444/500
    4/4 [==============================] - 0s 5ms/step - loss: 0.0112 - categorical_accuracy: 1.0000 - val_loss: 0.2650 - val_categorical_accuracy: 0.9167
    Epoch 445/500
    4/4 [==============================] - 0s 5ms/step - loss: 0.0120 - categorical_accuracy: 1.0000 - val_loss: 0.2658 - val_categorical_accuracy: 0.9167
    Epoch 446/500
    4/4 [==============================] - 0s 6ms/step - loss: 0.0110 - categorical_accuracy: 1.0000 - val_loss: 0.2952 - val_categorical_accuracy: 0.9167
    Epoch 447/500
    4/4 [==============================] - 0s 6ms/step - loss: 0.0127 - categorical_accuracy: 1.0000 - val_loss: 0.2959 - val_categorical_accuracy: 0.9167
    Epoch 448/500
    4/4 [==============================] - 0s 5ms/step - loss: 0.0128 - categorical_accuracy: 1.0000 - val_loss: 0.2676 - val_categorical_accuracy: 0.9167
    Epoch 449/500
    4/4 [==============================] - 0s 5ms/step - loss: 0.0113 - categorical_accuracy: 1.0000 - val_loss: 0.2607 - val_categorical_accuracy: 0.9167
    Epoch 450/500
    4/4 [==============================] - 0s 5ms/step - loss: 0.0119 - categorical_accuracy: 1.0000 - val_loss: 0.2662 - val_categorical_accuracy: 0.9167
    Epoch 451/500
    4/4 [==============================] - 0s 7ms/step - loss: 0.0111 - categorical_accuracy: 1.0000 - val_loss: 0.2712 - val_categorical_accuracy: 0.9167
    Epoch 452/500
    4/4 [==============================] - 0s 5ms/step - loss: 0.0109 - categorical_accuracy: 1.0000 - val_loss: 0.2681 - val_categorical_accuracy: 0.9167
    Epoch 453/500
    4/4 [==============================] - 0s 5ms/step - loss: 0.0108 - categorical_accuracy: 1.0000 - val_loss: 0.2704 - val_categorical_accuracy: 0.9167
    Epoch 454/500
    4/4 [==============================] - 0s 5ms/step - loss: 0.0109 - categorical_accuracy: 1.0000 - val_loss: 0.2768 - val_categorical_accuracy: 0.9167
    Epoch 455/500
    4/4 [==============================] - 0s 5ms/step - loss: 0.0107 - categorical_accuracy: 1.0000 - val_loss: 0.2758 - val_categorical_accuracy: 0.9167
    Epoch 456/500
    4/4 [==============================] - 0s 5ms/step - loss: 0.0105 - categorical_accuracy: 1.0000 - val_loss: 0.2694 - val_categorical_accuracy: 0.9167
    Epoch 457/500
    4/4 [==============================] - 0s 5ms/step - loss: 0.0109 - categorical_accuracy: 1.0000 - val_loss: 0.2638 - val_categorical_accuracy: 0.9167
    Epoch 458/500
    4/4 [==============================] - 0s 5ms/step - loss: 0.0112 - categorical_accuracy: 1.0000 - val_loss: 0.2666 - val_categorical_accuracy: 0.9167
    Epoch 459/500
    4/4 [==============================] - 0s 5ms/step - loss: 0.0104 - categorical_accuracy: 1.0000 - val_loss: 0.2839 - val_categorical_accuracy: 0.9167
    Epoch 460/500
    4/4 [==============================] - 0s 5ms/step - loss: 0.0110 - categorical_accuracy: 1.0000 - val_loss: 0.2914 - val_categorical_accuracy: 0.9167
    Epoch 461/500
    4/4 [==============================] - 0s 5ms/step - loss: 0.0113 - categorical_accuracy: 1.0000 - val_loss: 0.2787 - val_categorical_accuracy: 0.9167
    Epoch 462/500
    4/4 [==============================] - 0s 6ms/step - loss: 0.0105 - categorical_accuracy: 1.0000 - val_loss: 0.2725 - val_categorical_accuracy: 0.9167
    Epoch 463/500
    4/4 [==============================] - 0s 5ms/step - loss: 0.0104 - categorical_accuracy: 1.0000 - val_loss: 0.2731 - val_categorical_accuracy: 0.9167
    Epoch 464/500
    4/4 [==============================] - 0s 5ms/step - loss: 0.0103 - categorical_accuracy: 1.0000 - val_loss: 0.2701 - val_categorical_accuracy: 0.9167
    Epoch 465/500
    4/4 [==============================] - 0s 6ms/step - loss: 0.0106 - categorical_accuracy: 1.0000 - val_loss: 0.2668 - val_categorical_accuracy: 0.9167
    Epoch 466/500
    4/4 [==============================] - 0s 5ms/step - loss: 0.0105 - categorical_accuracy: 1.0000 - val_loss: 0.2745 - val_categorical_accuracy: 0.9167
    Epoch 467/500
    4/4 [==============================] - 0s 6ms/step - loss: 0.0108 - categorical_accuracy: 1.0000 - val_loss: 0.2929 - val_categorical_accuracy: 0.9167
    Epoch 468/500
    4/4 [==============================] - 0s 7ms/step - loss: 0.0105 - categorical_accuracy: 1.0000 - val_loss: 0.2837 - val_categorical_accuracy: 0.9167
    Epoch 469/500
    4/4 [==============================] - 0s 5ms/step - loss: 0.0103 - categorical_accuracy: 1.0000 - val_loss: 0.2794 - val_categorical_accuracy: 0.9167
    Epoch 470/500
    4/4 [==============================] - 0s 5ms/step - loss: 0.0101 - categorical_accuracy: 1.0000 - val_loss: 0.2871 - val_categorical_accuracy: 0.9167
    Epoch 471/500
    4/4 [==============================] - 0s 5ms/step - loss: 0.0101 - categorical_accuracy: 1.0000 - val_loss: 0.2834 - val_categorical_accuracy: 0.9167
    Epoch 472/500
    4/4 [==============================] - 0s 5ms/step - loss: 0.0099 - categorical_accuracy: 1.0000 - val_loss: 0.2813 - val_categorical_accuracy: 0.9167
    Epoch 473/500
    4/4 [==============================] - 0s 5ms/step - loss: 0.0101 - categorical_accuracy: 1.0000 - val_loss: 0.2716 - val_categorical_accuracy: 0.9167
    Epoch 474/500
    4/4 [==============================] - 0s 5ms/step - loss: 0.0101 - categorical_accuracy: 1.0000 - val_loss: 0.2750 - val_categorical_accuracy: 0.9167
    Epoch 475/500
    4/4 [==============================] - 0s 5ms/step - loss: 0.0097 - categorical_accuracy: 1.0000 - val_loss: 0.2828 - val_categorical_accuracy: 0.9167
    Epoch 476/500
    4/4 [==============================] - 0s 5ms/step - loss: 0.0097 - categorical_accuracy: 1.0000 - val_loss: 0.2870 - val_categorical_accuracy: 0.9167
    Epoch 477/500
    4/4 [==============================] - 0s 5ms/step - loss: 0.0097 - categorical_accuracy: 1.0000 - val_loss: 0.2831 - val_categorical_accuracy: 0.9167
    Epoch 478/500
    4/4 [==============================] - 0s 6ms/step - loss: 0.0095 - categorical_accuracy: 1.0000 - val_loss: 0.2766 - val_categorical_accuracy: 0.9167
    Epoch 479/500
    4/4 [==============================] - 0s 5ms/step - loss: 0.0096 - categorical_accuracy: 1.0000 - val_loss: 0.2773 - val_categorical_accuracy: 0.9167
    Epoch 480/500
    4/4 [==============================] - 0s 5ms/step - loss: 0.0096 - categorical_accuracy: 1.0000 - val_loss: 0.2826 - val_categorical_accuracy: 0.9167
    Epoch 481/500
    4/4 [==============================] - 0s 5ms/step - loss: 0.0097 - categorical_accuracy: 1.0000 - val_loss: 0.2874 - val_categorical_accuracy: 0.9167
    Epoch 482/500
    4/4 [==============================] - 0s 5ms/step - loss: 0.0095 - categorical_accuracy: 1.0000 - val_loss: 0.2807 - val_categorical_accuracy: 0.9167
    Epoch 483/500
    4/4 [==============================] - 0s 6ms/step - loss: 0.0096 - categorical_accuracy: 1.0000 - val_loss: 0.2800 - val_categorical_accuracy: 0.9167
    Epoch 484/500
    4/4 [==============================] - 0s 5ms/step - loss: 0.0099 - categorical_accuracy: 1.0000 - val_loss: 0.2904 - val_categorical_accuracy: 0.9167
    Epoch 485/500
    4/4 [==============================] - 0s 5ms/step - loss: 0.0093 - categorical_accuracy: 1.0000 - val_loss: 0.2848 - val_categorical_accuracy: 0.9167
    Epoch 486/500
    4/4 [==============================] - 0s 6ms/step - loss: 0.0092 - categorical_accuracy: 1.0000 - val_loss: 0.2820 - val_categorical_accuracy: 0.9167
    Epoch 487/500
    4/4 [==============================] - 0s 6ms/step - loss: 0.0093 - categorical_accuracy: 1.0000 - val_loss: 0.2840 - val_categorical_accuracy: 0.9167
    Epoch 488/500
    4/4 [==============================] - 0s 7ms/step - loss: 0.0091 - categorical_accuracy: 1.0000 - val_loss: 0.2827 - val_categorical_accuracy: 0.9167
    Epoch 489/500
    4/4 [==============================] - 0s 5ms/step - loss: 0.0091 - categorical_accuracy: 1.0000 - val_loss: 0.2836 - val_categorical_accuracy: 0.9167
    Epoch 490/500
    4/4 [==============================] - 0s 5ms/step - loss: 0.0091 - categorical_accuracy: 1.0000 - val_loss: 0.2851 - val_categorical_accuracy: 0.9167
    Epoch 491/500
    4/4 [==============================] - 0s 5ms/step - loss: 0.0091 - categorical_accuracy: 1.0000 - val_loss: 0.2812 - val_categorical_accuracy: 0.9167
    Epoch 492/500
    4/4 [==============================] - 0s 6ms/step - loss: 0.0092 - categorical_accuracy: 1.0000 - val_loss: 0.2823 - val_categorical_accuracy: 0.9167
    Epoch 493/500
    4/4 [==============================] - 0s 6ms/step - loss: 0.0090 - categorical_accuracy: 1.0000 - val_loss: 0.2877 - val_categorical_accuracy: 0.9167
    Epoch 494/500
    4/4 [==============================] - 0s 6ms/step - loss: 0.0095 - categorical_accuracy: 1.0000 - val_loss: 0.2990 - val_categorical_accuracy: 0.9167
    Epoch 495/500
    4/4 [==============================] - 0s 6ms/step - loss: 0.0089 - categorical_accuracy: 1.0000 - val_loss: 0.2882 - val_categorical_accuracy: 0.9167
    Epoch 496/500
    4/4 [==============================] - 0s 6ms/step - loss: 0.0090 - categorical_accuracy: 1.0000 - val_loss: 0.2807 - val_categorical_accuracy: 0.9167
    Epoch 497/500
    4/4 [==============================] - 0s 6ms/step - loss: 0.0090 - categorical_accuracy: 1.0000 - val_loss: 0.2855 - val_categorical_accuracy: 0.9167
    Epoch 498/500
    4/4 [==============================] - 0s 6ms/step - loss: 0.0089 - categorical_accuracy: 1.0000 - val_loss: 0.2861 - val_categorical_accuracy: 0.9167
    Epoch 499/500
    4/4 [==============================] - 0s 5ms/step - loss: 0.0087 - categorical_accuracy: 1.0000 - val_loss: 0.2903 - val_categorical_accuracy: 0.9167
    Epoch 500/500
    4/4 [==============================] - 0s 5ms/step - loss: 0.0087 - categorical_accuracy: 1.0000 - val_loss: 0.2930 - val_categorical_accuracy: 0.9167


Vamos criar um gráfico com o erro e a acurácia da rede ao longo das épocas de treinamento. Caso uma parcela do conjunto de treinamento tenha sido utilizada para validação, novas linhas no gráfico exibirão o desempenho da rede neste conjunto separado de dados.


```python
def plot_metrics(history):
  fig, axes = plt.subplots(2,1, True,figsize=(8,12))

  #  "Accuracy"
  axes[0].plot(history.history['categorical_accuracy'])
  axes[0].set_title('Acurácia')
  # "Loss"
  axes[1].plot(history.history['loss'])
  axes[1].set_title('Erro')

  if 'val_loss' in history.history.keys():
    axes[0].plot(history.history['val_categorical_accuracy'])
    axes[1].plot(history.history['val_loss'])
    axes[0].legend(['Treino', 'Validação'])
    axes[1].legend(['Treino', 'Validação'])

  plt.xlabel('Épocas')

  plt.show()

plot_metrics(history)
```


![png](../images/output_11_0.png)


### Overfitting

O treino da rede neural por muitas épocas no conjunto de treinamento pode fazer com que o modelo se especialize neste conjunto de dados, um fenômeno chamado *overfitting*.

Essa especialização prejudica o desempenho da rede neural, uma vez que, ao se especializar, ela falha em generalizar sua capacidade de classificação para dados não observados

O *overfitting* pode ser visualizado quando o erro de classificação no conjunto de treinamento continua a diminuir, ao mesmo tempo em que o erro no conjunto de validação começa a aumentar. Este ponto pode ser considerado um bom momento para encerrar o treinamento da rede neural.

Considerando as observações no primeiro treinamento da rede neural, vamos treiná-la novamente, dessa vez ebcerrando o treinamento mais cedo evitando o *overfitting*. Adicionalmente, vamos utilizar o conjunto de treinamento em sua totalidade nesta rodada de treinamento, sem validação


```python
model = create_model()
history = model.fit(X_train, y_train_onehot, epochs=150, verbose=0)
plot_metrics(history)
```


![png](../images/output_13_0.png)


## Realizando inferências com o modelo treinado

Vamos utilizar a rede treinada para classificar os dados de teste e visualizar as classes preditas pelo modelo. A rede neural classifica todo o conjunto de testes de uma vez.


```python
y_pred = model.predict(X_test)
print(y_pred)
```

    [[8.2396846e-03 4.6900004e-01 5.2276021e-01]
     [1.4892984e-03 8.7060414e-02 9.1145027e-01]
     [9.2495506e-04 5.3999864e-02 9.4507515e-01]
     [3.0304896e-02 9.4576114e-01 2.3933986e-02]
     [9.6803588e-01 3.0797675e-02 1.1664870e-03]
     [8.4938472e-03 4.8247686e-01 5.0902927e-01]
     [1.7124804e-02 8.9714748e-01 8.5727692e-02]
     [9.6672821e-01 3.2062069e-02 1.2096572e-03]
     [9.6589357e-01 3.2869406e-02 1.2370276e-03]
     [1.7976636e-02 9.0755904e-01 7.4464321e-02]
     [2.5119220e-03 1.4685105e-01 8.5063702e-01]
     [9.6831447e-01 3.0526053e-02 1.1594761e-03]
     [2.1516284e-02 9.3756503e-01 4.0918734e-02]
     [2.3089170e-03 1.3449498e-01 8.6319613e-01]
     [1.0535454e-03 6.1713681e-02 9.3723273e-01]
     [9.3496195e-04 5.4513000e-02 9.4455200e-01]
     [9.6945411e-01 2.9419001e-02 1.1269355e-03]
     [9.6843040e-01 3.0412735e-02 1.1568622e-03]
     [1.7598620e-02 9.0281260e-01 7.9588786e-02]
     [9.6643317e-01 3.2343209e-02 1.2235941e-03]
     [9.6844238e-01 3.0400287e-02 1.1574024e-03]
     [8.9927334e-03 5.1106507e-01 4.7994223e-01]
     [9.6267527e-01 3.5994362e-02 1.3303764e-03]
     [7.2146221e-03 4.1258824e-01 5.8019710e-01]
     [9.6677101e-01 3.2022767e-02 1.2063318e-03]
     [9.6722925e-01 3.1576350e-02 1.1944455e-03]
     [9.6692634e-01 3.1864643e-02 1.2090066e-03]
     [1.5530937e-03 9.0322651e-02 9.0812427e-01]
     [1.2112972e-03 7.0489652e-02 9.2829907e-01]
     [9.6774071e-01 3.1081626e-02 1.1776128e-03]]


A saída da rede é um conjunto de 3 valores para cada vetor de entrada. Cada um dos 3 valores indica a relevância da respectiva classe para o vetor de entrada. Quanto maior essa relevância, maiores as chances do dado de entrada pertencer àquela classe.

A função de ativação na última camada da rede neural dita a natureza dos valores de saída:

- `tanh`: tangente hiperbólica (intervalo $[-1; 1]$).
- `sigmoid`: sigmoide (intervalo $[0; 1]$).
- `softmax`: _softmax_ (intervalo $[0; 1]$, a soma dos valores preditos para cada dado de entrada deve ser igual a 1).

Para transformar os valores exibidos acima nas classes preditas, escolhemos a maior saída no eixo das classes.

Compare os valores verdadeiros com os valores preditos pela rede.


```python
y_pred_onehot = y_pred.argmax(axis=1)
print('Classes reais:\t\t', y_test)
print('Classes preditas:\t', y_pred_onehot)
```

    Classes reais:		 [1 2 2 1 0 2 1 0 0 1 2 0 1 2 2 2 0 0 1 0 0 2 0 2 0 0 0 2 2 0]
    Classes preditas:	 [2 2 2 1 0 2 1 0 0 1 2 0 1 2 2 2 0 0 1 0 0 1 0 2 0 0 0 2 2 0]


## Avaliando a rede treinada

Utilizamos o *scikit-learn* para produzir relatórios de classificação utilizando os valores reais e os preditos.

As medidas de desempenho da rede são precisão, *recall* (revocação) e F1 [[link]](https://en.wikipedia.org/wiki/Precision_and_recall#Definition_(classification_context).

A matriz de confusão compara, em números absolutos, as classes reais e preditas. Valores na diagonal principal indicam classificações corretas e qualquer outro valor indica erros de classificação no conjunto de testes.


```python
print(classification_report(y_test, y_pred_onehot))
print(confusion_matrix(y_test, y_pred_onehot))
```

                  precision    recall  f1-score   support
    
               0       1.00      1.00      1.00        13
               1       0.83      0.83      0.83         6
               2       0.91      0.91      0.91        11
    
        accuracy                           0.93        30
       macro avg       0.91      0.91      0.91        30
    weighted avg       0.93      0.93      0.93        30
    
    [[13  0  0]
     [ 0  5  1]
     [ 0  1 10]]

