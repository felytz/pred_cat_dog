import tensorflow as tf
import tensorflow_datasets as tfds
import time
import cv2
from tensorflow import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.models import Sequential
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

t = time.time()

### Descargar el set de datos de perros y gatos ###
setattr(tfds.image_classification.cats_vs_dogs, '_URL',"https://download.microsoft.com/download/3/E/1/3E1C3F21-ECDB-4869-8368-6DEBA77B919F/kagglecatsanddogs_5340.zip")

datos, metadatos = tfds.load('cats_vs_dogs', as_supervised=True, with_info=True)

### Preparacion de datos ###
datos_entrenamiento = []
TAMANO_IMG=100
for i, (imagen, etiqueta) in enumerate(datos['train']): #Todos los datos
    # cambiar calidad
    imagen = cv2.resize(imagen.numpy(), (TAMANO_IMG, TAMANO_IMG))
    # escala de grises
    imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    # cambiar de tamaño
    imagen = imagen.reshape(TAMANO_IMG, TAMANO_IMG, 1) #Cambiar tamano a 100,100,1
    datos_entrenamiento.append([imagen, etiqueta])

X = [] #imagenes de entrada (pixeles)
y = [] #etiquetas (perro o gato)
for imagen, etiqueta in datos_entrenamiento:
  X.append(imagen)
  y.append(etiqueta)

# Normalizar los datos de las X (imagenes). Se pasan a numero flotante y dividen entre 255 para quedar de 0-1 en lugar de 0-255
X = np.array(X).astype(float) / 255
# Convertir etiquetas en arreglo simple
y = np.array(y)

# Dividir en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)

### Modelo Regular con dropout

modeloDenso1 = keras.models.Sequential([
    Input(shape=(100, 100, 1)),
    Flatten(),
    
    Dense(150, activation='relu'),
    Dropout(0.5),  # 50% dropout after first dense layer
    
    Dense(150, activation='relu'),
    Dropout(0.5),  # 50% dropout after second dense layer
    
    Dense(50, activation='relu'),
    Dropout(0.3),  # 30% dropout (smaller since it's closer to output)
    
    Dense(1, activation='sigmoid')
])

modeloDenso1.compile(optimizer='adam',
                    loss='binary_crossentropy',
                    metrics=['accuracy'])
                    
history_Denso1 = modeloDenso1.fit(
        X_train, y_train,  
        validation_data=(X_test, y_test),
        epochs=50,
        batch_size=32,
        verbose=0
    )

modeloDenso1.build(input_shape=(None, 100, 100, 1))  # Explicitly build with input shape
modeloDenso1.save('dense_perros_gatos.h5')  # Save with shape info embedded

### Graficas
plt.figure( figsize=(20,5))

plt.subplot(1,2,1)
plt.plot(history_Denso1.epoch, history_Denso1.history['loss'], 'b',label='loss')
plt.plot(history_Denso1.epoch, history_Denso1.history['val_loss'], 'k',label='val_loss')
plt.title(u'Regular loss')
plt.xlabel(u'época')
plt.ylabel(r'$loss$')
plt.ylim(0, max(max(history_Denso1.history['loss']),max(history_Denso1.history['val_loss'])))
plt.grid()
plt.legend(loc='upper right')


plt.subplot(1,2,2)
plt.plot(history_Denso1.epoch, history_Denso1.history['accuracy'], 'b',label='accuracy')
plt.plot(history_Denso1.epoch, history_Denso1.history['val_accuracy'], 'k',label='val_accuracy')
plt.title(u'Regular accuracy')
plt.xlabel(u'época')
plt.ylabel(r'$accuracy$')
plt.ylim(0,1)
plt.grid()
plt.legend(loc='lower right')
plt.savefig('dense_graphs.png')
