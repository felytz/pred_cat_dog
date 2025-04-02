import tensorflow as tf
import tensorflow_datasets as tfds
import time
import cv2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
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

### Modelo CNN con dropout
modeloCNN = Sequential(name="sequential")  # Important: Match this with TF.js expectations

# Input layer with explicit name
modeloCNN.add(Input(shape=(100, 100, 1), name="input_layer"))

# Conv/Pooling blocks with unique names
modeloCNN.add(Conv2D(32, (3,3), activation='relu', name="conv2d_1"))
modeloCNN.add(MaxPooling2D((2,2), name="maxpool2d_1"))

modeloCNN.add(Conv2D(64, (3,3), activation='relu', name="conv2d_2"))
modeloCNN.add(MaxPooling2D((2,2), name="maxpool2d_2"))

modeloCNN.add(Conv2D(128, (3,3), activation='relu', name="conv2d_3"))
modeloCNN.add(MaxPooling2D((2,2), name="maxpool2d_3"))

# Classifier head
modeloCNN.add(Flatten(name="flatten"))
modeloCNN.add(Dense(100, activation='relu', name="dense_1"))
modeloCNN.add(Dropout(0.5, name="dropout_1"))
modeloCNN.add(Dense(1, activation='sigmoid', name="dense_output"))

modeloCNN.compile(optimizer='adam',
                    loss='binary_crossentropy',
                    metrics=['accuracy'])
                    
#Estimado 1 h y media

history_CNN1 = modeloCNN.fit(
        X_train, y_train,  
        validation_data=(X_test, y_test),
        epochs=50,
        batch_size=32,
        verbose=0
    )
    
modeloCNN.save('cnn_model.keras') 
modeloCNN.export('cnn_model')

# Gráfica de Accuracy
plt.figure(figsize=(8, 6))
plt.plot(history_CNN1.history['accuracy'], 'b-', label='Train Accuracy')
plt.plot(history_CNN1.history['val_accuracy'], 'r-', label='Validation Accuracy')
plt.title('Model Accuracy - CNN_AD')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.ylim([0, 1.1])
plt.legend(loc='lower right')
plt.grid(True)
plt.savefig('cnn_acc_graph.png', bbox_inches='tight', dpi=300)
plt.close()

# Gráfica de Loss
plt.figure(figsize=(8, 6))
plt.plot(history_CNN1.history['loss'], 'b-', label='Train Loss')
plt.plot(history_CNN1.history['val_loss'], 'r-', label='Validation Loss')
plt.title('Model Loss - CNN_AD')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.ylim([0, 1])
plt.legend(loc='upper right')
plt.grid(True)
plt.savefig('cnn_loss_graph.png', bbox_inches='tight', dpi=300)
plt.close()