import tensorflow as tf
import tensorflow_datasets as tfds
import time
import cv2
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import keras_tuner as kt
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

### Time consuming method:
# Define model building function
### We use CNN structure
# def build_model_cnn(hp):
#     model = Sequential()
#     model.add(Input(shape=(100, 100, 1)))  # Input shape fixed for images

#     # Tune the number of Conv2D filters and kernel size
#     for i in range(hp.Int("num_conv_layers", 2, 4)):  # 2 to 4 conv layers
#         model.add(Conv2D(
#             filters=hp.Int(f"filters_{i}", 32, 128, step=32),  # 32, 64, 96, 128
#             kernel_size=hp.Choice(f"kernel_size_{i}", [(3,3), (5,5)]),  # 3x3 or 5x5
#             activation="relu",
#             padding="same"
#         ))
#         model.add(MaxPooling2D((2, 2)))  # Fixed pooling

#     model.add(Flatten())

#     # Tune the number of Dense units and dropout rate
#     for i in range(hp.Int("num_dense_layers", 1, 2)):  # 1 or 2 dense layers
#         model.add(Dense(
#             units=hp.Int(f"dense_units_{i}", 50, 200, step=50),  # 50, 100, 150, 200
#             activation="relu"
#         ))
#         model.add(Dropout(
#             rate=hp.Float(f"dropout_{i}", 0.3, 0.6, step=0.1)  # 0.3, 0.4, 0.5, 0.6
#         ))

#     model.add(Dense(1, activation="sigmoid"))  # Binary classification output

#     # Tune learning rate and optimizer
#     lr = hp.Choice("lr", [1e-3, 1e-4, 1e-5])
#     optimizer_name = hp.Choice("optimizer", ["adam", "rmsprop", "sgd"])
#     if optimizer_name == "adam":
#         optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
#     elif optimizer_name == "rmsprop":
#         optimizer = tf.keras.optimizers.RMSprop(learning_rate=lr)
#     else:
#         optimizer = tf.keras.optimizers.SGD(learning_rate=lr, momentum=0.9)

#     model.compile(
#         optimizer=optimizer,
#         loss="binary_crossentropy",
#         metrics=["accuracy"]
#     )
    
#     return model

# # Hyperparameter tuning with balanced data
# tuner = kt.RandomSearch(
#     build_model_cnn,
#     objective='val_accuracy',
#     executions_per_trial=1,
#     directory='salida',
#     project_name='intro_to_HP',
#     overwrite=True
# )

# early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# tuner.search(
#     X_train, y_train,
#     validation_data=(X_test, y_test),
#     epochs=100,
#     batch_size=32,
#     callbacks=[early_stopping],
#     verbose=0
# )

# # Get best model and train on FULL balanced dataset
# best_hps = tuner.get_best_hyperparameters()[0]
# best_model = tuner.hypermodel.build(best_hps)

# history = best_model.fit(
#     X_train, y_train,  # Use balanced data consistently
#     validation_data=(X_test, y_test),
#     epochs=100,
#     verbose=0
# )
    

### Modelo CNN con dropout

modeloCNN = Sequential([
  Input( shape=(100, 100, 1) ),

  Conv2D( 32, (3,3), activation='relu' ),
  MaxPooling2D( 2, 2 ),
  Conv2D( 64, (3,3), activation='relu' ),
  MaxPooling2D( 2, 2 ),
  Conv2D( 128, (3,3), activation='relu' ),
  MaxPooling2D( 2, 2 ),

  Flatten(),
  Dense(100, activation='relu'),
  Dropout(0.5),  # Drops 50% of neurons randomly
  Dense(1, activation='sigmoid')
])

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

### Se guarda el modelo en h5
modeloCNN.save('cnn_perros_gatos.h5')

### Graficas
plt.figure( figsize=(20,5))

plt.subplot(1,2,1)
plt.plot(history_CNN1.epoch, history_CNN1.history['loss'], 'b',label='loss')
plt.plot(history_CNN1.epoch, history_CNN1.history['val_loss'], 'k',label='val_loss')
plt.title(u'CNN loss')
plt.xlabel(u'época')
plt.ylabel(r'$loss$')
plt.ylim(0, max(max(history_CNN1.history['loss']),max(history_CNN1.history['val_loss'])))
plt.grid()
plt.legend(loc='upper right')


plt.subplot(1,2,2)
plt.plot(history_CNN1.epoch, history_CNN1.history['accuracy'], 'b',label='accuracy')
plt.plot(history_CNN1.epoch, history_CNN1.history['val_accuracy'], 'k',label='val_accuracy')
plt.title(u'CNN accuracy')
plt.xlabel(u'época')
plt.ylabel(r'$accuracy$')
plt.ylim(0,1)
plt.grid()
plt.legend(loc='lower right')
plt.savefig('cnn_graphs.png')
