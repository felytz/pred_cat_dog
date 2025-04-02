import time
import numpy as np
import datetime
import zoneinfo
import matplotlib.pyplot as plt
import cv2
from sklearn.metrics import confusion_matrix
from sklearn.metrics import  ConfusionMatrixDisplay
#import tensorrt
import tensorflow as tf
import keras
import tensorflow_datasets as tfds
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Random control
np.random.seed(42)
tf.random.set_seed(42)

setattr(tfds.image_classification.cats_vs_dogs, '_URL',"https://download.microsoft.com/download/3/E/1/3E1C3F21-ECDB-4869-8368-6DEBA77B919F/kagglecatsanddogs_5340.zip")

#Descargar el set de datos de perros y gatos
datos, metadatos = tfds.load('cats_vs_dogs', as_supervised=True, with_info=True)

TAMANO_IMG=100

datos_entrenamiento = []

for i, (imagen, etiqueta) in enumerate(datos['train']): #Todos los datos de entrenamiento
  imagen = cv2.resize(imagen.numpy(), (TAMANO_IMG, TAMANO_IMG))
  imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
  imagen = imagen.reshape(TAMANO_IMG, TAMANO_IMG, 1) #Cambiar tamano a 100,100,1
  datos_entrenamiento.append([imagen, etiqueta])
  
#Preparar mis variables X (entradas) y y (etiquetas) separadas
X = [] #imagenes de entrada (pixeles)
y = [] #etiquetas (perro o gato)

for imagen, etiqueta in datos_entrenamiento:
  X.append(imagen)
  y.append(etiqueta)
  
#Normalizar los datos de las X (imagenes). Se pasan a numero flotante y dividen entre 255 para quedar de 0-1 en lugar de 0-255
X = np.array(X).astype(float) / 255  # <---- PENDIENTE: cambiar a una capa del modelo

#Convertir etiquetas en arreglo simple
y = np.array(y)

unique_elements, count_elements = np.unique(y,return_counts=True)



#Realizar el aumento de datos con varias transformaciones. Al final, se grafican 6 para ejemplificar

datagen = ImageDataGenerator(
    rotation_range     = 20,
    width_shift_range  = 0.2, # mover la imagen a los lados
    height_shift_range = 0.2,
    shear_range        = 15, # inclinar la imagen
    zoom_range         = [0.8, 1.2],
    horizontal_flip    = True,
    vertical_flip      = False
)

datagen.fit(X)

#Separar los datos de entrenamiento y los datos de pruebas en variables diferentes

len(X) * 0.85 # <-- corresponde a 19,700 imágenes (TRAIN)
len(X) * 0.15 # <-- corresponde a  3,562 imágenes (VALIDATION)

X_entrenamiento = X[:19700]
X_validacion = X[19700:]

y_entrenamiento = y[:19700]
y_validacion = y[19700:]

#Usar la funcion flow del generador para crear un iterador que podamos enviar como entrenamiento a la funcion FIT del modelo
data_gen_entrenamiento = datagen.flow( X_entrenamiento, y_entrenamiento, batch_size=32, shuffle=True  )
data_gen_validacion    = datagen.flow( X_validacion,    y_validacion,    batch_size=32, shuffle=False )


modeloCNN_AD = tf.keras.models.Sequential([
  tf.keras.layers.Input(shape=(100, 100, 1)),

  tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
  tf.keras.layers.MaxPooling2D(2, 2),
  tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
  tf.keras.layers.MaxPooling2D(2, 2),
  tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
  tf.keras.layers.MaxPooling2D(2, 2),

  tf.keras.layers.Dropout(0.5),

  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(250, activation='relu'),
  tf.keras.layers.Dense(1, activation='sigmoid')
])

modeloCNN_AD.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

hist_CNN_AD = modeloCNN_AD.fit( data_gen_entrenamiento, epochs=20,  batch_size=32, validation_data=data_gen_validacion, verbose=1 )

# Gráfica de Accuracy
plt.figure(figsize=(8, 6))
plt.plot(hist_CNN_AD.history['accuracy'], 'b-', label='Train Accuracy')
plt.plot(hist_CNN_AD.history['val_accuracy'], 'r-', label='Validation Accuracy')
plt.title('Model Accuracy - CNN_AD')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.ylim([0, 1.1])
plt.legend(loc='lower right')
plt.grid(True)
plt.savefig('cnn_ad_acc_graph.png', bbox_inches='tight', dpi=300)
plt.close()

# Gráfica de Loss
plt.figure(figsize=(8, 6))
plt.plot(hist_CNN_AD.history['loss'], 'b-', label='Train Loss')
plt.plot(hist_CNN_AD.history['val_loss'], 'r-', label='Validation Loss')
plt.title('Model Loss - CNN_AD')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.ylim([0, 1])
plt.legend(loc='upper right')
plt.grid(True)
plt.savefig('cnn_ad_loss_graph.png', bbox_inches='tight', dpi=300)
plt.close()

evaluacion = modeloCNN_AD.evaluate(X_validacion,y_validacion)
evaluacion

prediccion = modeloCNN_AD.predict(X_validacion)

con = confusion_matrix(y_validacion, np.round(prediccion))

disp = ConfusionMatrixDisplay( confusion_matrix = con ).plot()
plt.savefig('cnn_ad_matrix.png')

modeloCNN_AD.save('cnn_ad_model.keras') 
modeloCNN_AD.export('cnn_ad_model')



