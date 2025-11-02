# ----------------------------------------------------------------------------------
#                                    Librerías
# ----------------------------------------------------------------------------------

import os
import re
import cv2
import zipfile
from PIL import Image

import random
from collections import Counter

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.patches import Patch

import seaborn as sns

from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split


import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)

import tensorflow as tf
import tensorflow.keras as keras

import keras_tuner as kt

from tensorflow.keras import models, layers
from tensorflow.keras.utils import to_categorical

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam, RMSprop, SGD
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, LeakyReLU, BatchNormalization

from tensorflow.keras.models import Model
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

from tensorflow.keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator

import warnings
warnings.filterwarnings('ignore')


####################################################################################

# ----------------------------------------------------------------------------------
#                        Carga y preprocesamiento de los datos
# ----------------------------------------------------------------------------------

# Carpetas base
images = "images_tumulos"

# Salidas
tumulos = []
no_tumulos = []

# Extensiones válidas
extensiones = (".jpg", ".jpeg", ".png")

# Carpeta base del dataset
images_base = "images_tumulos"

# ----------------------------------------------------------------------------------

# Función 1: Recolectar imágenes por clase

def recolectar_imagenes(base_path):
    tumulos, no_tumulos = [], []
    for root, _, files in os.walk(base_path):
        root_lower = root.lower()
        for file in files:
            if file.lower().endswith(extensiones):
                ruta = os.path.join(root, file)
                if os.path.basename(root_lower) == "tumulos":
                    tumulos.append(ruta)
                elif os.path.basename(root_lower) == "no_tumulos":
                    no_tumulos.append(ruta)
    return tumulos, no_tumulos

# ----------------------------------------------------------------------------------

# Función 2: Preprocesamiento satelital RGB

def preprocess_satellite_image(path, target_size=(224, 224)):
    """
    Preprocesa una imagen satelital aplicando normalización y mejora de contraste.
    
    Pasos del pipeline:
      1. Carga la imagen desde el archivo
      2. Convierte de BGR (OpenCV) a RGB
      3. Redimensiona a tamaño objetivo (224×224 px)
      4. Ajusta contraste mediante CLAHE en espacio de color LAB
      5. Normaliza valores de píxel al rango [-1, 1] (compatible con MobileNetV2)
    
    Parámetros:
        path (str): Ruta de la imagen a preprocesar.
        target_size (tuple): Dimensiones objetivo (ancho, alto). Default: (224, 224).
    
    Retorna:
        numpy.ndarray: Imagen preprocesada con forma (224, 224, 3).
    
    Raises:
        ValueError: Si la imagen no puede ser leída.
    """
    
    # 1. Cargar imagen
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"No se pudo leer la imagen: {path}")
    
    # 2. Convertir BGR a RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # 3. Redimensionar a tamaño objetivo
    img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
    
    # 4. Ajuste de contraste mediante CLAHE
    # Conversión a espacio de color LAB
    # L = luminancia (brillo), A y B = crominancia (información de color)
    lab = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    
    # CLAHE (Contrast Limited Adaptive Histogram Equalization)
    # Ecualiza el histograma localmente en bloques de 8×8 píxeles
    # Aumenta el contraste de la imagen por bloques para resaltar detalles locales.
    # clipLimit=2.0 limita la amplificación del contraste para reducir ruido
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    
    # Reconstruir imagen con luminancia corregida
    lab = cv2.merge((l, a, b))
    img = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB).astype("float32")
    
    # 5. Normalización al rango [-1, 1]
    # Este rango es compatible con la función de preprocesamiento de MobileNetV2
    # y mejora la convergencia durante el entrenamiento
    img = img / 127.5 - 1.0
    img = img_to_array(img)
    
    return img


# ----------------------------------------------------------------------------------

# Función 3: Procesar listas de imágenes

def procesar_lista_imagenes(rutas):
    procesadas = []
    for ruta in rutas:
        try:
            tensor = preprocess_satellite_image(ruta)
            procesadas.append(tensor)
        except Exception as e:
            print(f"Error procesando {ruta}: {e}")
    return np.array(procesadas)

# ------------------------------------------------------------------------------

# Recolectar imágenes
print("Recolectando imágenes...")
tumulos, no_tumulos = recolectar_imagenes(images)

# Extrae número de "tumuloXX" o "no_tumuloXX" del nombre de archivo
def extraer_numero_archivo(path):
    nombre = os.path.basename(path).lower()
    match = re.search(r'(?:tumulo|no)(\d+)', nombre)
    return int(match.group(1)) if match else float('inf')

# Ordenar por número
tumulos = sorted(tumulos, key=extraer_numero_archivo)
no_tumulos = sorted(no_tumulos, key=extraer_numero_archivo)

# Lista combinada con todas las imágenes
image_paths = tumulos + no_tumulos

print(f"Total imágenes de túmulos: {len(tumulos)}")
print(f"Total imágenes de no_túmulos: {len(no_tumulos)}")

print("\nPreprocesando imágenes...")
tumulos_proc = procesar_lista_imagenes(tumulos)
no_tumulos_proc = procesar_lista_imagenes(no_tumulos)

# Etiquetas
X = np.concatenate([tumulos_proc, no_tumulos_proc], axis=0)
y = np.array([1]*len(tumulos_proc) + [0]*len(no_tumulos_proc))


####################################################################################

# ----------------------------------------------------------------------------------
#           División conjuntos de entrenamiento, validación y prueba
# ----------------------------------------------------------------------------------


# Separación 80% entrenamiento, 10% validación, 10% prueba
X_train, X_temp, y_train, y_temp, paths_train, paths_temp = train_test_split(
    X, y, image_paths, test_size=0.2, random_state=50, stratify=y
)

X_val, X_test, y_val, y_test, paths_val, paths_test = train_test_split(
    X_temp, y_temp, paths_temp, test_size=0.5, random_state=50, stratify=y_temp
)

print('División dataset...')
print(f'train:{len(X_train)} \nval:{len(X_val)} \ntest:{len(X_test)}')

# print("X_train:", X_train.shape)
# print("y_train:", y_train.shape)
# print("X_val:", X_val.shape)
# print("y_val:", y_val.shape)
# print("X_test:", X_test.shape)
# print("y_test:", y_test.shape)


# One-Hot encoding
y_train = to_categorical(y_train, num_classes=2)
y_val = to_categorical(y_val, num_classes=2)
y_test = to_categorical(y_test, num_classes=2)


####################################################################################

# ----------------------------------------------------------------------------------
#                  Generador de aumento de datos / Data Augmentation
# ----------------------------------------------------------------------------------

# Generador con aumentación solo para entrenamiento
train_datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator()
test_datagen = ImageDataGenerator()


# Crear flujos de datos
train_generator = train_datagen.flow(X_train, y_train, batch_size=32, shuffle=True)
val_generator = val_datagen.flow(X_val, y_val, batch_size=32, shuffle=False)
test_generator = test_datagen.flow(X_test, y_test, batch_size=32, shuffle=False)


####################################################################################

# ----------------------------------------------------------------------------------
#                             Construcción Keras Tuner
# ----------------------------------------------------------------------------------

# 1) Función para construir el modelo

def build_model(hp):
    base_model = MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights="imagenet"
    )

    # Capas a descongelar (fine-tuning)
    fine_tune_layers = hp.Choice('fine_tune_layers', values=[0, 30, 50, 80, 100])
    if fine_tune_layers == 0:
        base_model.trainable = False
    else:
        base_model.trainable = True
        for layer in base_model.layers[:-fine_tune_layers]:
            layer.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)

    # Capa densa intermedia
    hp_units = hp.Int('units', min_value=64, max_value=512, step=64)
    hp_dropout = hp.Float('dropout', min_value=0.2, max_value=0.6, step=0.1)
    hp_activation = hp.Choice('activation', values=['relu', 'elu', 'leaky_relu','tanh','sigmoid'])

    x = Dense(hp_units)(x)
    if hp_activation == 'leaky_relu':
        x = LeakyReLU(alpha=0.1)(x)
    else:
        x = tf.keras.layers.Activation(hp_activation)(x)

    x = Dropout(hp_dropout)(x)
    outputs = Dense(2, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=outputs)

    
    # Optimizador y tasa de aprendizaje
    optimizer_choice = hp.Choice('optimizer', values=['adam', 'rmsprop', 'sgd'])
    lr = hp.Choice('learning_rate', values=[1e-4, 3e-4, 1e-3])

    if optimizer_choice == 'adam':
        optimizer = Adam(learning_rate=lr)
    elif optimizer_choice == 'rmsprop':
        optimizer = RMSprop(learning_rate=lr)
    else:
        momentum = hp.Float('momentum', 0.0, 0.9, step=0.3)
        optimizer = SGD(learning_rate=lr, momentum=momentum)

    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


# 2) Configuración del tuner (búsqueda de hiperparámetros)

tuner = kt.Hyperband(
    build_model,
    objective='val_accuracy',
    max_epochs=20, 
    factor=3,   # Controla la tasa de reducción de modelos entre rondas de Hyperband.
                # Con factor=3, solo un tercio de los mejores modelos avanza en cada fase.
    directory='tuner_results4',
    project_name='mobilenetv2_full_tuning4'
)


# 3) Callback para detener entrenamiento si no mejora

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True
)


# 4) Añadir búsqueda de batch size

# Keras-tuner no ajusta el batch size dentro de `build_model()`, 
# por lo que su búsqueda para obtener el valor óptimo se realiza 
# al momento de hacer `tuner.search()`

batch_size_values = [16, 32, 64]

best_overall_hp = None
best_overall_val_acc = 0
best_batch_size = None

for bs in batch_size_values:
    print(f"\n🔍 Probando batch size = {bs}\n")

    tuner.search(
        train_generator,
        validation_data=val_generator,
        epochs=20,
        batch_size=bs,
        callbacks=[early_stop],
        verbose=1
    )

    best_hp = tuner.get_best_hyperparameters(1)[0]
    best_model = tuner.hypermodel.build(best_hp)

    _, val_acc = best_model.evaluate(val_generator, verbose=0)
    if val_acc > best_overall_val_acc:
        best_overall_hp = best_hp
        best_overall_val_acc = val_acc
        best_batch_size = bs


# 5) Mostrar los mejores hiperparámetros

print("\n Mejores hiperparámetros encontrados:")
print(f" - Batch size: {best_batch_size}")
print(f" - Unidades densas: {best_overall_hp.get('units')}")
print(f" - Dropout: {best_overall_hp.get('dropout')}")
print(f" - Activación: {best_overall_hp.get('activation')}")
print(f" - Fine-tuning: últimas {best_overall_hp.get('fine_tune_layers')} capas desbloqueadas")
print(f" - Optimizador: {best_overall_hp.get('optimizer')}")
print(f" - Learning rate: {best_overall_hp.get('learning_rate')}")
if best_overall_hp.get('optimizer') == 'sgd':
    print(f" - Momentum: {best_overall_hp.get('momentum')}")


####################################################################################

# ----------------------------------------------------------------------------------
#               Entrenamiento del modelo con los mejores hiperparámetros
# ----------------------------------------------------------------------------------

# Callback para detener entrenamiento si no mejora
early_stopper = EarlyStopping(
    monitor='val_loss',
    patience=3,
    mode='min',
    restore_best_weights=True
)

# Reconstruir el modelo con los mejores HP
modelo = tuner.hypermodel.build(best_hp)

# Entrenar el modelo
history = modelo.fit(
    train_generator,
    validation_data=val_generator,
    epochs=30,
    batch_size=best_batch_size,
    callbacks=[early_stop],
    verbose=1
)

# Evaluación del modelo en Validación
val_loss, val_acc = modelo.evaluate(val_generator)
print(f"Accuracy validación: {val_acc:.4f}")
print(f"Loss en validación: {val_loss:.4f}")

# Evaluación del modelo en Test
test_loss, test_acc = modelo.evaluate(test_generator)
print(f"Accuracy test: {test_acc:.4f}")
print(f"Loss en test: {test_loss:.4f}")

####################################################################################

# ----------------------------------------------------------------------------------
#          Guardado del modelo entrenado en formato .h5 para uso posterior
# ----------------------------------------------------------------------------------

modelo.save("deteccion_tumulos.h5")
