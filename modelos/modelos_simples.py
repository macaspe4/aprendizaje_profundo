# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow.keras import layers, models

def crear_red_simple(input_shape):
    """
    Crea una red neuronal simple para clasificación.
    Recibe: input_shape (número de columnas de X)
    Devuelve: El modelo compilado
    """
    model = models.Sequential([
        layers.Input(shape=(input_shape,)),
        layers.Dense(3, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model
