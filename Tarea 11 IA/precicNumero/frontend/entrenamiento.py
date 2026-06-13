import os
import tensorflow as tf
import numpy as np

# 1. Cargar y normalizar el dataset MNIST
print("Cargando el dataset MNIST...")
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train[..., np.newaxis].astype("float32") / 255.0
x_test = x_test[..., np.newaxis].astype("float32") / 255.0

# 2. Definir una arquitectura CNN robusta compatible con TF.js
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 3. Entrenar el modelo (10 épocas garantizan >98.5% de precisión sin sobreentrenar)
print("\nIniciando entrenamiento del modelo...")
model.fit(x_train, y_train, epochs=10, batch_size=64, validation_data=(x_test, y_test))

# 4. Guardar modelo nativo de Keras (.h5) por si acaso
model.save("modelo_mnist.h5")
print("\n¡Modelo Keras guardado como 'modelo_mnist.h5'!")

# 5. CONVERSIÓN CRÍTICA A TENSORFLOW.JS
print("\nConvirtiendo el modelo al formato de TensorFlow.js...")
import tensorflowjs as tfjs

output_folder = "modelo_web"
tfjs.converters.save_keras_model(model, output_folder)

print(f"\n" + "="*60)
print(f"¡PROCESO COMPLETADO EXCELENTEMENTE!")
print(f"Se ha creado la carpeta '{output_folder}/' con los fragmentos binarios.")
print(f"Ya puedes cerrar Python por completo.")
print("="*60 + "\n")