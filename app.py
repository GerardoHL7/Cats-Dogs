import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import os

# 📌 Ruta del modelo guardado (debe estar en la misma carpeta que este script)
RUTA_MODELO = "modeloCNN3.h5"

# 📌 Cargar el modelo
@st.cache_resource
def cargar_modelo():
    if not os.path.exists(RUTA_MODELO):
        st.error("El modelo no se encuentra en la ruta especificada.")
        return None
    return load_model(RUTA_MODELO)

modelo = cargar_modelo()

# 📌 Diccionario de clases
clases = {0: "🐱 Gato", 1: "🐶 Perro"}

# 📌 Interfaz de Streamlit
st.title("Clasificador de Gatos y Perros 🐱🐶")
st.write("Sube una imagen y el modelo intentará clasificarla.")

imagen_subida = st.file_uploader("Sube una imagen...", type=["jpg", "png", "jpeg"])

if imagen_subida is not None:
    # 📌 Cargar y mostrar la imagen
    imagen = Image.open(imagen_subida)
    imagen = imagen.convert("RGB")  # Convertir a RGB
    st.image(imagen, caption="Imagen cargada", use_column_width=True)

    # 📌 Procesar la imagen
    if modelo:
        input_shape = modelo.input_shape  # (None, 100, 100, 1) por ejemplo
        img_size = (input_shape[1], input_shape[2])  # Extraer tamaño esperado

        imagen = imagen.resize(img_size)  # Redimensionar
        imagen = np.array(imagen, dtype=np.float32) / 255.0  # Normalizar

        # 📌 Convertir a escala de grises si el modelo espera solo 1 canal
        if input_shape[3] == 1:
            imagen = np.mean(imagen, axis=-1, keepdims=True)

        # 📌 Expandir dimensiones para que tenga forma (1, altura, ancho, canales)
        imagen = np.expand_dims(imagen, axis=0)

        # 📌 Hacer la predicción
        try:
            prediccion = modelo.predict(imagen)
            clase_predicha = int(prediccion[0][0] > 0.5)  # 0 = Gato, 1 = Perro
            st.write(f"El modelo predice: **{clases[clase_predicha]}**")
        except Exception as e:
            st.error(f"Error en la predicción: {e}")
