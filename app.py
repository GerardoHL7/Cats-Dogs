import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import gdown
import os

# 📌 ID del modelo en Google Drive (REEMPLÁZALO CON TU ID CORRECTO)
ID_MODELO = "1Epg3b63zLXUXbUmZcCpQucJVBziJW9GR"
URL_MODELO = f"https://drive.google.com/uc?id={ID_MODELO}"
RUTA_MODELO = "modeloCNN3.h5"

# 🔄 Descargar el modelo si no existe
if not os.path.exists(RUTA_MODELO):
    with st.spinner("📥 Descargando modelo... Esto puede tardar un momento ⏳"):
        gdown.download(URL_MODELO, RUTA_MODELO, quiet=False)
    st.success("✅ Modelo descargado exitosamente.")

# 🧠 Cargar el modelo con caché para mejorar rendimiento
@st.cache_resource
def cargar_modelo():
    modelo = load_model(RUTA_MODELO)
    return modelo

# 📌 Diccionario de clases
clases = {0: "🐱 Gato", 1: "🐶 Perro"}

# 🎨 Interfaz de Streamlit
st.title("Clasificación de Imágenes: ¿Perro o Gato? 🐾")
st.write("Sube una imagen y el modelo te dirá si es un **gato** o un **perro**.")

# 📤 Cargar imagen
imagen_subida = st.file_uploader("📤 Sube una imagen...", type=["jpg", "png", "jpeg"])

if imagen_subida is not None:
    # Mostrar la imagen subida
    imagen = Image.open(imagen_subida)
    st.image(imagen, caption="📷 Imagen cargada", use_column_width=True)

    # 🔄 Cargar el modelo
    modelo = cargar_modelo()
    input_shape = modelo.input_shape  # Obtener la forma de entrada esperada por el modelo

    # 🔄 Redimensionar la imagen al tamaño esperado por el modelo
    imagen = imagen.convert("L")  # Convertir a escala de grises si es necesario
    imagen = imagen.resize((input_shape[1], input_shape[2]))  # Redimensionar la imagen

    # 🔄 Convertir a array numpy y normalizar (valores entre 0 y 1)
    imagen_array = np.array(imagen, dtype=np.float32) / 255.0
    imagen_array = np.expand_dims(imagen_array, axis=0)  # Añadir la dimensión batch
    imagen_array = np.expand_dims(imagen_array, axis=-1)  # Añadir la dimensión de canal si es necesario

    # 🔍 Hacer la predicción
    try:
        prediccion = modelo.predict(imagen_array)[0][0]
        clase_predicha = 1 if prediccion > 0.5 else 0  # Si >0.5 es perro, si <0.5 es gato

        # 🎯 Mostrar el resultado
        st.markdown(f"## **Predicción: {clases[clase_predicha]}**")

    except Exception as e:
        st.error(f"❌ Error al hacer la predicción: {e}")
