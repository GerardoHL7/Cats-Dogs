import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import gdown
import os

# 📌 Enlace de Google Drive con el modelo (cambia esto con el ID real de tu modelo)
ID_MODELO = "1Epg3b63zLXUXbUmZcCpQucJVBziJW9GR"
URL_MODELO = f"https://drive.google.com/uc?id={ID_MODELO}"
RUTA_MODELO = "modelo_gatos_perros.h5"

# 📌 Descargar el modelo si no existe
if not os.path.exists(RUTA_MODELO):
    with st.spinner("Descargando modelo... Esto puede tardar un momento ⏳"):
        gdown.download(URL_MODELO, RUTA_MODELO, quiet=False)

# 📌 Cargar el modelo solo cuando sea necesario
@st.cache_resource
def cargar_modelo():
    modelo = load_model(RUTA_MODELO)
    return modelo

# 📌 Diccionario de clases
clases = {0: "🐱 Gato", 1: "🐶 Perro"}

# 📌 Interfaz de Streamlit
st.title("Clasificador de Gatos y Perros 🐱🐶")
st.write("Sube una imagen y el modelo la clasificará como gato o perro.")

imagen_subida = st.file_uploader("Sube una imagen...", type=["jpg", "png", "jpeg"])

if imagen_subida is not None:
    # Cargar y mostrar la imagen
    imagen = Image.open(imagen_subida)
    imagen = imagen.convert("RGB")  # Asegurarse de que sea RGB
    st.image(imagen, caption="Imagen cargada", use_column_width=True)

    # Redimensionar la imagen según el modelo
    modelo = cargar_modelo()
    input_shape = modelo.input_shape  # (None, 100, 100, 1)
    img_size = (input_shape[1], input_shape[2])
    
    imagen = imagen.resize(img_size)  # Ajustar tamaño
    imagen = np.array(imagen, dtype=np.float32) / 255.0  # Normalizar

    # Si el modelo espera escala de grises, convertir la imagen
    if input_shape[3] == 1:
        imagen = np.mean(imagen, axis=-1, keepdims=True)  # Convertir a escala de grises

    # Expandir dimensiones para que tenga la forma correcta (1, height, width, channels)
    imagen = np.expand_dims(imagen, axis=0)

    # Hacer la predicción
    try:
        prediccion = modelo.predict(imagen)
        clase_predicha = int(prediccion[0][0] > 0.5)  # 0 = Gato, 1 = Perro

        # Mostrar resultado
        st.write(f"El modelo predice: **{clases[clase_predicha]}**")
    except Exception as e:
        st.error(f"Ha ocurrido un error durante la predicción: {e}")

