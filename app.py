import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import gdown
import os

# 📌 Reemplaza con el ID de tu modelo en Google Drive
ID_MODELO = "1-3xbX688NVdwm1g50x3f9IlOVCTe-2gh"
URL_MODELO = f"https://drive.google.com/uc?id={ID_MODELO}"
RUTA_MODELO = "modeloCNN3.h5"

# 📥 Descargar el modelo si no existe
if not os.path.exists(RUTA_MODELO):
    with st.spinner("Descargando modelo... Esto puede tardar un momento ⏳"):
        gdown.download(URL_MODELO, RUTA_MODELO, quiet=False)

# 🚀 Cargar el modelo solo cuando sea necesario
@st.cache_resource
def cargar_modelo():
    return tf.keras.models.load_model(RUTA_MODELO)

modelo = cargar_modelo()

# 📌 Función para preprocesar la imagen
def preprocesar_imagen(imagen):
    imagen = imagen.convert("L")  # Convertir a escala de grises
    imagen = imagen.resize((100, 100))  # Redimensionar
    imagen = np.array(imagen) / 255.0  # Normalización
    imagen = np.expand_dims(imagen, axis=-1)  # Añadir canal de profundidad
    imagen = np.expand_dims(imagen, axis=0)  # Añadir dimensión batch
    return imagen

# 🎨 Interfaz Streamlit
st.title("Clasificador de Perros y Gatos 🐶🐱")
st.write("Sube una imagen y el modelo te dirá si es un perro o un gato.")

archivo_subido = st.file_uploader("Sube una imagen...", type=["jpg", "png", "jpeg"])

if archivo_subido is not None:
    imagen = Image.open(archivo_subido)
    st.image(imagen, caption="Imagen subida", use_column_width=True)

    # 🔄 Preprocesar la imagen
    imagen_procesada = preprocesar_imagen(imagen)

    # 🧠 Hacer predicción
    prediccion = modelo.predict(imagen_procesada)[0][0]

    # 📌 Mostrar resultado
    if prediccion > 0.5:
        st.success("¡Es un 🐶 **PERRO**!")
    else:
        st.success("¡Es un 🐱 **GATO**!")

st.write("📌 Modelo basado en una red neuronal convolucional (CNN) entrenada con TensorFlow/Keras.")
