import streamlit as st
import tensorflow as tf
import gdown
import os
import numpy as np
from PIL import Image

# Configuración de la app
st.title("Clasificación de Imágenes: Perro 🐶 o Gato 🐱")
st.write("Sube una imagen y el modelo la clasificará.")

# URL del modelo en Google Drive (reemplázala con tu ID correcto)
url = ""1Epg3b63zLXUXbUmZcCpQucJVBziJW9GR"
modelo_path = "modeloCNN3.h5"

# Descargar el modelo si no existe
if not os.path.exists(modelo_path):
    st.write("🔄 Descargando el modelo, por favor espera...")
    try:
        gdown.download(url, modelo_path, quiet=False)
        st.success("✅ Modelo descargado correctamente.")
    except Exception as e:
        st.error("❌ Error al descargar el modelo. Revisa la URL o intenta de nuevo.")
        st.stop()

# Cargar el modelo con caché para optimizar rendimiento
@st.cache_resource
def cargar_modelo():
    try:
        modelo = tf.keras.models.load_model(modelo_path)
        st.success("✅ Modelo cargado correctamente.")
        return modelo
    except Exception as e:
        st.error("❌ Error al cargar el modelo.")
        st.stop()

modelo = cargar_modelo()

# Subir imagen
imagen_subida = st.file_uploader("📤 Elige una imagen...", type=["jpg", "png", "jpeg"])

if imagen_subida is not None:
    # Cargar y preprocesar la imagen
    imagen = Image.open(imagen_subida).convert("L")  # Convertir a escala de grises
    imagen = imagen.resize((100, 100))  # Redimensionar
    imagen = np.array(imagen) / 255.0  # Normalizar
    imagen = imagen.reshape(1, 100, 100, 1)  # Ajustar formato

    # Mostrar la imagen subida
    st.image(imagen_subida, caption="Imagen Cargada", use_column_width=True)

    # Realizar la predicción
    prediccion = modelo.predict(imagen)[0][0]

    # Determinar si es un perro o un gato
    resultado = "🐶 Perro" if prediccion > 0.5 else "🐱 Gato"

    # Mostrar el resultado
    st.markdown(f"## **Predicción: {resultado}**")


