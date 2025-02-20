import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Función para cargar el modelo con caché en Streamlit
@st.cache_resource
def cargar_modelo():
    modelo = tf.keras.models.load_model("modeloCNN3.h5")
    return modelo

modelo = cargar_modelo()

# Configuración de la interfaz
st.title("Clasificación de Imágenes: Perro 🐶 o Gato 🐱")
st.write("Sube una imagen y el modelo la clasificará.")

# Subir la imagen
imagen_subida = st.file_uploader("Elige una imagen...", type=["jpg", "png", "jpeg"])

if imagen_subida is not None:
    # Cargar y preprocesar la imagen
    imagen = Image.open(imagen_subida).convert("L")  # Convertir a escala de grises
    imagen = imagen.resize((100, 100))  # Redimensionar
    imagen = np.array(imagen) / 255.0  # Normalizar
    imagen = imagen.reshape(1, 100, 100, 1)  # Ajustar formato

    # Realizar la predicción
    prediccion = modelo.predict(imagen)[0][0]

    # Determinar si es un perro o un gato
    if prediccion > 0.5:
        resultado = "🐶 Perro"
    else:
        resultado = "🐱 Gato"

    # Mostrar la imagen y el resultado
    st.image(imagen_subida, caption="Imagen Cargada", use_column_width=True)
    st.write(f"### **Predicción: {resultado}**")

