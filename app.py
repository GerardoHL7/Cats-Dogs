import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import gdown
import os

# Enlace de Google Drive con el modelo (reemplaza con tu ID de archivo)
ID_MODELO = "1-TdpYJNCcDv8nqHuUmAfMUdjp_psJgPD"
URL_MODELO = f"https://drive.google.com/uc?id={ID_MODELO}"
RUTA_MODELO = "modeloCNN3.h5"

# Descargar el modelo si no existe
if not os.path.exists(RUTA_MODELO):
    with st.spinner("Descargando modelo... Esto puede tardar un momento ⏳"):
        gdown.download(URL_MODELO, RUTA_MODELO, quiet=False)

# Cargar el modelo solo cuando sea necesario
@st.cache_resource
def cargar_modelo():
    modelo = load_model(RUTA_MODELO)
    return modelo

# Función para preprocesar la imagen
def preprocesar_imagen(imagen):
    imagen = imagen.convert("L")  # Convertir a escala de grises
    imagen = imagen.resize((100, 100))  # Redimensionar
    imagen = np.array(imagen) / 255.0  # Normalización
    imagen = np.expand_dims(imagen, axis=-1)  # Añadir canal de profundidad
    imagen = np.expand_dims(imagen, axis=0)  # Añadir dimensión batch
    return imagen

# Interfaz de Streamlit
st.title("Clasificador de Perros y Gatos 🐶🐱")
st.write("Sube una imagen y el modelo te dirá si es un perro o un gato.")

archivo_subido = st.file_uploader("Sube una imagen...", type=["jpg", "png", "jpeg"])

if archivo_subido is not None:
    imagen = Image.open(archivo_subido)
    st.image(imagen, caption="Imagen subida", use_column_width=True)

    # Preprocesar la imagen
    imagen_procesada = preprocesar_imagen(imagen)

    # Hacer la predicción
    prediccion = modelo.predict(imagen_procesada)[0][0]

    # Mostrar el resultado
    if prediccion > 0.5:
        st.success("¡Es un 🐶 **PERRO**! 🐾")
    else:
        st.success("¡Es un 🐱 **GATO**! 🐾")

st.write("📌 Modelo basado en una CNN entrenada con TensorFlow/Keras.")
