import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Cargar el modelo entrenado
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("modelCCN3.h5")

model = load_model()

# Preprocesamiento de imagen
def preprocess_image(image):
    image = image.convert("L")  # Convertir a escala de grises si el modelo lo requiere
    image = image.resize((100, 100))  # Redimensionar a 100x100 píxeles
    image = np.array(image) / 255.0  # Normalización
    image = np.expand_dims(image, axis=-1)  # Añadir canal de profundidad si es necesario
    image = np.expand_dims(image, axis=0)  # Añadir dimensión batch
    return image

# Interfaz en Streamlit
st.title("Clasificador de Perros y Gatos 🐶🐱")
st.write("Sube una imagen y el modelo te dirá si es un perro o un gato.")

uploaded_file = st.file_uploader("Sube una imagen...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Imagen subida", use_column_width=True)

    # Preprocesar la imagen
    processed_image = preprocess_image(image)

    # Realizar predicción
    prediction = model.predict(processed_image)[0][0]

    # Mostrar resultado
    if prediction > 0.5:
        st.success("¡Es un 🐶 **PERRO**!")
    else:
        st.success("¡Es un 🐱 **GATO**!")

st.write("📌 Modelo basado en una red neuronal convolucional (CNN) entrenada con TensorFlow/Keras.")
