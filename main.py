import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

model = tf.keras.models.load_model('fashion_mnist_model.keras')

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot']

def preprocess_image(image):
    image = image.convert('L')
    image = image.resize((28, 28))
    image = np.array(image) / 255.0
    image = image.reshape(1, 28, 28, 1)
    return image

def predict(image):
    image = preprocess_image(image)
    predictions = model.predict(image)
    predicted_class = np.argmax(predictions)
    return class_names[predicted_class], predictions[0][predicted_class]

st.title('Fashion MNIST Image Classifier')
st.write('Cargue una imagen de ropa para predecir su clase.')

uploaded_file = st.file_uploader("Elija una imagen", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Imagen cargada', use_column_width=True)
    st.write("Clasificando...")

    predicted_class, confidence = predict(image)

    st.write(f"Predicción: {predicted_class}")
    st.write(f"Confianza: {confidence*100:.2f}%")

    st.bar_chart(dict(zip(class_names, model.predict(preprocess_image(image)).flatten())))
