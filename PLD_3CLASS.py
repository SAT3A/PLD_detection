import tensorflow as tf
import streamlit as st
import cv2
from PIL import Image, ImageOps
import numpy as np
from tensorflow.keras.applications.inception_v3 import preprocess_input, decode_predictions
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np

model1 = tf.keras.models.load_model('new_3CLASS_INCV3.hdf5')
model2 = tf.keras.models.load_model('new_3CLASS_ResNet50.hdf5')
model3 = tf.keras.models.load_model('new_3CLASS_VGG19.hdf5')

MODELS = {
    "INCEPTIONV3" : model1,
    "ResNet50" : model2,
    "VGG19" : model3,
}


st.set_page_config(
    page_title="PLD DETECTION",
    page_icon="3196633.ico",
    layout="centered",
    initial_sidebar_state="expanded",
)

st.title("PLD DETECTION")


# define a dictionary that maps model names to their classes
# inside Keras
model_list = ['INCEPTIONV3', 'ResNet50', 'VGG19']
#model_list = [MODELS["INCEPTIONV3"], MODELS["ResNet50"], MODELS["VGG19"]]
sb = st.selectbox("Select the Model", model_list)

if (sb == 'INCEPTIONV3'):
    model = MODELS["INCEPTIONV3"]
elif (sb == 'ResNet50'):
    model = MODELS["ResNet50"]
elif (sb == 'VGG19'):
    model = MODELS["VGG19"]

st.write("Please upload an image file")
file = st.file_uploader("type file must be jpg or png", type=["jpg", "png"])


def import_and_predict(image_data, model_list):

    size = (150, 150)
    image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
    image = np.asarray(image)
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_resize = (cv2.resize(img, dsize=(200, 200),
                             interpolation=cv2.INTER_CUBIC))/255.

    img_reshape = img_resize[np.newaxis, ...]

    prediction = model_list.predict(img_reshape)
    confidence = prediction*100

    return confidence


if file is None:
    st.text("There's No image Uploaded")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    prediction = import_and_predict(image, model)

    if np.argmax(prediction) == 0:
        st.write("Healthy")
    elif np.argmax(prediction) == 1:
        st.write("Early Blight")
    elif np.argmax(prediction) == 2:
        st.write("Late Blight")

    st.text("PREDICTION = [0: Healthy, 1: Early_Blight, 2:Late_Blight]")
    st.write(prediction)
