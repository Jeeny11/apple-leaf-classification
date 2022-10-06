#Import Libraries
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras 
import numpy as np
from tensorflow.keras.preprocessing import image
from PIL import Image

import streamlit as st

logo='apple.jpg'
DIR_MODEL='' #'/content/drive/MyDrive/Master/TFM/Data/Models/'
model_file ='InceptionV3.h5'
#model_file ='DenseNet_model.h5'



st.title("Classification of apple leaf")

##load model once and save it in the cache to avoid continues loading of model
#load model, set cache to prevent reloading
@st.cache(allow_output_mutation=True)
def load_model(model_file):
    model=tf.keras.models.load_model(model_file)
    return model

def model_changed():  
    #model_file=st.session_state.model
    sel_model_file = st.session_state.sel_model+".h5"
    st.info(f"aktueller status: {st.session_state.sel_model}. Model das geladen werden soll {sel_model_file}")
    new_model=load_model(sel_model_file)
    st.info(f"new model: {st.session_state.sel_model}, model: {model_file} ")
    predict_Image(new_model,file)


with st.spinner("Loading Model...."):
    model=load_model(model_file)




#Display an Image
st.sidebar.image(logo, width=200)



file = st.sidebar.file_uploader("Please upload image file of apple leaf", type=["jpg", "png"])
##avoid warning that always appears at file upload
st.set_option('deprecation.showfileUploaderEncoding', False)


if 'sel_model' not in st.session_state:  st.session_state.sel_model = 'InceptionV3'
   
option=st.selectbox("Select data model for prediction: ", ['InceptionV3', 'DenseNet','CNN_model'], 
                                  key='sel_model',
                                  on_change=model_changed)

st.write("**Actual model** :" ,option)
#size image was trained on
IMG_SIZE=224
LABELS=['healthy', 'multiple_diseases', 'rust', 'scab']


def predict_Image(modelFile, imageFile):
    #load image
    imageLoaded = image.load_img(imageFile,target_size=(IMG_SIZE,IMG_SIZE))
    #convert image to array
    imageArray = image.img_to_array(imageLoaded)
    #scale imageArray
    imageArray = imageArray/255
    #we need further dimension to scope with 4D shape which expects trained model
    imageArray = np.expand_dims(imageArray,axis=0)
   
    predictions = modelFile.predict(imageArray)
    predicted_class = LABELS[np.argmax(predictions[0])]
    accuracy = round(100 * (np.max(predictions[0])), 2)
    image_result = plt.imshow(imageLoaded)
    #plt.title(predicted_class)
 
    info="Model predicts for this image  **{}** with an accurray of **{:.2f}%**".format(predicted_class, accuracy)
    st.image(imageLoaded, use_column_width=False)
    st.write(info)
    
    
    ##neuerungen
    #st.image(imageLoaded, use_column_width=False)
    #st.write(predicted_class)
    #st.write(accuracy)
    #st.write(info)
    return predicted_class, accuracy,imageLoaded

if file is None:
    st.text("You need to upload image (format jpg or png)")
else:
    predicted_class, accuracy,imageLoaded=predict_Image(model,file)
   
   
