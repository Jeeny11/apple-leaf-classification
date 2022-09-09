
import streamlit as st
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras 
import numpy as np
from tensorflow.keras.preprocessing import image
from PIL import Image

DIR_MODEL=''
model_file =DIR_MODEL+'InceptionV3_model.h5'
#model_file ='InceptionV3_model.h5'

@st.cache(allow_output_mutation=True)
def load_model():
  model=tf.keras.models.load_model(model_file)
  return model
with st.spinner('Model is being loaded..'):
  model=load_model()

#title of Website
st.write("""
         # Classification of apple leaf
         """
         )
##some description
st.write(" we use model ", model_file)

file = st.file_uploader("Please upload image file of apple leaf", type=["jpg", "png"])

st.set_option('deprecation.showfileUploaderEncoding', False)




IMG_SIZE=224
LABELS=['healthy', 'multiple_diseases', 'rust', 'scab']


def predict_Image(modelFile, imageFile):
  
    imageLoaded = image.load_img(imageFile,target_size=(IMG_SIZE,IMG_SIZE))
    imageArray = image.img_to_array(imageLoaded)
    imageArray = imageArray/255
    imageArray = np.expand_dims(imageArray,axis=0)
   
    predictions = modelFile.predict(imageArray)
    predicted_class = LABELS[np.argmax(predictions[0])]
    accuracy = round(100 * (np.max(predictions[0])), 2)
    image_result = plt.imshow(imageLoaded)
    #plt.title(predicted_class)
 
    info="Model predicts for this image  << {} >> with an accurray of {:.2f}%".format(predicted_class, accuracy)
  
    ##neuerungen
    st.image(imageLoaded, use_column_width=False)
    st.write(predicted_class)
    st.write(accuracy)
    st.write(info)
    return predicted_class, accuracy


if file is None:
    st.text("You need to upload image (format jpg or png)")
else:
    predicted_class, accuracy=predict_Image(model,file)
   
