import tensorflow_addons as tfa
from tensorflow_addons.metrics import HammingLoss

import tensorflow as tf

from PIL import Image, ImageOps
import pandas as pd
import numpy as np
from cv2 import cv2

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.preprocessing import image
from keras.metrics import binary_accuracy
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from keras.layers import BatchNormalization

import streamlit as st

model = keras.models.load_model('sofas.h5', custom_objects={'HammingLoss': HammingLoss} )

st.title('Welcome to the sofa classifier (from the makers of the table classifier)!')

st.markdown("**Purpose of this model**: It takes an image of a sofa and predicts whether its width/depth/height dimensions are bigger than those of a 'standard' 3-seat sofa (Width: 90ins, Depth: 38ins, Height: 34ins. Those dimensions were taken from this article on allform.com: [link](https://allform.com/pages/standard-sofa-size#:~:text=Dimensions%20Explained,vary%20from%20brand%20to%20brand.))")
st.write("Each prediction for width, depth and height will be a probability between 0% (i.e. the dimension is a lot smaller than a standard sofa) and 100% (i.e. it's much bigger than a standard sofa)")
st.write("As you'll see when you upload an image, we'll add those 3 scores together to create a weighting to decide whether to charge more for transporting the item")

with st.expander("Click here for more details about how this model was built"):
        st.write("""This is a Multilabel Classification model using a Convolutional Neural Network (CNN) to convert images into grids of numbers, which it then scans to discover patterns.""") 
        st.write("""Each image also comes with 3x labels, which correspond to the 3x dimensions: width, depth and height. They were scored with a 1 if the dimension was bigger than a standard sofa and 0 if was smaller. The model uses that information along with the patterns it learned to calculate the probability of whether a new image is bigger/smaller than a standard 3-seater.""")
        st.write("""Over 1,000 images were used to train and test the model. That's A LOT of photo shoots!""")
            
@st.cache

def import_and_predict(image_data, model):
    
        size = (256,256)
        image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
        image = np.asarray(image)
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_resize = (cv2.resize(img, dsize=(256, 256), interpolation=cv2.INTER_CUBIC))/255.
        
        img_reshape = img_resize[np.newaxis,...]    
        
        prediction = model.predict(img_reshape)

        return prediction

file = st.file_uploader("Please upload your image...", type=["png","jpg","jpeg"])

if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image, width=200)
    prediction = import_and_predict(image, model)
    st.write("Here's the model's predictions on whether your image is wider/deeper/taller than a 'standard' sofa:")
  
    df = pd.DataFrame(prediction, columns = ['% probability of being Wider','% probability of being Deeper','% probability of being Taller'])
    percent = df*100
    
    st.write(percent)

    sum = percent.sum(axis = "columns")
    
    st.write("Now let's add those probabilities together to create a threshold for whether we should charge extra for moving it. **The following scoring range is based on results from testing the model:**")
    st.write("If the score is between 0-50, the sofa is probably smaller than a standard 3-seater;")
    st.write("If the score is between 50-90, it's probably the same size;")
    st.write("If the score is 90+, it's probably bigger than a standard sofa - so we need to charge extra.")
    st.write("The overall score for this sofa image is...drumroll please...")

    st.write(sum)
    
    if sum[0]>90:
        st.write("""#### Analysis: This looks bigger than a standard sofa, so we should charge extra to move it""")
    else:
        st.write("""#### Analysis: This doesn't look bigger than a standard sofa, so we don't need to charge extra to move it""")

    
