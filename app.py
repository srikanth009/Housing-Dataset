import streamlit as st
import cv2 
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.models import model_from_json

k=193
def load_images(file,m,n):
    img = cv2.imread(file)
    img= cv2.resize(img,(m,n), interpolation = cv2.INTER_AREA)
    img=np.array(255*(img / 255) ** 2, dtype = 'uint8')
    return img

k=193

radio = st.sidebar.radio(label="What are you", options=["Select", "Employee", "Customer"])
if radio=='Employee':
    st.title("Welcome to your Dream House project!!")
    email=st.text_input('Please enter your mail id (only for employees)')
    if email:
        if email=='eranki007':
          l = st.sidebar.selectbox("model select only for employees",(1,2,3,4,5,6))
          st.title('we will test with {} models'.format(l))
          k=(l*32)+1
        else:
            st.title('should i call the cops? are you a fake employee?')

if radio=='Customer':
    name=st.text_input('Please enter your name')
    number=st.text_input('please enter your mobile number')
    json_file = open('model_32*32.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights('model_32*32.h5')
    filename = st.file_uploader("Choose a file from your device")
    json_file1 = open('model_64*64.json', 'r')
    loaded_model_json1 = json_file1.read()
    json_file1.close()
    loaded_model1 = model_from_json(loaded_model_json1)
    loaded_model1.load_weights('model_64*64.h5')
    json_file2 = open('model_96*96.json', 'r')
    loaded_model_json2 = json_file2.read()
    json_file2.close()
    loaded_model2 = model_from_json(loaded_model_json2)
    loaded_model2.load_weights('model_96*96.h5')
            
    if filename and name and number:
        st.image(filename,300,300)
        json_file3 = open('model_128*128.json', 'r')
        loaded_model_json3 = json_file3.read()
        json_file3.close()
        loaded_model3 = model_from_json(loaded_model_json3)
        loaded_model3.load_weights('model_128*128.h5')
        json_file4 = open('model_160*160.json', 'r')
        loaded_model_json4 = json_file4.read()
        json_file4.close()
        loaded_model4 = model_from_json(loaded_model_json4)
        loaded_model4.load_weights('model_160*160.h5')
        json_file5 = open('model_192*192.json', 'r')
        loaded_model_json5 = json_file5.read()
        json_file5.close()
        loaded_model5 = model_from_json(loaded_model_json5)
        loaded_model5.load_weights('model_192*192.h5')
        model=[loaded_model,loaded_model1,loaded_model2,loaded_model3,loaded_model4,loaded_model5]
        p=[]
        for i in range(32,k,32):
            img=load_images(filename.name, i, i)
            p.append(model[(i//32)-1].predict(img.reshape(1,i,i,3))[0][0])
        c=0
        for i in range(len(p)):
            if p[i]>=0.5:
                c+=1
        if c/(k//32)>=0.5:
            st.title('Image Accepted, and added to the records of {},{}'.format(number,name))
            
        else:
            st.title('Sorry!! Please upload an appropriate image with good lighting condition')




