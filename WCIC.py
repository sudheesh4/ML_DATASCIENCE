API=""
import pathlib
import textwrap
import os

import google.generativeai as genai
import streamlit as st 
from streamlit_back_camera_input import back_camera_input
import PIL.Image
genai.configure(api_key=API)


st.set_page_config(layout='wide')

@st.cache_resource
def getmodel():
    return genai.GenerativeModel("gemini-pro")

@st.cache_resource
def getvision():
    return genai.GenerativeModel("gemini-pro-vision")

def queryimage(mdl,prompt,img):
    response=mdl.generate_content([prompt,img])
    response.resolve()
    return (response.text)
def querytext(mdl,prompt):
    #print(prompt)
    response=mdl.generate_content(prompt)
    return (response.text)  

model=getmodel()
vision=getvision()

st.header("What Can I cook?")

image = back_camera_input()#st.camera_input("What Can I cook?")
if image:
    prompt="Describe ingredients you can see."
    #st.image(image)
    
    txt=''
    open("test.jpg","wb").write(image.getbuffer())
    img=PIL.Image.open("test.jpg")
    st.image(img)
    st.write("ASKING GEMINI...")
    txt=queryimage(vision,prompt,img)
    out=querytext(model,f"Using some of the following ingredients, suggest the multiple recipes to cook. Ingredients:{txt}")
    print(txt)
    st.write(txt)
    st.write(out)