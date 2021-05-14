import streamlit as st
import plotly.express as px
import numpy as np
import pickle
import load_data
import time

# make a timer on the page
# with st.spinner(text='In progress'):
#     time.sleep(5)
#     st.success('Done')

# input images / video -- link to the file you want
# could upload file to the specific df
file = st.file_uploader('File uploader')

# have user pick a color
st.color_picker('Pick a color')

# name of the app we want to appear on the top of the screen
st.title("My Awesome Flower Predictor")
st.header("We predict Iris types")
st.subheader("No joke")

# get input from user about Iris and predict what type/species it is

# load data
df_iris = load_data.load_iris()

# make plots on the page
st.plotly_chart(px.scatter(df_iris, 'sepal_width', 'sepal_length'))

# make it an option to show only if the user wants to

# question prompt for checkbox -- save result as boolean
show_df = st.checkbox("Do you want to see the data?")

if show_df: # if True
    df_iris

# make interactive -- get user input so we can make a prediction
# must manually list the questions in the RIGHT ORDER as the df table

# sepal length
s_l = st.number_input('Input the Sepal Length')

# sepal width
s_w = st.number_input('Input the Sepal Width')

# petal length
p_l = st.number_input('Input the Petal Length')

# petal width
p_w = st.number_input('Input the Petal Width')

# we need a df or np array to feed into
user_values = np.array([s_l, s_w, p_l, p_w])

# load model -- make sure the model made was saved in the same env
# rb -- read binary
# wb -- write binary

with open('saved-iris-model-2.pkl', 'rb') as f:
    model = pickle.load(f)

# make prediction
# .predict, since our model is just a regular Logistic Regression

with st.echo():
    # st.echo() will output the actual code -- could have a checkbox for people to see the code if they want
    prediction = model.predict(user_values.reshape(1, -1))

    # need to have variables by itself for itto display in the web app
    prediction

# make our predictions look nicer for the user -- [0]
st.header(f'The model predicts: {prediction[0]}')

# easter egg
# st.balloons()

# make vertical columns (optional)
st.beta_container()

col1, col2, col3 = st.beta_columns(3)
# col1.subheader('Columnisation')
# col2.subheader('Columnisation')
# col3.subheader('Columnisation')

# can do the above OR context blocks like below
with col1:
    'I am printing things'
with col2:
    df_iris
with col3:
    st.subheader("cool stuff")