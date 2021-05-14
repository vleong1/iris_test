import streamlit as st
import plotly.express as px

# caching / saving + creating a decorator to enhance performance / faster
@st.cache # decorator
def load_iris():
    return px.data.iris()