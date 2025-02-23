import os
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from modules.nav import Navbar

st.set_page_config(page_title="Forecast", layout="wide")

data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/Anexos_7/total_incomes.csv'))
data = pd.read_csv(data_path)
data['Date'] = pd.to_datetime(data['Date'], format='%Y_%m_%d')

def Inicio():
    Navbar()
    st.title("Pronóstico de Casos de Enfermedades Respiratorias Agudas")
    st.header("Objetivo del proyecto")
    st.expander("_Hello_ there")
    
    

if __name__ == '__main__':
    Inicio()
