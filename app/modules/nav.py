import streamlit as st


def Navbar():
    with st.sidebar:
        st.page_link('app.py', label='Inicio')
        st.page_link('pages/data_analysis.py', label='Análisis de datos', icon='📊')
        st.page_link('pages/predictions.py', label='Predicciones', icon='🔮')
        st.page_link('pages/about.py', label='Acerca de', icon='ℹ️')