import streamlit as st
from modules.nav import Navbar

def main():
    Navbar()

    if 'data' not in st.session_state:
        st.warning("Por favor, carga los datos primero.")
        return
    
    data = st.session_state.data
    results = st.selectbox("Selecciona el tipo de visualización", 
                                    options=["Resultados de RNN", "Resultados de AutoML",
                                            "Resultados de ARIMA", "Resultados de XGBoost"])
    if results == "Resultados de RNN":
        # escoger que provincia predecir
        # runear la rnn con los datos normales y aumentados
        # mostrarlo con ploty 
        # mostrar una seccion con las estadisticas
        st.header("Resultados de RNN")
    if results == "Resultados de AutoML":
        st.header("Resultados de AutoML")
    if results == "Resultados de ARIMA":
        st.header("Resultados de ARIMA")
    if results == "Resultados de XGBoost":
        st.header("Resultados de XGBoost")

if __name__ == '__main__':
    main()