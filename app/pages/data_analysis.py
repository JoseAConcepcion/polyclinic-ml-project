import os
import streamlit as st
import pandas as pd
import plotly.express as px
from modules.nav import Navbar

st.set_page_config(page_title="Ánalisis de Datos")


@st.cache_data
def load_data(uploaded_file):
    data = pd.read_csv(uploaded_file)
    data['Date'] = pd.to_datetime(data['Date'], format='%Y_%m_%d')
    st.session_state.data = data
    return data

def main():
    Navbar()  

    st.header("Análisis de Datos")
    st.text("En esta sección se muestra un primer análisis básico de los datos de casos de enfermedades respiratorias agudas por provincia.")
    
    uploaded_file = st.file_uploader("Cargar archivo CSV", type=["csv"])
    
    if uploaded_file is not None:
        data = load_data(uploaded_file)

        visualization = st.selectbox("Selecciona el tipo de visualización", 
                                      options=["Mostrar DataFrame", "Estadísticas",
                                               "Evolución por mes", "Comparación por provincia"])

        # start_date = pd.to_datetime('2022-09-01')
        # end_date = pd.to_datetime('2023-12-01')
        # filtered_data = data[(data['Date'] >= start_date) & (data['Date'] <= end_date)]

        if visualization == "Evolución por mes":
            monthly_data = data.groupby(pd.Grouper(key='Date', freq='M')).sum()
            fig = px.line(monthly_data, x=monthly_data.index, y=monthly_data.columns, title="Evolución de casos por mes")
            st.plotly_chart(fig)

        elif visualization == "Comparación por provincia":
            total_by_province = data.select_dtypes(include='number').sum()
            fig = px.bar(total_by_province, 
                         x=total_by_province.index, 
                         y=total_by_province.values,
                         title="Total de casos por provincia")
            st.plotly_chart(fig)

        if visualization == "Estadísticas":
            if 'Date' in data.columns:
                data['Date'] = pd.to_datetime(data['Date'])

                start_date = data['Date'].min()
                end_date = data['Date'].max()
                all_dates = pd.date_range(start=start_date, end=end_date)

                present_dates = data['Date'].unique()

                # Get missing days
                missing_dates = all_dates[~all_dates.isin(present_dates)]

                missing_data_df = pd.DataFrame(missing_dates, columns=['Missing Dates'])
                missing_data_df['Year'] = missing_data_df['Missing Dates'].dt.year
                missing_data_df['Month'] = missing_data_df['Missing Dates'].dt.month
                missing_data_df['Day'] = missing_data_df['Missing Dates'].dt.day

                years = missing_data_df['Year'].unique()
                selected_year = st.selectbox("Selecciona un año:", years)

                # Filtering
                missing_monthly = missing_data_df[missing_data_df['Year'] == selected_year]
                missing_count = missing_monthly.groupby('Month').size().reset_index(name='Missing Count')

                fig = px.bar(missing_count, x='Month', y='Missing Count', 
                            title=f'Cantidad de Días Faltantes por Mes en {selected_year}',
                            labels={'Month': 'Mes', 'Missing Count': 'Cantidad de Días Faltantes'})
                st.plotly_chart(fig)
            else:
                st.warning("La columna 'Date' no se encuentra en el DataFrame.")

            st.write("Estadísticas descriptivas:")
            st.dataframe(data.describe())

        elif visualization == "Mostrar DataFrame":
            st.write("Datos originales:")
            st.dataframe(data)
    else:
        st.warning("Por favor, carga un archivo CSV para continuar.")

data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data/Anexos_7/total_incomes.csv'))
data = pd.read_csv(data_path)
data['Date'] = pd.to_datetime(data['Date'], format='%Y_%m_%d')

if __name__ == '__main__':
    main()