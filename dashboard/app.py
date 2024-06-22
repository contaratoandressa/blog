import streamlit as st
import pandas as pd
import numpy as np
import time
from mitosheet.streamlit.v1 import spreadsheet
from mitosheet.streamlit.v1.spreadsheet import _get_mito_backend


# Add title
st.write("Dashboard - Dados de Redes Sociais")
st.set_page_config(layout="wide")

# Dont clean the cache 
@st.cache_data
def get_tesla_data():
    df = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/tesla-stock-price.csv')
    df = df.drop(0)
    df['volume'] = df['volume'].astype(float)
    return df

tesla_data = get_tesla_data()


# Add a selectbox to the sidebar:
add_selectbox = st.sidebar.selectbox(
    'Como gostaria de entrar em contato?',
    ('Email', 'Home phone', 'Mobile phone')
)


"""
# Quantidade de pessoas que usam rede social por tipo
No período de tempo de 2022 a 2023 na cidade do Rio de Janeiro:
"""

st.table(pd.DataFrame({
  'Rede Social': ["Facebook", "Instagram", "Tik Tok", "Whatsapp"],
  'Quantidade de Pessoas': [10, 20, 30, 40]
})
)

"""
# Quantidade de pessoas que usam rede social por tipo
No período de tempo de 2022 a 2023 na cidade do Rio de Janeiro: \\
Dados para download e aplicando estilo na tabela
"""

dataframe = pd.DataFrame({
  'Rede Social': ["Facebook", "Instagram", "Tik Tok", "Whatsapp"],
  'Quantidade de Pessoas': [10, 20, 30, 40]
})
st.dataframe(dataframe.style.highlight_max(axis=0)) 

"""
# Série temporal do uso de redes sociais por tipo de rede social
No período de tempo de 2022 a 2023 na cidade do Rio de Janeiro: \\
Dados para download e aplicando estilo na tabela
"""

df = pd.DataFrame()
df["Facebook"] = [1,2,3,4,5]
df["Instagram"] = [10,20,30,40,50]
df["Tik Tok"] = [1,10,40,25,2]
df["Whatsapp"] = [1,1,1,1,1]
st.line_chart(df)

x = st.slider('x')  # 👈 this is a widget
st.write(x, 'squared is', x * x)

"""
# Mapa do uso de redes sociais no estado do Rio de Janeiro
No período de tempo de 2022 a 2023 na cidade do Rio de Janeiro: \\
Dados para download e aplicando estilo na tabela
"""

map_data = pd.DataFrame(
    np.concatenate((np.random.randn(1000, 2) / [50, 50] + [-22.91, -43.61], np.random.randn(1000, 2) / [50, 50] + [35, 139]), axis=0),
    columns=['lat', 'lon'])

st.map(map_data)

y = st.slider('y')  # 👈 this is a widget
st.write(y, 'squared is', y * y^2)

st.text_input("Nome da Cidade", key="name") # access key
st.session_state.name

if st.checkbox('Mostrar os dados do Mapa'):
    map_data

df = pd.DataFrame({
    'first column': [1, 2, 3, 4],
    'second column': [10, 20, 30, 40]
    })

if st.checkbox('Mostrar os dados da Tabela'):
    df

if "counter" not in st.session_state:
    st.session_state.counter = 0

st.session_state.counter += 1

st.header(f"This page has run {st.session_state.counter} times.")
st.button("Run it again")

if "df" not in st.session_state:
    st.session_state.df = pd.DataFrame(np.random.randn(20, 2), columns=["x", "y"])

st.header("Choose a datapoint color")
color = st.color_picker("Color", "#FF0000")
st.divider()
st.scatter_chart(st.session_state.df, x="x", y="y", color=color)
