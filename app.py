import streamlit as st
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
import sklearn
import random
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix

# Configuracion Inicial
st.set_page_config(page_title= '🍎Sistema de Recomendación de Produtos', layout = 'wide')

st.markdown("<h1 style='text-align: center;'>🍾 Recomendación de Productos Tienda Gourmet</h1>",
            unsafe_allow_html=True)

st.markdown("---")

st.subheader("Datos Ciencia Sistema de Recomendación")

#Imagen

image = Image.open('imagenes/logotipofavicon.png')
st.image(image, caption='Datos Ciencia',  width=100)


# Importacion / Manipulacion de los Datos
def importa_dado():
    datos= pd.read_csv('data/datos_recomendacion_limpios.csv')

    return datos

datos = importa_dado()

df = datos
#st.dataframe(datos.style.highlight_max(axis=0))

# Configurar la primera columna con el DataFrame
col1, col2 = st.columns(2)
col1.subheader('Ventas')
df1 = df.loc[:,["product_id", "name", "rating_1"]]
col1.dataframe(df1)

# Configurar la segunda columna con la foto
col2.subheader('Tienda Gourmet Shopmyseke')
image = Image.open('imagenes/shopmyseke.png')  # Reemplaza con la URL de tu imagen
col2.image(image, use_column_width=True)

#Construir la Matrix 
#Creamos un pivot table train:
pivot_table=df.pivot_table(index='product_id',columns='id_customer',values='rating_1').fillna(0)

features_matrix=csr_matrix(pivot_table.values)

# Construir Modelo KNN:

model = NearestNeighbors(metric='cosine', algorithm='brute')
model.fit(features_matrix) 

# Función de recomendación
def recommend_products(product_id, num_recommendations=8):
    #query_ind=np.random.choice(pivot_table.shape[1])
    query_ind = random.randint(0, pivot_table.shape[0]-1)
    distances,indices=model.kneighbors(pivot_table.iloc[query_ind,:].values.reshape(1,-1),n_neighbors=8)
    prediccion = indices.flatten()
    product_names = df.loc[prediccion, 'name']
    return prediccion, product_names

# Interfaz de usuario con Streamlit
def main():
    st.title('Sistema de Recomendación de Productos')
    
    # Mostrar formulario para ingresar el ID del producto
    product_id = st.text_input('Escriba el ID del producto:')
    
    if st.button('Recomendar'):
        recommendations, product_names = recommend_products(product_id)
        
        if len(recommendations) > 0:
            st.subheader('Productos recomendados:')
            recommendations_df= pd.DataFrame({'Nombre': product_names})
            st.table(recommendations_df)
        else:
            st.info('No se encontraron recomendaciones para el producto seleccionado.')

if __name__ == '__main__':
    main()

# Agregar un espacio
st.markdown("---")

# Configurar la primera columna con el DataFrame
#col1, col2 = st.columns(2)
#col1.subheader('Analisis de Datos')
#image = Image.open('imagenes/analytics.png')
#width = '150px' 
#col1.image(image, use_column_width=width)


# Crear un espacio vacío
#spacer = st.empty()


# Productos más vendidos

freq_vendas=df.groupby('name').count().sort_values('product_id', ascending=False).reset_index()[['product_id', 'name']]
top_10=freq_vendas.head(10)
grafico1= px.bar(top_10, x='name', y='product_id', title='Productos más Vendidos',labels={"name":"Produtos", "product_id":"N Ventas"},
                color_discrete_sequence=px.colors.sequential.Aggrnyl,)

grafico1 = px.bar(top_10, x='name', y= 'product_id', title='Productos más Vendidos')

grafico1






# Remove Estilo Streamlit

remove_st_estilo = """
    <style>
        #MainMenu {visibility: hidden;}
        #footer {visibility: hidden;}
        #header {visibility: hidden;}    

    </style>
"""
st.markdown(remove_st_estilo, unsafe_allow_html=True)