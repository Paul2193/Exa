
import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency, kruskal, ttest_ind
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from streamlit_folium import st_folium
import os
import gdown

st.set_page_config(layout="wide")
st.title('Análisis Estadístico de Crímenes en la ciudad Los Ángeles')
st.title('Autores: Paul Ulloa, Jose Valverde ')
@st.cache_data
def load_data():
    file_id = '1-skQNlFb1w4RIRQRG4-WH-txfQA3RtlP'
    url = f'https://drive.google.com/uc?id={file_id}'
    output = 'Crime_Data_from_2020_to_Present.csv'
    if not os.path.exists(output):
        gdown.download(url, output, quiet=False)
    df = pd.read_csv(output, nrows=90000)

    column_translation = {
        'DR_NO': 'Número Reporte',
        'Date Rptd': 'Fecha Reporte',
        'DATE OCC': 'Fecha Ocurrencia',
        'TIME OCC': 'Hora Ocurrencia',
        'AREA': 'Código Área',
        'AREA NAME': 'Área',
        'Rpt Dist No': 'Distrito Reporte',
        'Part 1-2': 'Clasificación Crimen',
        'Crm Cd': 'Código Crimen',
        'Crm Cd Desc': 'Descripción Crimen',
        'Mocodes': 'Modus Operandi',
        'Vict Age': 'Edad Víctima',
        'Vict Sex': 'Sexo Víctima',
        'Vict Descent': 'Descendencia Víctima',
        'Premis Cd': 'Código Lugar',
        'Premis Desc': 'Descripción Lugar',
        'Weapon Used Cd': 'Código Arma',
        'Weapon Desc': 'Descripción Arma',
        'Status': 'Código Estado',
        'Status Desc': 'Descripción Estado',
        'Crm Cd 1': 'Código Crimen 1',
        'Crm Cd 2': 'Código Crimen 2',
        'Crm Cd 3': 'Código Crimen 3',
        'Crm Cd 4': 'Código Crimen 4',
        'LOCATION': 'Dirección',
        'Cross Street': 'Calle Cercana',
        'LAT': 'Latitud',
        'LON': 'Longitud'
    }

    df.rename(columns=column_translation, inplace=True)
    df = df.dropna(subset=['Edad Víctima', 'Sexo Víctima', 'Latitud', 'Longitud'])
    df = df[df['Sexo Víctima'].isin(['M', 'F'])]
    df['Edad Víctima'] = pd.to_numeric(df['Edad Víctima'], errors='coerce')
    df_completo = df.dropna(subset=['Código Arma', 'Descripción Arma', 'Código Estado', 'Descripción Estado'])
    df_valid_age = df[df['Edad Víctima'] > 0].copy()
    return df, df_completo, df_valid_age

df, df_completo, df_valid_age = load_data()
df_sample = df.sample(n=5000, random_state=42)

st.header('Exploración Inicial del Dataset')
st.dataframe(df.head(100))

st.subheader(' Top 10 Tipos de Crimen')
st.bar_chart(df['Descripción Crimen'].value_counts().head(10))

st.subheader(' Top 10 Áreas con Más Crímenes')
fig_area, ax = plt.subplots()
df['Área'].value_counts().head(10).plot(kind='barh', ax=ax, color='skyblue')
ax.set_title('Áreas con Mayor Cantidad de Crímenes')
st.pyplot(fig_area)

st.subheader(' Distribución de Edad de las Víctimas')
fig_age, ax = plt.subplots()
sns.histplot(df_valid_age['Edad Víctima'], bins=30, color='purple', ax=ax)
ax.set_title('Distribución de Edad')
st.pyplot(fig_age)

st.header(' Mapas de Crímenes Agrupados por Sexo')
with st.container():
    st.subheader(' Crímenes contra Hombres')
    m_male = folium.Map(location=[34.05, -118.25], zoom_start=10)
    male_grouped = df_sample[df_sample['Sexo Víctima'] == 'M'].groupby('Área').agg({
        'Latitud': 'mean', 'Longitud': 'mean', 'Número Reporte': 'count'
    }).reset_index()
    for _, row in male_grouped.iterrows():
        folium.CircleMarker(
            location=[row['Latitud'], row['Longitud']],
            radius=5 + (row['Número Reporte'] / 50),
            color='blue', fill=True, fill_opacity=0.5,
            popup=f"Área: {row['Área']}<br>Crímenes: {row['Número Reporte']}"
        ).add_to(m_male)
    st_folium(m_male, width=1200, height=500)

with st.container():
    st.subheader('Crímenes contra Mujeres')
    m_female = folium.Map(location=[34.05, -118.25], zoom_start=10)
    female_grouped = df_sample[df_sample['Sexo Víctima'] == 'F'].groupby('Área').agg({
        'Latitud': 'mean', 'Longitud': 'mean', 'Número Reporte': 'count'
    }).reset_index()
    for _, row in female_grouped.iterrows():
        folium.CircleMarker(
            location=[row['Latitud'], row['Longitud']],
            radius=5 + (row['Número Reporte'] / 50),
            color='red', fill=True, fill_opacity=0.5,
            popup=f"Área: {row['Área']}<br>Crímenes: {row['Número Reporte']}"
        ).add_to(m_female)
    st_folium(m_female, width=1200, height=500)

st.header('Análisis Estadístico')

# Análisis 1: Chi-cuadrado
st.subheader(" Análisis 1: Asociación entre Sexo y Tipo de Crimen")
contingencia = pd.crosstab(df['Descripción Crimen'], df['Sexo Víctima'])
chi2, p_chi2, _, _ = chi2_contingency(contingencia)
st.write(f"**Chi-cuadrado = {chi2:.2f}**, **p-valor = {p_chi2:.6e}**")
pivoted = df.groupby(['Descripción Crimen', 'Sexo Víctima']).size().unstack().fillna(0)
fig1, ax1 = plt.subplots(figsize=(10, 6))
pivoted.head(10).plot(kind='bar', stacked=True, ax=ax1)
ax1.set_title('Crímenes por Tipo y Sexo')
st.pyplot(fig1)
if p_chi2 < 0.05:
    st.write(" Conclusión: La prueba Chi-cuadrado indica una relación significativa entre el tipo de crimen y el sexo de la víctima. Por ejemplo, delitos como violencia doméstica se presentan más en mujeres, mientras que agresiones físicas son más comunes en hombres.")
else:
    st.write(" Conclusión: No se halló una asociación estadísticamente significativa entre el tipo de crimen y el sexo de la víctima.")

# Análisis 2: Kruskal-Wallis
st.subheader("Análisis 2: Edad según Tipo de Crimen")
top_crimes = df_valid_age['Descripción Crimen'].value_counts().head(3).index
samples = [df_valid_age[df_valid_age['Descripción Crimen'] == crime]['Edad Víctima'] for crime in top_crimes]
stat, p_kruskal = kruskal(*samples)
st.write(f"**Kruskal-Wallis H = {stat:.2f}**, **p-valor = {p_kruskal:.6e}**")
fig2, ax2 = plt.subplots()
sns.boxplot(data=df_valid_age[df_valid_age['Descripción Crimen'].isin(top_crimes)],
            x='Descripción Crimen', y='Edad Víctima', ax=ax2)
ax2.set_title("Distribución de Edad según Tipo de Crimen")
st.pyplot(fig2)
if p_kruskal < 0.05:
    st.write("Conclusión: Las edades de las víctimas varían significativamente según el tipo de crimen. Por ejemplo, los delitos de robo suelen afectar a personas jóvenes, mientras que fraudes y estafas involucran con mayor frecuencia a personas mayores.")
else:
    st.write("Conclusión: No se detectaron diferencias significativas de edad entre los tipos de crimen analizados.")

# Análisis 3: T-Test
st.subheader("Análisis 3: Edad Promedio por Sexo")
male_age = df_valid_age[df_valid_age['Sexo Víctima'] == 'M']['Edad Víctima']
female_age = df_valid_age[df_valid_age['Sexo Víctima'] == 'F']['Edad Víctima']
t_stat, p_ttest = ttest_ind(male_age, female_age, equal_var=False)
st.write(f"**t = {t_stat:.2f}**, **p-valor = {p_ttest:.6e}**")
fig3, ax3 = plt.subplots()
sns.boxplot(data=df_valid_age, x='Sexo Víctima', y='Edad Víctima', ax=ax3)
ax3.set_title("Edad de las Víctimas según Sexo")
st.pyplot(fig3)
if p_ttest < 0.05:
    st.write("Conclusión: Hay una diferencia significativa en la edad promedio de víctimas según su sexo. Las mujeres víctimas tienden a ser más jóvenes en comparación con los hombres víctimas, lo cual podría reflejar distintos perfiles de riesgo.")
else:
    st.write("Conclusión: No se encontró diferencia estadísticamente significativa en la edad promedio entre hombres y mujeres víctimas.")

# Análisis 4: Correlación numérica
st.subheader("Análisis 4: Correlación entre Edad, Latitud y Longitud")
numeric_vars = df_valid_age[['Edad Víctima', 'Latitud', 'Longitud']].dropna()
cor_matrix = numeric_vars.corr()
st.dataframe(cor_matrix)
fig4, ax4 = plt.subplots()
sns.heatmap(cor_matrix, annot=True, cmap='coolwarm', fmt=".2f", ax=ax4, vmin=-1, vmax=1)
ax4.set_title('Mapa de Calor - Correlación entre Variables Numéricas')
st.pyplot(fig4)
st.write("Conclusión: Las correlaciones entre edad, latitud y longitud son muy bajas (cercanas a cero), lo que indica que no existe una relación lineal fuerte entre la edad de las víctimas y la ubicación geográfica del crimen.")

# Conclusión global
st.header('Conclusiones Generales')
st.markdown("""
### **Pregunta 1:** ¿Existe una relación entre el tipo de crimen y el sexo de la víctima?

**Respuesta:**  
Sí, existe una relación significativa entre el tipo de crimen y el sexo de la víctima. La prueba de **Chi-cuadrado** mostró un **p-valor < 0.05**, lo que indica dependencia entre ambas variables.Esto se respalda visualmente con el gráfico de barras apiladas, en el cual se observa que ciertos delitos son más frecuentes en hombres (como Battery - Simple Assault o Aggravated Assault), mientras que otros presentan una mayor proporción de mujeres víctimas, aunque en menor número general.
Este patrón sugiere que el sexo de la víctima influye en la probabilidad de ser afectado por determinados tipos de crímenes."

---

### **Pregunta 2:** ¿La edad de las víctimas varía según el tipo de crimen?

**Respuesta:**  
Sí, la edad de las víctimas varía significativamente según el tipo de crimen. La prueba de Kruskal-Wallis arrojó un valor de H = 134.12 con un p-valor ≈ 7.52e-30, lo que confirma diferencias estadísticamente significativas entre los grupos analizados. En el boxplot se observa que delitos como “Battery - Simple Assault” presentan víctimas de mayor edad en promedio, mientras que crímenes como “Burglary from Vehicle” y “Assault with Deadly Weapon” tienden a involucrar víctimas más jóvenes. Además, se aprecia una mayor dispersión de edades en algunos delitos

---

### **Pregunta 3:** ¿Existe una diferencia significativa en la edad promedio entre víctimas hombres y mujeres?

**Respuesta:**  
Sí, existe una diferencia significativa en la edad promedio entre víctimas hombres y mujeres. La prueba t-test para muestras independientes arrojó un valor de t = 16.29 con un p-valor ≈ 1.63e-59, indicando que la diferencia observada no es producto del azar. En el boxplot se aprecia que los hombres víctimas presentan una mediana de edad más alta en comparación con las mujeres víctimas, cuya distribución está más concentrada en edades jóvenes. 

---

### **Pregunta 4:** ¿Existe una correlación entre la edad de la víctima y la ubicación del crimen?

**Respuesta:**  
No se encontró una correlación significativa entre la edad de la víctima y la ubicación geográfica del crimen. La matriz de correlación muestra valores cercanos a cero entre Edad Víctima y las variables Latitud y Longitud, específicamente -0.01 y 0.01, lo que indica ausencia de una relación lineal entre estas variables. Aunque Latitud y Longitud presentan una correlación negativa perfecta entre sí (-1.00), esto responde a una característica propia del sistema de coordenadas y no a una relación con la edad. En conclusion, no hay evidencia estadística que sugiera que la edad de la víctima varíe según la zona geográfica en la que ocurre el crimen.
""")
