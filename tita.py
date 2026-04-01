import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Анализ пассажиров Титаника", layout="wide")
st.title("Дашборд по ТИТАНИКУ")

    
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('titanic.csv')
    except FileNotFoundError:
        st.info("Локальный файл titanic.csv не найден. Обратимся к всемирной сети")
        df = sns.load_dataset('titanic')
    df.rename(columns={
        'survived': 'Выжил',
        'pclass': 'Класс',
        'sex': 'Пол',
        'age': 'Возраст',
        'sibsp': 'СибСп',
        'parch': 'Парч',
        'fare': 'Стоимость',
        'embarked': 'Порт',
        'class': 'Класс_кат',
        'who': 'Кто',
        'adult_male': 'Взрослый_муж',
        'deck': 'Палуба',
        'embark_town': 'Город_посадки',
        'alive': 'Статус',
        'alone': 'Один'
    }, inplace=True)
    return df

df = load_data()


st.sidebar.header("Фильтры")
sex_filter = st.sidebar.selectbox("Пол", options=["Все", "male", "female"])
class_filter = st.sidebar.selectbox("Класс", options=["Все", 1, 2, 3])
age_min = int(df['Возраст'].min())
age_max = int(df['Возраст'].max())
age_range = st.sidebar.slider("Возраст", min_value=age_min, max_value=age_max, value=(age_min, age_max))

filtered_df = df.copy()
if sex_filter != "Все":
    filtered_df = filtered_df[filtered_df['Пол'] == sex_filter]
if class_filter != "Все":
    filtered_df = filtered_df[filtered_df['Класс'] == class_filter]
filtered_df = filtered_df[(filtered_df['Возраст'] >= age_range[0]) & (filtered_df['Возраст'] <= age_range[1])]

st.header("Cтатистика")
col1, col2 = st.columns(2)
with col1:
    st.write("Первые 5 строк данных:")
    st.dataframe(df.head())
with col2:
    st.write("Информация о датасете:")
    st.write(f"Количество строк: {df.shape[0]}")
    st.write(f"Количество столбцов: {df.shape[1]}")
    st.write("Типы данных:")
    st.write(df.dtypes)

if st.checkbox("Показать сводную статистику числовых признаков"):
    st.write(df.describe())

st.header("Вывод произвольного количества строк")
n_rows = st.number_input("Введите количество строк", min_value=1, max_value=len(df), value=5, step=1)
st.write(f"Первые {n_rows} строк таблицы:")
st.dataframe(filtered_df.head(n_rows))

st.header("Графики")

st.subheader("1. Распределение возраста ")
age_sex = st.selectbox("Выберите пол для графика возраста", options=["Все", "male", "female"])
if age_sex == "Все":
    age_data = filtered_df['Возраст'].dropna()
    title = "Возраст всех пассажиров"
else:
    age_data = filtered_df[filtered_df['Пол'] == age_sex]['Возраст'].dropna()
    title = f"Возраст пассажиров ({age_sex})"

fig_age = px.histogram(x=age_data, nbins=30, title=title,
                       labels={'x': 'Возраст', 'y': 'Количество'},
                       template='simple_white')
st.plotly_chart(fig_age, use_container_width=True)


st.subheader("2. Выживаемость по полу")
surv_sex = filtered_df.groupby('Пол')['Выжил'].mean().reset_index()
fig_sex = px.bar(surv_sex, x='Пол', y='Выжил',
                 title="Доля выживших по полу",
                 labels={'Выжил': 'Доля выживших'},
                 text_auto='.2f',
                 template='simple_white')
st.plotly_chart(fig_sex, use_container_width=True)


st.subheader("3. Выживаемость по классу")
surv_class = filtered_df.groupby('Класс')['Выжил'].mean().reset_index()
fig_class = px.bar(surv_class, x='Класс', y='Выжил',
                   title="Доля выживших по классу",
                   labels={'Выжил': 'Доля выживших'},
                   text_auto='.2f',
                   template='simple_white')
st.plotly_chart(fig_class, use_container_width=True)

st.subheader("4. Зависимость стоимости билета от возраста")
fig_scatter = px.scatter(filtered_df.dropna(subset=['Возраст', 'Стоимость']),
                         x='Возраст', y='Стоимость',
                         color='Выжил',
                         title="Возраст и стоимость билета",
                         labels={'Выжил': 'Выжил (1=да, 0=нет)'},
                         template='simple_white')
st.plotly_chart(fig_scatter, use_container_width=True)

st.subheader("5. Корреляция числовых признаков")
numeric_cols = ['Выжил', 'Класс', 'Возраст', 'СибСп', 'Парч', 'Стоимость']
corr = filtered_df[numeric_cols].corr()
fig_corr = go.Figure(data=go.Heatmap(
    z=corr.values,
    x=corr.columns,
    y=corr.columns,
    colorscale='RdBu',
    zmin=-1, zmax=1,
    text=corr.round(2).values,
    texttemplate='%{text}',
    textfont={"size": 10}
))
fig_corr.update_layout(title="Матрица корреляции")
st.plotly_chart(fig_corr, use_container_width=True)
