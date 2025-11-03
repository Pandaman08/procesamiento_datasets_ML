import streamlit as st
from src.utils import load_titanic, load_student, load_iris
from src.preprocessing import preprocess_titanic, preprocess_student, preprocess_iris
from src.visualization import plot_iris_scatter
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Procesamiento ML", layout="wide")
st.title("Actividad Individual: Procesamiento de Datasets en ML")

with st.expander("Ejercicio 1: Titanic"):
    df_t = load_titanic()
    X_t, y_t = preprocess_titanic(df_t)
    X_train, X_test, y_train, y_test = train_test_split(X_t, y_t, test_size=0.3, random_state=42)
    st.write("Primeros 5 registros procesados:")
    st.dataframe(X_t.head())
    st.write(f"Entrenamiento: {X_train.shape}, Prueba: {X_test.shape}")

with st.expander("Ejercicio 2: Student Performance"):
    df_s = load_student()
    X_s, y_s = preprocess_student(df_s)
    X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(X_s, y_s, test_size=0.2, random_state=42)
    st.write("Primeros 5 registros procesados:")
    st.dataframe(X_s.head())
    st.write(f"Entrenamiento: {X_train_s.shape}, Prueba: {X_test_s.shape}")
    corr = df_s[["G1", "G2", "G3"]].corr()
    st.write("Matriz de correlación (G1, G2, G3):")
    st.dataframe(corr)

with st.expander("Ejercicio 3: Iris"):
    df_i = load_iris()
    df_scaled, X_raw, y_i = preprocess_iris(df_i)
    X_train_i, X_test_i, y_train_i, y_test_i = train_test_split(X_raw, y_i, test_size=0.3, random_state=42)
    st.write("Estadísticas del dataset estandarizado:")
    st.dataframe(df_scaled.describe())
    fig = plot_iris_scatter(X_raw, y_i)
    st.pyplot(fig)