import pandas as pd

def load_titanic():
    return pd.read_csv("data/raw/titanic.csv")

def load_student():
    return pd.read_csv("data/raw/student-mat.csv", sep=";")

def load_iris():
    from sklearn.datasets import load_iris
    data = load_iris(as_frame=True)
    df = data.frame
    return df
