import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
import numpy as np

def preprocess_titanic(df):
    df = df.drop(columns=["Name","Ticket","Cabin"], errors='ignore')
    df['Age'].fillna(df['Age'].median(), inplace=True)
    df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
    df = pd.get_dummies(df, columns=['Sex', 'Embarked'], drop_first=True)
    numeric_cols = ['Age', 'Fare']
    scaler = StandardScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    X = df.drop(columns=['Survived'], errors='ignore')
    y = df['Survived'] if 'Survived' in df.columns else None
    return X, y
def preprocess_student(df):
    df = df.drop_duplicates()
    categorical_cols = ["school", "sex", "address", "famsize", "Pstatus", "Mjob", "Fjob",
                        "reason", "guardian", "schoolsup", "famsup", "paid", "activities",
                        "nursery", "higher", "internet", "romantic"]
    encoder = OneHotEncoder(drop="first", sparse_output=False)
    encoded = encoder.fit_transform(df[categorical_cols])
    encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(categorical_cols))
    numeric_cols = ["age", "absences", "G1", "G2"]
    scaler = StandardScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    X = pd.concat([df[numeric_cols].reset_index(drop=True), encoded_df], axis=1)
    y = df["G3"]
    return X, y

def preprocess_iris(df):
    from sklearn.preprocessing import StandardScaler
    X = df.drop(columns=["target"])
    y = df["target"]
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    X_scaled["target"] = y
    return X_scaled, X, y
