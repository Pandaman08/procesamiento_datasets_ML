import matplotlib.pyplot as plt

def plot_iris_scatter(X, y):
    plt.figure(figsize=(6, 4))
    for cls in y.unique():
        subset = X[y == cls]
        plt.scatter(subset["sepal length (cm)"], subset["petal length (cm)"], label=f"Clase {cls}", alpha=0.7)
    plt.xlabel("Sepal Length (cm)")
    plt.ylabel("Petal Length (cm)")
    plt.legend()
    return plt