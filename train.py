import pandas as pd

# Cargar el dataset
df = pd.read_csv('data/dataset.csv')  # Ruta a nuestro dataset

# Revisano las primeras filas
print(df.head())

# Eliminando filas nulas
df = df.dropna()
