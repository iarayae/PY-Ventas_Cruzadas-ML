import pandas as pd
from mlxtend.preprocessing import TransactionEncoder

# Leer archivo excel
archivo = "datos/ventas_cruzadas_ampliado.xlsx"
df = pd.read_excel(archivo)

# Mostrar primeras filas
print("Vista previa de los datos")
print(df.head())

# Verificar cantidad de boletas y productos
print("\nTotal de boletas: ", df["Boleta"].nunique())
print("Total de filas: ", len(df))

# Agrupar productos por boletas en listas
transacciones = df.groupby("Boleta")["Nombre Producto"].apply(list).values.tolist()

# Codificar como matriz boleta-producto (1 si es compra, 0 si no)
te = TransactionEncoder()
transacciones_codificadas = te.fit(transacciones).transform(transacciones)

df_matrix = pd.DataFrame(transacciones_codificadas, columns=te.columns_)

# Mostrar las primeras filas de la matriz
print("Matriz transaccional (boleta vs producto)")
print(df_matrix.head())