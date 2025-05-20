import pandas as pd

# Leer archivo excel
archivo = "datos/ventas_cruzadas_ampliado.xlsx"
df = pd.read_excel(archivo)

# Mostrar primeras filas
print("Vista previa de los datos")
print(df.head())

# Verificar cantidad de boletas y productos
print("\nTotal de boletas: ", df["Boleta"].nunique())
print("Total de filas: ", len(df))