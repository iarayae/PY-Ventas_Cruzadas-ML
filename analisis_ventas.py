import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules


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

# Generar combinaciones frecuentes con al menos 5% de soporte
frecuentes = apriori(df_matrix, min_support=0.05, use_colnames=True)

# Mostrar los resultados
print("Conjuntos frecuentes de productos (soporte mínimo 5%): ")
print(frecuentes.sort_values(by="support", ascending=False))

# Generar reglas desde los itemsets frecuentes
reglas = association_rules(frecuentes, metric="confidence", min_threshold=0.4)

# Filtrar reglas útiles (al menos un producto en antecedente y consecuente)
reglas = reglas[(reglas["antecedents"].apply(len) >= 1) & (reglas["consequents"].apply(len) >= 1)]

# Mostrar reglas ordenadas por confianza
print("\nReglas de asociación (confianza mínima 40%): ")
print(reglas[["antecedents", "consequents", "support", "confidence", "lift"]].sort_values(by="confidence", ascending=False))

# Guardar resultados
output_path = "output/reglas_ventas.xlsx"
reglas.to_excel(output_path, index=False)
print(f"\nReglas guardadas en: {output_path}")