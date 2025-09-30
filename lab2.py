"""
LABORATORIO 2 - ANÁLISIS DE DATOS CON PANDAS
Universidad Gerardo Barrios
Dataset: Titanic
"""

import pandas as pd
import numpy as np

print("INICIANDO ANÁLISIS DEL DATASET TITANIC")

try:
    df = pd.read_csv('train.csv')
    print("Dataset 'train.csv' cargado exitosamente!")
    
except FileNotFoundError:
    print("Error: No se encontró el archivo 'train.csv'")
    print("Asegúrate de que esté en la misma carpeta que este script")
    exit()

print("\n" + "="*60)
print("INFORMACIÓN BÁSICA DEL DATASET")
print("="*60)

print(f"📏 Dimensiones del dataset: {df.shape[0]} filas x {df.shape[1]} columnas")
print(f"📋 Columnas disponibles: {list(df.columns)}")

print("\n" + "="*60)
print("RESUMEN ESTADÍSTICO CON describe()")
print("="*60)

print("Resumen estadístico de columnas numéricas:")
resumen = df.describe()
print(resumen)

print("\n" + "="*60)
print("TIPOS DE DATOS CON dtypes")
print("="*60)

print("Tipos de datos de cada columna:")
print(df.dtypes)

print("\nAnálisis de tipos de datos:")
for columna in df.columns:
    tipo = df[columna].dtype
    print(f"{columna}: {tipo}", end="")
    
    if tipo in ['int64', 'float64']:
        print(" → Numérico: Estadísticas, gráficos, correlaciones")
    elif tipo == 'object':
        print(" → Texto/Categórico: Conteos, moda, análisis categórico")
    else:
        print(" → Otro tipo: Verificar formato")

print("\n" + "="*60)
print("PRIMEROS Y ÚLTIMOS REGISTROS")
print("="*60)

print("PRIMEROS 8 REGISTROS:")
print(df.head(8))

print("\nÚLTIMOS 8 REGISTROS:")
print(df.tail(8))

print("\nCOMPARACIÓN ENTRE PRIMERO Y ÚLTIMO REGISTRO:")
print("Primer registro - Pasajero ID:", df.iloc[0]['PassengerId'], "| Sobrevivió:", df.iloc[0]['Survived'])
print("Último registro - Pasajero ID:", df.iloc[-1]['PassengerId'], "| Sobrevivió:", df.iloc[-1]['Survived'])

print("\n" + "="*60)
print("ORDENAR RESULTADOS CON sort_values()")
print("="*60)

print("PRIMEROS 5 PASAJEROS MÁS JÓVENES:")
jovenes = df.sort_values(by='Age').head()
print(jovenes[['PassengerId', 'Name', 'Age', 'Pclass', 'Survived']])

print("\nPRIMEROS 5 PASAJEROS CON TARIFAS MÁS ALTAS:")
ricos = df.sort_values(by='Fare', ascending=False).head()
print(ricos[['PassengerId', 'Name', 'Fare', 'Pclass', 'Survived']])

print("\nORDENADO POR CLASE (1ª, 2ª, 3ª) Y DENTRO DE CADA CLASE POR EDAD:")
clase_edad = df.sort_values(by=['Pclass', 'Age']).head(8)
print(clase_edad[['PassengerId', 'Pclass', 'Name', 'Age', 'Fare']])

print("\n" + "="*60)
print("MEDIDAS ESTADÍSTICAS")
print("="*60)

print("ANÁLISIS ESTADÍSTICO DE LA EDAD (Age):")

edad_limpia = df['Age'].dropna()

media_edad = np.mean(edad_limpia)
mediana_edad = np.median(edad_limpia)
desviacion_edad = np.std(edad_limpia)

print(f"Media (promedio): {media_edad:.2f} años")
print(f"Mediana (valor central): {mediana_edad:.2f} años")
print(f"Desviación estándar: {desviacion_edad:.2f} años")
print(f"Mínimo: {np.min(edad_limpia):.0f} años")
print(f"Máximo: {np.max(edad_limpia):.0f} años")
print(f"Datos válidos: {len(edad_limpia)} de {len(df)} registros")

print("\nANÁLISIS ESTADÍSTICO DE LA TARIFA (Fare):")
media_fare = np.mean(df['Fare'])
mediana_fare = np.median(df['Fare'])
desviacion_fare = np.std(df['Fare'])

print(f"Media: ${media_fare:.2f}")
print(f"Mediana: ${mediana_fare:.2f}")
print(f"Desviación estándar: ${desviacion_fare:.2f}")
print(f"Mínimo: ${np.min(df['Fare']):.2f}")
print(f"Máximo: ${np.max(df['Fare']):.2f}")

print("\n" + "="*60)
print("ANÁLISIS ADICIONAL - SOBREVIVENCIA")
print("="*60)

total_pasajeros = len(df)
sobrevivientes = df['Survived'].sum()
tasa_supervivencia = (sobrevivientes / total_pasajeros) * 100

print(f"TOTAL DE PASAJEROS: {total_pasajeros}")
print(f"SOBREVIVIENTES: {sobrevivientes}")
print(f"FALLECIDOS: {total_pasajeros - sobrevivientes}")
print(f"TASA DE SUPERVIVENCIA: {tasa_supervivencia:.1f}%")

print("\nSUPERVIVENCIA POR CLASE:")
for clase in sorted(df['Pclass'].unique()):
    pasajeros_clase = df[df['Pclass'] == clase]
    surv_clase = pasajeros_clase['Survived'].sum()
    tasa_clase = (surv_clase / len(pasajeros_clase)) * 100
    print(f"  Clase {clase}: {surv_clase}/{len(pasajeros_clase)} ({tasa_clase:.1f}%)")

print("\n" + "="*60)
print("INTERPRETACIÓN DE RESULTADOS")
print("="*60)

print("""
a. 📋 DESCRIPCIÓN DEL DATASET:
   El dataset contiene información de los pasajeros del Titanic, incluyendo 
   edad, género, clase del boleto, tarifa pagada, puerto de embarque y si 
   sobrevivieron al naufragio.

b. 📊 INFORMACIÓN DEL RESUMEN ESTADÍSTICO:
   - La edad promedio es alrededor de 30 años, con una distribución amplia
   - Las tarifas varían significativamente, indicando diferencias económicas
   - Aproximadamente el 38% de los pasajeros sobrevivieron

c. 📈 TENDENCIAS DETECTADAS:
   - Los pasajeros de primera clase pagan tarifas mucho más altas
   - Hay correlación entre clase social y tasa de supervivencia
   - La edad muestra una distribución normal con algunos valores extremos

d. 🏆 CATEGORÍAS QUE SOBRESALEN:
   - Clase 1: Mayor tasa de supervivencia y tarifas más altas
   - Mujeres y niños: Prioridad en los protocolos de rescate
   - Pasajeros con tarifas altas: Mejores ubicaciones en el barco

e. 🔄 DIFERENCIAS PRIMEROS/ÚLTIMOS REGISTROS:
   - Los registros parecen estar ordenados por PassengerId
   - No hay patrón evidente de diferencias sistemáticas

f. 📐 VALOR DE LAS MEDIDAS ESTADÍSTICAS:
   - La desviación estándar alta en tarifas indica gran desigualdad económica
   - La diferencia entre media y mediana en tarifas sugiere sesgo hacia valores altos
   - La desviación estándar en edad indica diversidad generacional
""")

print("🎉 ANÁLISIS COMPLETADO EXITOSAMENTE!")
