# Este script se basa en tu notebook.
# Deberás ejecutarlo una vez para crear los archivos del modelo.
# Asegúrate de tener el archivo 'TotalFeatures-ISCXFlowMeter.csv' en la carpeta 'datasets/'.
# (Este script debe ejecutarse desde la carpeta raíz del proyecto 'Django')

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from pandas import DataFrame
import joblib
import json
import os

# --- Constantes ---
TARGET_COLUMN = 'calss'
DATASET_PATH = 'datasets/TotalFeatures-ISCXFlowMeter.csv'
RUTA_MODELO_DIR = 'modelo'

# --- Funciones Auxiliares (de tu script) ---
def train_val_test_split(df, rstate=42, shuffle=True, stratify=None):
    strat = df[stratify] if stratify else None
    train_set, test_set = train_test_split(
        df, test_size=0.4, random_state=rstate, shuffle=shuffle, stratify=strat)
    strat = test_set[stratify] if stratify else None
    val_set, test_set = train_test_split(
        test_set, test_size=0.5, random_state=rstate, shuffle=shuffle, stratify=strat)
    return (train_set, val_set, test_set)

def remove_labels(df, label_name):
    X = df.drop(label_name, axis=1)
    y = df[label_name].copy()
    return (X, y)

print("Iniciando el script de entrenamiento...")

# --- 1. Carga y Preparación de Datos ---
try:
    df = pd.read_csv(DATASET_PATH)
except FileNotFoundError:
    print(f"Error: No se encontró el archivo '{DATASET_PATH}'")
    print("Asegúrate de que el archivo CSV esté en la ubicación correcta y que ejecutas el script desde la raíz del proyecto.")
    exit()

print("Datos cargados exitosamente.")

# Copia para transformación
X_prep = df.copy()

# --- CORRECCIÓN AÑADIDA ---
# Limpiar NaNs ANTES de hacer cualquier otra cosa.
# Esto asegura que el modelo se entrene con los mismos datos que usará la API.
print(f"Filas antes de limpiar NaNs: {len(X_prep)}")
features = [col for col in df.columns if col != TARGET_COLUMN]
X_prep = X_prep.dropna(subset=features + [TARGET_COLUMN])
print(f"Filas después de limpiar NaNs: {len(X_prep)}")
# ---------------------------

# Guardar el mapeo de etiquetas ANTES de dividir los datos
# Esto asegura que tengamos todas las etiquetas posibles
etiquetas_factorizadas, etiquetas_unicas = pd.factorize(X_prep[TARGET_COLUMN])
X_prep[TARGET_COLUMN] = etiquetas_factorizadas

# Convertir los índices (0, 1, 2...) a las etiquetas string ("benign", "adware"...)
# Nota: json guarda las claves como strings, así que guardamos como {"0": "label1", "1": "label2"}
mapeo_etiquetas = {str(i): label for i, label in enumerate(etiquetas_unicas)}

print(f"Mapeo de etiquetas creado (primeras 5): {list(mapeo_etiquetas.items())[:5]}")

# --- 2. División del DataSet ---
train_set, val_set, test_set = train_val_test_split(X_prep, stratify=TARGET_COLUMN)

X_train, y_train = remove_labels(train_set, TARGET_COLUMN)
X_val, y_val = remove_labels(val_set, TARGET_COLUMN)
X_test, y_test = remove_labels(test_set, TARGET_COLUMN)

print(f"Datos divididos: {len(X_train)} de entrenamiento.")

# Guardar la lista de columnas (features)
# El modelo esperará los datos de entrada exactamente en este orden
lista_features = list(X_train.columns)
print(f"El modelo usará {len(lista_features)} features.")

# --- 3. Entrenamiento del Modelo ---
# Basado en tu análisis, el RandomForestClassifier SIN escalar dio el mejor resultado.
print("Entrenando RandomForestClassifier (sin escalar)...")
rf_clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)

# Entrenamos el modelo con el conjunto de entrenamiento
rf_clf.fit(X_train, y_train)

print("Modelo entrenado.")

# --- 4. Guardar Artefactos del Modelo ---
# Crear la carpeta 'modelo' si no existe
if not os.path.exists(RUTA_MODELO_DIR):
    os.makedirs(RUTA_MODELO_DIR)

# Guardar el modelo entrenado
ruta_modelo_joblib = os.path.join(RUTA_MODELO_DIR, 'rf_clasificador.joblib')
joblib.dump(rf_clf, ruta_modelo_joblib)
print(f"Modelo guardado en: {ruta_modelo_joblib}")

# Guardar el mapeo de etiquetas
ruta_mapeo = os.path.join(RUTA_MODELO_DIR, 'mapeo_etiquetas.json')
with open(ruta_mapeo, 'w') as f:
    json.dump(mapeo_etiquetas, f, indent=4)
print(f"Mapeo de etiquetas guardado en: {ruta_mapeo}")

# Guardar la lista de features
ruta_features = os.path.join(RUTA_MODELO_DIR, 'columnas_features.json')
with open(ruta_features, 'w') as f:
    json.dump(lista_features, f, indent=4)
print(f"Lista de features guardada en: {ruta_features}")

print("\n¡Entrenamiento completado y artefactos guardados!")
print("Ahora puedes iniciar tu servidor Django.")