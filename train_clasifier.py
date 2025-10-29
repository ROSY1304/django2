# train_classifier.py
# Este script entrena y guarda SOLAMENTE el modelo RandomForestClassifier
# y los metadatos necesarios (features, mapeo de etiquetas).

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier # <-- CLASIFICADOR
# from sklearn.preprocessing import RobustScaler # No necesario aquí
# from sklearn.ensemble import RandomForestRegressor # No necesario aquí
from pandas import DataFrame
import joblib
import json
import os

print("--- Iniciando script para entrenar CLASIFICADOR ---")

# --- 1. Definir Rutas y Constantes ---
django_project_path = '/home/rosy/Escritorio/Pro. Log. y Func./Django/' # Ajusta si es necesario
dataset_path = os.path.join(django_project_path, 'datasets', 'TotalFeatures-ISCXFlowMeter.csv')
model_dir_path = os.path.join(django_project_path, 'modelo')
target_column = 'calss'

os.makedirs(model_dir_path, exist_ok=True)
print(f"Directorio de modelos: {model_dir_path}")

# --- 2. Cargar Datos y Features ---
try:
    df = pd.read_csv(dataset_path)
    print(f"Dataset cargado: {dataset_path} (Tamaño: {df.shape})")
except FileNotFoundError:
    print(f"ERROR: No se encontró el dataset en {dataset_path}")
    exit()
except Exception as e:
    print(f"ERROR al cargar el dataset: {e}")
    exit()

if target_column not in df.columns:
    print(f"ERROR: Columna target '{target_column}' no encontrada.")
    exit()

# Definir features ANTES de limpiar NaNs (para guardar la lista completa si es necesario)
features = [col for col in df.columns if col != target_column]

# Limpiar NaNs (Importante que coincida con la API)
print(f"Filas antes de limpiar NaNs: {len(df)}")
df = df.dropna(subset=features + [target_column])
print(f"Filas después de limpiar NaNs: {len(df)}")
if len(df) == 0:
    print("ERROR: El dataset quedó vacío después de limpiar NaNs.")
    exit()

# --- 3. Preparar Datos (Factorize y Split) ---
X = df[features]
y_class_str = df[target_column]
y_numeric, unique_labels = pd.factorize(y_class_str) # Convertir a números (0, 1, 2...)

# Crear y guardar mapeo ANTES de dividir para capturar todas las clases
# Mapeo: {"0": "benign", "1": "asware", ...}
mapeo_etiquetas = {str(i): label for i, label in enumerate(unique_labels)}
ruta_mapeo = os.path.join(model_dir_path, 'mapeo_etiquetas.json')
try:
    with open(ruta_mapeo, 'w') as f:
        json.dump(mapeo_etiquetas, f, indent=4)
    print(f"Mapeo de etiquetas guardado en: {ruta_mapeo}")
except Exception as e:
    print(f"ERROR al guardar mapeo: {e}")

# Guardar lista de features usadas para entrenar
ruta_features = os.path.join(model_dir_path, 'columnas_features.json')
try:
    with open(ruta_features, 'w') as f:
        json.dump(features, f, indent=4) # Guardamos la lista 'features'
    print(f"Lista de features guardada en: {ruta_features}")
except Exception as e:
    print(f"ERROR al guardar features: {e}")


# Dividir datos para entrenamiento
# Usaremos una división simple aquí, ya que solo necesitamos entrenar el modelo final
X_train, _, y_train, _ = train_test_split(X, y_numeric, train_size=0.8, random_state=42, stratify=y_numeric)
print(f"Datos divididos para entrenamiento (Tamaño X_train: {X_train.shape})")

# --- 4. Entrenar el CLASIFICADOR ---
print("Entrenando RandomForestClassifier (sin escalar)...")
# Usar los mismos hiperparámetros que en tu notebook si son importantes
rf_clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_clf.fit(X_train, y_train) # Entrenar con datos SIN escalar y etiquetas NUMÉRICAS
print("Clasificador entrenado.")

# --- 5. Guardar el CLASIFICADOR Entrenado ---
classifier_path = os.path.join(model_dir_path, 'rf_clasificador.joblib')
try:
    joblib.dump(rf_clf, classifier_path)
    print(f"Modelo CLASIFICADOR guardado en: {classifier_path}")
except Exception as e:
    print(f"ERROR al guardar rf_clf: {e}")

print("--- Script finalizado ---")