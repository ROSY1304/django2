import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestRegressor
import joblib
import os
import json # Para cargar las features

print("--- Iniciando script para generar modelos escalados ---")

# --- 1. Definir Rutas ---
# Asegúrate de que esta ruta sea la correcta a tu proyecto Django
django_project_path = '/home/rosy/Escritorio/Pro. Log. y Func./Django/'
dataset_path = os.path.join(django_project_path, 'datasets', 'TotalFeatures-ISCXFlowMeter.csv')
model_dir_path = os.path.join(django_project_path, 'modelo')
features_path = os.path.join(model_dir_path, 'columnas_features.json') # Necesario para saber qué columnas usar

# Crear la carpeta 'modelo' si no existe
os.makedirs(model_dir_path, exist_ok=True)
print(f"Directorio de modelos: {model_dir_path}")

# --- 2. Cargar Datos y Features ---
try:
    df = pd.read_csv(dataset_path)
    print(f"Dataset cargado: {dataset_path} (Tamaño: {df.shape})")
except FileNotFoundError:
    print(f"ERROR: No se encontró el dataset en {dataset_path}")
    exit() # Salir si no hay datos
except Exception as e:
    print(f"ERROR al cargar el dataset: {e}")
    exit()

try:
    with open(features_path, 'r') as f:
        columnas_features = json.load(f)
    print("Archivo de características cargado.")
    # Verificar que las columnas existan en el DataFrame
    missing_features = [col for col in columnas_features if col not in df.columns]
    if missing_features:
        print(f"ERROR: Columnas faltantes en el dataset: {missing_features}")
        exit()
except FileNotFoundError:
    print(f"ERROR: No se encontró el archivo de características en {features_path}")
    exit()
except Exception as e:
    print(f"ERROR al cargar/verificar características: {e}")
    exit()

target_column = 'calss'
if target_column not in df.columns:
    print(f"ERROR: Columna target '{target_column}' no encontrada.")
    exit()

# --- 3. Preparar Datos (Factorize y Split) ---
# Usar todo el dataset para entrenar los modelos guardados es lo ideal,
# pero si es muy grande, puedes tomar una muestra grande aquí.
# Para este ejemplo, usaremos todo:
X = df[columnas_features]
y_class_str = df[target_column]
y_numeric, _ = pd.factorize(y_class_str) # Convertir a números (0, 1, 2...)

# Dividir solo para obtener X_train (necesario para ajustar el scaler)
# No necesitamos test/val aquí, solo entrenar y guardar
# Usamos una proporción grande para entrenamiento (ej. 80%)
X_train, _, y_train, _ = train_test_split(X, y_numeric, train_size=0.8, random_state=42, stratify=y_numeric)
print(f"Datos divididos para entrenamiento (Tamaño X_train: {X_train.shape})")

# --- 4. Ajustar el Scaler ---
print("Ajustando RobustScaler con X_train...")
scaler = RobustScaler()
# Ajustar SOLO con X_train
scaler.fit(X_train)
print("Scaler ajustado.")

# --- 5. Escalar X_train ---
print("Transformando X_train con el scaler ajustado...")
X_train_scaled = scaler.transform(X_train)
print("X_train transformado.")

# --- 6. Entrenar el Regresor Escalado ---
print("Entrenando RandomForestRegressor con datos escalados...")
# Asegúrate de usar los mismos parámetros que en tu notebook si son importantes
rf_reg_scaled = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf_reg_scaled.fit(X_train_scaled, y_train)
print("RandomForestRegressor (escalado) entrenado.")

# --- 7. Guardar el Modelo Escalado y el Scaler ---
scaled_regressor_path = os.path.join(model_dir_path, 'rf_regressor_scaled.joblib')
scaler_path = os.path.join(model_dir_path, 'robust_scaler.joblib')

try:
    joblib.dump(rf_reg_scaled, scaled_regressor_path)
    print(f"Modelo de regresión escalado GUARDADO en: {scaled_regressor_path}")
except Exception as e:
    print(f"ERROR al guardar rf_reg_scaled: {e}")

try:
    joblib.dump(scaler, scaler_path)
    print(f"Scaler GUARDADO en: {scaler_path}")
except Exception as e:
    print(f"ERROR al guardar scaler: {e}")

print("--- Script finalizado ---")