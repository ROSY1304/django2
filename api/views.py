from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score # <-- CORREGIDO: Importar f1_score
from django.conf import settings # Importar settings

# --- IMPORTANTE: Configurar backend ANTES de importar pyplot ---
import matplotlib
matplotlib.use('Agg')
# -----------------------------------------------------------
import matplotlib.pyplot as plt
import io
import base64
from io import StringIO
from django.shortcuts import render
import os
import numpy as np

# --- Importaciones para Regresión (necesarias para los modelos de comparación) ---
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import RobustScaler
# ----------------------------------------

# --- Rutas Correctas para los modelos ---
# Modelo Principal: CLASIFICADOR
MODEL_PATH = os.path.join(settings.BASE_DIR, 'modelo', 'rf_clasificador.joblib')
FEATURES_PATH = os.path.join(settings.BASE_DIR, 'modelo', 'columnas_features.json')
LABELS_PATH = os.path.join(settings.BASE_DIR, 'modelo', 'mapeo_etiquetas.json')
DATASET_PATH = os.path.join(settings.BASE_DIR, 'datasets', 'TotalFeatures-ISCXFlowMeter.csv')
# Modelos/Scaler para COMPARACIÓN de Regresión
REG_NO_SCALE_PATH = os.path.join(settings.BASE_DIR, 'modelo', 'rf_regressor_no_scale.joblib')
REG_SCALED_PATH = os.path.join(settings.BASE_DIR, 'modelo', 'rf_regressor_scaled.joblib')
SCALER_PATH = os.path.join(settings.BASE_DIR, 'modelo', 'robust_scaler.joblib')
# -----------------------------------------

# --- Cargar TODOS los modelos y el scaler al inicio ---
try:
    rf_model_classifier = joblib.load(MODEL_PATH) # <-- CARGAR CLASIFICADOR
    rf_reg_no_scale_model = joblib.load(REG_NO_SCALE_PATH) # Regresor sin escalar
    rf_reg_scaled_model = joblib.load(REG_SCALED_PATH)     # Regresor escalado
    robust_scaler = joblib.load(SCALER_PATH)             # Scaler

    with open(FEATURES_PATH, 'r') as f:
        columnas_features = json.load(f)
    with open(LABELS_PATH, 'r') as f:
        mapeo_etiquetas = json.load(f) # Mapeo { "0": "benign", "1": "asware", ...}
    print("Clasificador, Regresores, Scaler y metadatos cargados correctamente.")
except Exception as e:
    print(f"Error crítico al cargar modelos/scaler/metadatos: {e}")
    rf_model_classifier = None
    rf_reg_no_scale_model = None
    rf_reg_scaled_model = None
    robust_scaler = None
    columnas_features = []
    mapeo_etiquetas = {}
# ----------------------------------------------------

# Función auxiliar para gráficos de regresión (sin cambios)
def create_regression_plot(y_true, y_pred, title):
    plt.figure(figsize=(10, 6))
    plt.scatter(y_true, y_pred, alpha=0.5, label='Predicciones')
    # Añadir línea ideal y=x, asegurando límites correctos
    min_val = min(np.min(y_true), np.min(y_pred))
    max_val = max(np.max(y_true), np.max(y_pred))
    line = np.linspace(min_val, max_val, 100)
    plt.plot(line, line, 'r--', label='Línea Ideal (y=x)')
    plt.xlabel("Valores Reales (Numéricos)")
    plt.ylabel("Valores Predichos (Numéricos)")
    plt.xlim(min_val - 0.1, max_val + 0.1) # Añadir un pequeño margen
    plt.ylim(min_val - 0.1, max_val + 0.1)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()
    return img_base64

@csrf_exempt
def train_and_evaluate(request):
    if request.method == 'POST':
        # Verificar que los modelos estén cargados
        if not all([rf_model_classifier, rf_reg_no_scale_model, rf_reg_scaled_model, robust_scaler]):
             return JsonResponse({'error': 'Uno o más modelos/scaler no se cargaron correctamente al iniciar. Revisa logs.'}, status=500)
        try:
            data = json.loads(request.body)
            num_samples = int(data.get('num_samples'))

            # Limitar num_samples
            MAX_SAMPLES_ALLOWED = 10000 # Ajusta si es necesario
            if num_samples > MAX_SAMPLES_ALLOWED:
                print(f"Advertencia: num_samples ({num_samples}) excede el límite. Se usará {MAX_SAMPLES_ALLOWED}.")
                num_samples = MAX_SAMPLES_ALLOWED

            # Cargar y preparar datos
            try:
                df = pd.read_csv(DATASET_PATH)
            except FileNotFoundError:
                return JsonResponse({'error': f'Dataset no encontrado: {DATASET_PATH}'}, status=404)
            # ... (otros errores de carga) ...

            target_column = 'calss'
            if target_column not in df.columns:
                 return JsonResponse({'error': f'Columna target "{target_column}" no encontrada.'}, status=400)
            missing_features = [col for col in columnas_features if col not in df.columns]
            if missing_features:
                 return JsonResponse({'error': f'Features faltantes: {missing_features}'}, status=400)

            # Limpiar NaNs ANTES de samplear
            df = df.dropna(subset=columnas_features + [target_column])

            if num_samples > len(df):
                num_samples = len(df) # Usar todo si se pide más de lo disponible
                print(f"Advertencia: num_samples ajustado al tamaño del dataset limpio: {num_samples}")

            df_subset = df.sample(n=num_samples, random_state=42).copy()

            X_subset = df_subset[columnas_features]
            y_class_subset_str = df_subset[target_column]
            y_reg_subset_numeric, _ = pd.factorize(y_class_subset_str)

            # Dividir datos
            X_train, X_test, y_train_reg, y_test_reg = train_test_split(
                X_subset, y_reg_subset_numeric, test_size=0.2, random_state=42, stratify=y_reg_subset_numeric
            )

            # --- 1. CLASIFICACIÓN (Modelo Principal) ---
            # Predecir usando el CLASIFICADOR cargado (devuelve números)
            y_pred_class_numeric = rf_model_classifier.predict(X_test)
            # Calcular F1 Score comparando números vs números
            f1 = f1_score(y_test_reg, y_pred_class_numeric, average='weighted', zero_division=0)

            # Visualización del árbol del CLASIFICADOR
            tree_img = None
            try:
                from sklearn.tree import export_graphviz
                from graphviz import Source
                # Usar un estimador del CLASIFICADOR
                estimator = rf_model_classifier.estimators_[0]
                class_names_list = sorted(mapeo_etiquetas, key=lambda k: int(k)) # Ordenar claves numéricas "0", "1"...
                class_names_values = [mapeo_etiquetas[k] for k in class_names_list] # Obtener nombres string
                dot_data = StringIO()
                export_graphviz(estimator, out_file=dot_data,
                                feature_names=columnas_features,
                                class_names=class_names_values, # Nombres string para el gráfico
                                filled=True, rounded=True,
                                special_characters=True,
                                max_depth=5)
                graph = Source(dot_data.getvalue())
                tree_img = base64.b64encode(graph.pipe(format='png')).decode('utf-8')
            except Exception as e:
                print(f"Error al generar imagen del árbol: {e}")

            # --- 2. Regresión (Comparación usando modelos cargados) ---
            # a) Predicciones SIN escalar (usando el REGRESOR cargado)
            y_pred_reg_no_scale = rf_reg_no_scale_model.predict(X_test)

            # b) Predicciones CON escalar (usando el REGRESOR y SCALER cargados)
            X_test_scaled = robust_scaler.transform(X_test)
            y_pred_reg_scaled = rf_reg_scaled_model.predict(X_test_scaled)

            # c) Generar gráficos de regresión (comparan números vs números)
            plot_reg_no_scale_b64 = create_regression_plot(
                y_test_reg, y_pred_reg_no_scale, "Comparación Regressor - SIN ESCALAR"
            )
            plot_reg_scaled_b64 = create_regression_plot(
                y_test_reg, y_pred_reg_scaled, "Comparación Regressor - CON ESCALADO"
            )

            # --- 3. Respuesta JSON (Enviando F1 Score) ---
            return JsonResponse({
                'f1_score': f1, # <-- CORREGIDO: Enviar F1 Score
                'tree_visualization': tree_img,
                'regression_plot_no_scale': plot_reg_no_scale_b64,
                'regression_plot_scaled': plot_reg_scaled_b64
            })

        except json.JSONDecodeError:
            return JsonResponse({'error': 'Invalid JSON'}, status=400)
        except ValueError as ve:
             import traceback
             print(traceback.format_exc())
             return JsonResponse({'error': f'Error de valor: {ve}'}, status=400)
        except Exception as e:
            import traceback
            print(traceback.format_exc())
            return JsonResponse({'error': f'Error general: {str(e)}'}, status=500)

    return JsonResponse({'error': 'Solo se permiten solicitudes POST'}, status=405)

def index(request):
    return render(request, 'index.html')