# app/model.py
import os
import sys
import joblib
import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.feature_engineering import crear_features

_RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(_RAIZ, 'data', 'models', 'modelo_rf.pkl')
CSV_PATH = os.path.join(_RAIZ, 'data', 'processed', 'dataset_unificado.csv')

_modelo = None

def cargar_modelo():
    global _modelo
    if _modelo is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Modelo no encontrado en {MODEL_PATH}. Entrena con: python src/train.py")
        _modelo = joblib.load(MODEL_PATH)
    return _modelo


def predecir_top_n(n=8):
    """Usa el último estado del histórico para predecir el próximo sorteo."""
    modelo = cargar_modelo()
    df = pd.read_csv(CSV_PATH)
    df['Fecha'] = pd.to_datetime(df['Fecha'], errors='coerce')
    df = df.dropna(subset=['Fecha']).sort_values('Fecha').reset_index(drop=True)

    X, _ = crear_features(df)
    x_ultimo = X.iloc[-1:]

    proba = modelo.predict_proba(x_ultimo)
    proba_vector = np.zeros(100)
    for num, p in enumerate(proba):
        if p.shape[1] == 2:
            proba_vector[num] = p[0, 1]

    ordenados = np.argsort(proba_vector)[::-1][:n]
    return [(int(num), float(proba_vector[num])) for num in ordenados]