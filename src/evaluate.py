# src/evaluate.py
import pandas as pd
import numpy as np
import joblib
import logging
from src.feature_engineering import crear_features

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def evaluar_modelo():
    # Cargar datos y modelo
    df = pd.read_csv('data/processed/dataset_unificado.csv')
    modelo = joblib.load('data/models/modelo_rf.pkl')
    
    X, y = crear_features(df)
    X = X.iloc[1:].reset_index(drop=True)
    y = y.iloc[1:].reset_index(drop=True)
    
    # Usar el último 20% como test hold-out (nunca visto en entrenamiento)
    split_idx = int(len(X) * 0.8)
    X_test = X.iloc[split_idx:]
    y_test = y.iloc[split_idx:]
    
    logging.info(f"Evaluando en {len(X_test)} sorteos no vistos...")
    
    proba = modelo.predict_proba(X_test)
    proba_matrix = np.zeros((len(X_test), 100))
    for num, p in enumerate(proba):
        if p.shape[1] == 2:
            proba_matrix[:, num] = p[:, 1]
    
    # Calcular hit rate top-k
    for k in [1, 5, 8, 10]:
        hits = 0
        for i in range(len(y_test)):
            real = int(y_test.iloc[i].values.argmax())
            top_k = np.argsort(proba_matrix[i])[-k:]
            if real in top_k:
                hits += 1
        hit_rate = hits / len(y_test)
        logging.info(f"   Hit Rate Top-{k}: {hit_rate:.2%} (azar: {k}%)")

if __name__ == "__main__":
    evaluar_modelo()