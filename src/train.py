# src/train.py
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import log_loss
import joblib
import os
import logging
from src.feature_engineering import crear_features

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def hit_rate_top_k(y_true_binario, proba_matrix, k=8):
    """Calcula el hit rate top-k para un problema multi-etiqueta."""
    hits = 0
    for i in range(len(y_true_binario)):
        real = int(y_true_binario.iloc[i].values.argmax())
        top_k = np.argsort(proba_matrix[i])[-k:]
        if real in top_k:
            hits += 1
    return hits / len(y_true_binario)

def extraer_probabilidades(proba_lista, n_filas):
    """Convierte la lista de arrays de predict_proba en una matriz (n_filas x 100)."""
    matriz = np.zeros((n_filas, 100))
    for num, p in enumerate(proba_lista):
        if p.shape[1] == 2:
            matriz[:, num] = p[:, 1]
        else:
            matriz[:, num] = 0.0
    return matriz

def entrenar():
    csv_path = 'data/processed/dataset_unificado.csv'
    if not os.path.exists(csv_path):
        logging.error("⚠️ No existe el CSV. Ejecuta primero: python src/data_ingestion.py")
        return

    df = pd.read_csv(csv_path)
    logging.info(f"📊 Total de sorteos: {len(df)}")
    logging.info(f"   Rango: {df['Fecha'].min()} a {df['Fecha'].max()}")

    # Crear features y target
    X, y = crear_features(df)
    
    # Eliminar la primera fila (sin histórico)
    X = X.iloc[1:].reset_index(drop=True)
    y = y.iloc[1:].reset_index(drop=True)

    logging.info(f"   X shape: {X.shape} | y shape: {y.shape}")

    tscv = TimeSeriesSplit(n_splits=5)
    resultados_top8 = []
    resultados_top1 = []
    log_losses = []

    logging.info("\n🔍 Validación temporal (Walk-Forward):")
    for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        clf = MultiOutputClassifier(
            RandomForestClassifier(n_estimators=50, max_depth=6, random_state=42, n_jobs=-1)
        )
        clf.fit(X_train, y_train)
        
        proba = clf.predict_proba(X_test)
        proba_matrix = extraer_probabilidades(proba, len(X_test))

        # Calcular métricas
        top8 = hit_rate_top_k(y_test, proba_matrix, k=8)
        top1 = hit_rate_top_k(y_test, proba_matrix, k=1)
        
        # Log loss (aproximado para multi-etiqueta)
        # Nota: log_loss requiere probabilidades para cada clase, aquí usamos la matriz
        # Para simplificar, calculamos la entropía cruzada promedio de la clase real
        real_indices = y_test.values.argmax(axis=1)
        prob_real = proba_matrix[np.arange(len(real_indices)), real_indices]
        prob_real = np.clip(prob_real, 1e-15, 1 - 1e-15) # Evitar log(0)
        loss = -np.mean(np.log(prob_real))
        
        resultados_top8.append(top8)
        resultados_top1.append(top1)
        log_losses.append(loss)
        
        logging.info(f"   Fold {fold+1}: Top-8 = {top8:.2%} | Top-1 = {top1:.2%} | LogLoss = {loss:.4f}")

    logging.info(f"\n📈 Resultados promedio:")
    logging.info(f"   Hit Rate Top-8 : {np.mean(resultados_top8):.2%}  (azar: 8.00%)")
    logging.info(f"   Hit Rate Top-1 : {np.mean(resultados_top1):.2%}  (azar: 1.00%)")
    logging.info(f"   Log Loss       : {np.mean(log_losses):.4f}")

    # Comprobación de honestidad
    if np.mean(resultados_top1) > 0.05:
        logging.warning("\n   ⚠️ Top-1 > 5%: Posible fuga de datos. Revisar extracción de PDFs y features.")
    else:
        logging.info("\n   ✅ Modelo honesto: no supera el azar significativamente. Es lo esperado en lotería.")

    # Entrenamiento final con todos los datos
    logging.info("\n🚀 Entrenando modelo final con todos los datos...")
    clf_final = MultiOutputClassifier(
        RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42, n_jobs=-1)
    )
    clf_final.fit(X, y)
    os.makedirs('data/models', exist_ok=True)
    joblib.dump(clf_final, 'data/models/modelo_rf.pkl')
    logging.info("✅ Modelo guardado en data/models/modelo_rf.pkl")

if __name__ == "__main__":
    entrenar()