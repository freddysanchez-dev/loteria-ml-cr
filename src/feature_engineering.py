# src/feature_engineering.py
import pandas as pd

def crear_features(df):
    """Crea features de frecuencias móviles evitando data leakage."""
    y_binario = pd.DataFrame(0, index=df.index, columns=range(100))
    for idx, row in df.iterrows():
        y_binario.loc[idx, int(row['Numero_Ganador'])] = 1

    features_dict = {}
    for ventana in [5, 10, 20, 50]:
        for num in range(100):
            features_dict[f'freq_{num}_w{ventana}'] = (
                y_binario[num].rolling(window=ventana, min_periods=1).mean().shift(1)
            )

    features = pd.concat(features_dict, axis=1).fillna(0)
    return features, y_binario