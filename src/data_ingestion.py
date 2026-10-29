# src/data_ingestion.py
import pdfplumber
import pandas as pd
import re
import os
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def extraer_nuevos_tiempos(ruta_pdf):
    """
    Formato: Sorteo Fecha Hora Número Descripción
    Ejemplo: 16907 02/01/2019 12:55:00 pm 47 Paga 70 veces...
    El número ganador es el 4to token.
    """
    datos = []
    with pdfplumber.open(ruta_pdf) as pdf:
        for pagina in pdf.pages:
            texto = pagina.extract_text()
            if not texto:
                continue
            for linea in texto.split('\n'):
                # Buscar patrón: Sorteo Fecha Hora Número
                # Ejemplo: 16907 02/01/2019 12:55:00 pm 47
                match = re.search(r'(\d{4,5})\s+(\d{2}/\d{2}/\d{4})\s+(\d{1,2}:\d{2}:\d{2}\s*[ap]m)\s+(\d{1,2})\s+', linea)
                if match:
                    fecha = match.group(2)
                    numero_ganador = int(match.group(4))
                    if 0 <= numero_ganador <= 99:
                        datos.append({
                            'Fecha': fecha,
                            'Sorteo': 'Nuevos_Tiempos',
                            'Numero_Ganador': numero_ganador
                        })
    return pd.DataFrame(datos)

def extraer_tabla_generica(ruta_pdf, tipo_sorteo):
    """
    Formato: Sorteo Fecha Primer_Número Primer_Serie ...
    Ejemplo: 4525 06/01/2019 12 613 170.000.000,00 ...
    El número ganador es el 3er token (índice 2).
    """
    datos = []
    with pdfplumber.open(ruta_pdf) as pdf:
        for pagina in pdf.pages:
            texto = pagina.extract_text()
            if not texto:
                continue
            for linea in texto.split('\n'):
                # Buscar patrón: Sorteo Fecha Número
                # Ejemplo: 4525 06/01/2019 12
                match = re.match(r'^(\d{4,5})\s+(\d{2}/\d{2}/\d{4})\s+(\d{1,2})\s+', linea)
                if match:
                    fecha = match.group(2)
                    numero_ganador = int(match.group(3))
                    if 0 <= numero_ganador <= 99:
                        datos.append({
                            'Fecha': fecha,
                            'Sorteo': tipo_sorteo,
                            'Numero_Ganador': numero_ganador
                        })
    return pd.DataFrame(datos)

def unificar_datos():
    directorio_raw = 'data/raw/'
    archivos = {
        'Nacional_Regular': ('Numeros_Favorecidos_2019_2025_Loteria_Nacional.pdf', 'generico'),
        'Nuevos_Tiempos': ('Numeros_Favorecidos_2019_2025_Nuevos_Tiempos.pdf', 'nuevos_tiempos'),
        'Nacional_Extra': ('Numeros_Favorecidos_Extraordianrios_2019_2025_Loteria_Nacional.pdf', 'generico'),
        'Popular': ('Numeros_Favorecidos_2019_2025_Loteria_Popular.pdf', 'generico')  # Corregido typo
    }
    
    dfs = []
    for sorteo, (archivo, tipo) in archivos.items():
        ruta = os.path.join(directorio_raw, archivo)
        if os.path.exists(ruta):
            logging.info(f"Procesando: {archivo} ({tipo})")
            if tipo == 'nuevos_tiempos':
                df_temp = extraer_nuevos_tiempos(ruta)
            else:
                df_temp = extraer_tabla_generica(ruta, sorteo)
            
            if not df_temp.empty:
                dfs.append(df_temp)
                logging.info(f"   → {len(df_temp)} filas extraídas")
            else:
                logging.warning(f"   → 0 filas extraídas. Revisar formato.")
        else:
            logging.warning(f"⚠️ No encontrado: {ruta}")
    
    if not dfs:
        raise ValueError("No se encontraron PDFs válidos en data/raw/")
    
    df = pd.concat(dfs, ignore_index=True)
    df['Fecha'] = pd.to_datetime(df['Fecha'], dayfirst=True, errors='coerce')
    df = df.dropna(subset=['Fecha']).sort_values('Fecha').reset_index(drop=True)
    
    # Validación: Eliminar duplicados exactos
    df = df.drop_duplicates(subset=['Fecha', 'Sorteo', 'Numero_Ganador'])
    
    # Validación: Números en rango 0-99
    df = df[(df['Numero_Ganador'] >= 0) & (df['Numero_Ganador'] <= 99)]
    
    os.makedirs('data/processed', exist_ok=True)
    df.to_csv('data/processed/dataset_unificado.csv', index=False)
    
    logging.info(f"✅ Datos unificados: {len(df)} filas guardadas")
    logging.info(f"   Distribución de sorteos:\n{df['Sorteo'].value_counts()}")
    return df

if __name__ == "__main__":
    unificar_datos()