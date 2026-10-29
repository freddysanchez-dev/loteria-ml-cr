# app/main.py
from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, JSONResponse
import pandas as pd
import os
import sys
import logging
import traceback

# Configurar el logger para que muestre TODO en consola
logging.basicConfig(
    level=logging.DEBUG,  # Nivel DEBUG para ver todo
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.model import predecir_top_n, CSV_PATH, MODEL_PATH

app = FastAPI(title="Lotería Nacional CR - ML")

_RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_TEMPLATES_DIR = os.path.join(_RAIZ, "app", "templates")
templates = Jinja2Templates(directory=_TEMPLATES_DIR)

logging.info(f"📁 Templates en: {_TEMPLATES_DIR}")
logging.info(f"   - index.html existe: {os.path.exists(os.path.join(_TEMPLATES_DIR, 'index.html'))}")
logging.info(f"   - analytics.html existe: {os.path.exists(os.path.join(_TEMPLATES_DIR, 'analytics.html'))}")


# Reemplaza tu función leer_interfaz con esto:
@app.get("/", response_class=HTMLResponse)
async def leer_interfaz(request: Request):
    try:
        top = predecir_top_n(n=8)
        top_items = [(f"{n:02d}", f"{p*100:.2f}%") for n, p in top]
        numero_recomendado = top_items[0][0] if top_items else "--"
        prob_recomendado = top_items[0][1] if top_items else "0.00%"
        
        # Datos para los gráficos (los pasamos directo al HTML)
        df = pd.read_csv(CSV_PATH)
        conteo = df['Numero_Ganador'].value_counts().reindex(range(100), fill_value=0).tolist()
        top10 = df['Numero_Ganador'].value_counts().head(10)
        top10_data = [{"numero": int(n), "frecuencia": int(f)} for n, f in top10.items()]
        por_sorteo = df['Sorteo'].value_counts().to_dict()

        return templates.TemplateResponse("index.html", {
            "request": request,
            "numero_recomendado": numero_recomendado,
            "prob_recomendado": prob_recomendado,
            "top_items": top_items,
            "N_TOTAL": len(df),
            "exactitud_modelo": "9.05%",
            "exactitud_azar": "8.00%",
            "prob_top8_acumulada": f"{sum(p for _, p in top)*100:.2f}%",
            # Datos para graficos
            "conteo_numeros": conteo,
            "top10": top10_data,
            "por_sorteo": por_sorteo
        })
    except Exception as e:
        logging.error(traceback.format_exc())
        return templates.TemplateResponse("index.html", {"request": request, "error": str(e)})

@app.get("/api/prediccion")
async def api_prediccion():
    top = predecir_top_n(n=8)
    return JSONResponse({
        "top": [{"numero": n, "probabilidad": p} for n, p in top]
    })


@app.get("/api/analytics")
async def api_analytics():
    df = pd.read_csv(CSV_PATH)
    df['Fecha'] = pd.to_datetime(df['Fecha'], errors='coerce')
    df = df.sort_values('Fecha').reset_index(drop=True)

    conteo = df['Numero_Ganador'].value_counts().reindex(range(100), fill_value=0).tolist()
    top10 = df['Numero_Ganador'].value_counts().head(10)
    top10_data = [{"numero": int(n), "frecuencia": int(f)} for n, f in top10.items()]
    por_sorteo = df['Sorteo'].value_counts().to_dict()

    media_movil = df['Numero_Ganador'].rolling(30, min_periods=1).mean().round(2)
    paso = max(1, len(media_movil) // 200)
    fechas = df['Fecha'].dt.strftime('%Y-%m-%d').iloc[::paso].tolist()
    media_movil_list = media_movil.iloc[::paso].tolist()

    return JSONResponse({
        "conteo_numeros": conteo,
        "top10": top10_data,
        "por_sorteo": por_sorteo,
        "fechas": fechas,
        "media_movil": media_movil_list,
    })


@app.get("/analytics", response_class=HTMLResponse)
async def analytics(request: Request):
    return templates.TemplateResponse("analytics.html", {"request": request})


if __name__ == "__main__":
    import uvicorn
    # reload=False para evitar comportamientos extraños, y log_level="debug"
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="debug")