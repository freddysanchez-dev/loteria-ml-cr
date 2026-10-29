# Estructura del proyecto 
# Estructura del Proyecto

loteria-ml-cr/
├── .github/
│   └── workflows/
│       └── deploy.yml          # (Opcional) Para despliegue automático
├── app/
│   ├── __init__.py
│   ├── main.py                 # API FastAPI (Backend)
│   ├── model.py                # Lógica para cargar el modelo y predecir
│   └── ui.py                   # Interfaz HTML con Streamlit o Renderizado
├── data/
│   ├── raw/                    # Aquí subes los PDFs originales
│   │   ├── Numeros_Favorecidos_2019_2025_Loteria_Nacional.pdf
│   │   ├── Numeros_Favorecidos_2019_2025_Nuevos_Tiempos.pdf
│   │   ├── Numeros_Favorecidos_Extraordianrios_2019_2025_Loteria_Nacional.pdf
│   │   └── Numeros_Favoreecidos_2019_2025_Loteria_Popular.pdf
│   ├── processed/              # CSV unificado generado por el script
│   │   └── dataset_unificado.csv
│   └── models/                 # Modelos entrenados (.pkl)
│       └── modelo_rf.pkl
├── notebooks/
│   └── exploracion_datos.ipynb # (Opcional) Para probar la lectura de PDFs
├── src/
│   ├── __init__.py
│   ├── data_ingestion.py       # Script que lee los PDFs y limpia los datos
│   ├── feature_engineering.py  # Crea variables (lags, frecuencias, etc.)
│   ├── train.py                # Entrenamiento del modelo
│   └── evaluate.py             # Evaluación del modelo
├── tests/
│   └── test_data.py            # Pruebas unitarias básicas
├── Dockerfile                  # Configuración de Docker
├── docker-compose.yml          # Para levantar el servicio fácilmente
├── requirements.txt            # Dependencias de Python
├── README.md                   # Documentación del proyecto
└── .gitignore                  # Archivos que no deben subirse a Git

# Crear venv con Python 3.10
py -3.10 -m venv venv

# Activar
.\venv\Scripts\Activate.ps1

# Verificar versión dentro del venv
python --version


 Ejecutar el entrenamiento
Con el venv activo ((venv) visible en el prompt):

powershell
python src\train.py