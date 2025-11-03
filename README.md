# Procesamiento de Datasets en Machine Learning

## Requisitos
- Python 3.10 o 3.11
- Entorno virtual con `requirements.txt`

## Estructura
- `data/raw/`: Datasets sin procesar (`titanic.csv`, `student-mat.csv`)
- `src/`: Módulos de utilidades, preprocesamiento y visualización
- `app.py`: Aplicación Streamlit con los tres ejercicios

## Ejecución
```bash
streamlit run app.py
```

## Coloca los archivos CSV en `data/raw/`. Ejecuta desde la raíz del proyecto:

```bash
python -m venv venv
```
```bash
source venv/bin/activate  # Linux/macOS
```
# o
```bash
venv\Scripts\activate     # Windows
```
```bash
pip install -r requirements.txt
```
```bash
streamlit run app.py
```