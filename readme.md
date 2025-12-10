# 🍽️ Predicción del exito de calificaciones de restaurantes

Este proyecto es un ejercicio integral de Ciencia de Datos que abarca desde la ingeniería y limpieza de datos hasta el análisis multivariado (PCA, VarClus) y la creación de un Dashboard interactivo para la visualización de KPIs de calidad en el servicio.

## 📂 Estructura del Proyecto

* **`Main.ipynb`**: Notebook principal (Jupyter). Ejecuta todo el flujo: ETL, Limpieza, EDA, PCA, Selección de Variables y generación de archivos finales.
* **`app.py`**: Aplicación web (Dashboard) construida con Streamlit para visualizar los resultados.
* **`fn/`**: Módulo con funciones auxiliares (Ingeniería, Limpieza, Análisis, Gráficos).
* **`datos/`**: Carpeta con los datasets crudos (CSV).
* **`resultados/`**: Carpeta generada automáticamente donde se guardan los datos procesados, las tablas entregables y las gráficas (PNG).

---

## 🚀 Guía de Instalación

Sigue estos pasos para configurar el proyecto en tu entorno local.

### 1. Clonar el repositorio
Abre tu terminal y descarga los archivos:

git clone https://github.com/JosueJimenezA/ExamenModulo1DCD.git



### 2. Instalar dependencias
Instala las librerías necesarias (Pandas, Streamlit, Plotly, Scikit-learn, etc.):

pip install -r requirements.txt

> **Nota:** Para el guardado de imágenes estáticas de Plotly, asegúrate de tener instalada la librería de motor gráfico:
> pip install -U kaleido

---

## 📊 Cómo ejecutar el Análisis (Notebook)

El procesamiento de datos se realiza en el Notebook. Es necesario ejecutarlo al menos una vez para generar los archivos limpios que usa el Dashboard.

1.  Inicia Jupyter en la terminal:
    jupyter notebook

2.  Abre el archivo **`Main.ipynb`**.
3.  Ejecuta todas las celdas (menú *Cell > Run All*).
    * Esto creará la carpeta `resultados/` con el dataset limpio y las imágenes del PCA.

---

## 📈 Cómo ejecutar el Dashboard (App)

Una vez generados los datos, puedes lanzar la aplicación interactiva de reporte:

1.  En tu terminal, ejecuta:
    streamlit run app.py

2.  El sistema abrirá automáticamente una pestaña en tu navegador (usualmente en `http://localhost:8501`) donde podrás interactuar con los filtros y KPIs.

---

## 📋 Requisitos Técnicos
* Python 3.8+
* Librerías principales: `pandas`, `numpy`, `matplotlib`, `seaborn`, `plotly`, `streamlit`, `scikit-learn`.