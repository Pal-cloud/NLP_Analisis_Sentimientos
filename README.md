# 🚀 PROYECTO NLP: Detección de Mensajes de Odio en YouTube

> Proyecto de Data Science / AI Developer — Clasificación de texto con técnicas de NLP y Machine Learning para la detección automática de comentarios de odio.

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-orange?logo=scikit-learn&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red?logo=streamlit&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Containerized-blue?logo=docker&logoColor=white)
![MLflow](https://img.shields.io/badge/MLflow-Tracking-blueviolet?logo=mlflow&logoColor=white)
![Estado](https://img.shields.io/badge/Estado-En%20Desarrollo-yellow)

---

## 📜 Contexto del Proyecto

### 🔍 Problema a Resolver

YouTube enfrenta un crecimiento descontrolado de comentarios de odio que los equipos de moderación humana ya no pueden gestionar de forma escalable. Ante este reto, se nos encarga como consultora el diseño e implementación de una **solución automatizada basada en NLP y Machine Learning** capaz de detectar este tipo de mensajes en tiempo real.

El cliente prioriza una **solución funcional y desplegable** sobre la precisión perfecta del modelo: quieren poder actuar (eliminar mensajes, banear usuarios) de forma automática cuanto antes.

---

## 🎯 Objetivos del Proyecto

| # | Objetivo |
|---|----------|
| 1 | 🔎 Analizar y comprender el dataset de comentarios de YouTube |
| 2 | 🧹 Preprocesar y limpiar los datos de texto |
| 3 | 🔤 Aplicar técnicas clásicas de NLP (tokenización, stemming, lematización) |
| 4 | 🤖 Entrenar un modelo de clasificación binaria (odio / no odio) |
| 5 | 📈 Evaluar el rendimiento del modelo con métricas relevantes |
| 6 | ⚙️ Optimizar hiperparámetros del modelo seleccionado |

---

## 📦 Condiciones de Entrega

Para la fecha de entrega, los equipos deberán presentar:

✅ **Repositorio en GitHub** con el código fuente documentado.

✅ **Demo en vivo** mostrando el funcionamiento del modelo.

✅ **Presentación técnica**, explicando los objetivos, desarrollo y tecnologías utilizadas.

✅ **Tablero Kanban** con la gestión del proyecto (Trello, Jira, etc.).

---

## ⚙️ Stack Tecnológico

| Categoría | Herramientas |
|-----------|-------------|
| 🐍 Lenguaje | Python 3.11 |
| 🤖 ML / NLP | Scikit-learn, NLTK, SpaCy, Hugging Face Transformers |
| 📊 Datos | Pandas, NumPy |
| ⚙️ Optimización | Optuna |
| 🌐 Scraping | BeautifulSoup, Requests, Scrapy |
| 🖥️ App / API | Streamlit, FastAPI |
| 📦 Contenedores | Docker, Docker Compose |
| 🔬 Experimentos | MLflow |
| 🗂️ Versiones | Git / GitHub |
| 📋 Gestión | Trello / Jira |

---

## 🏆 Datos

> Dataset de comentarios de YouTube etiquetados como odio / no odio.

📥 [Descargar dataset — Youtube Comments](https://drive.google.com/file/d/1bG7fA273jIBgJfc6YS1vsKfr1qRiNUTU/view?usp=sharing)

---

## 🏆 Niveles de Entrega

### 🟢 **Nivel Esencial:**
✅ Un modelo de ML que reconozca los mensajes de odio.

✅ Controlar el overfitting, que la diferencia entre las métricas de training y las de test sea inferior a 5 puntos porcentuales.

✅ Una solución que productivice el modelo (una interfaz, API o lo que se os ocurra, que permita a un usuario consultar si un mensaje es o no de odio).

✅ Repositorio Git con ramas bien organizadas y commits limpios y descriptivos.

✅ Documentación del código y un README en GitHub.

### 🟡 **Nivel Medio:**
✅ Un modelo de ML con técnicas de ensemble que reconozca mensajes de odio.

✅ Una solución que permita reconocer los posibles mensajes de odio dado un enlace a un vídeo en concreto.

✅ Incluir tests unitarios.

✅ Optimización del modelo escogido con técnicas de ajuste de hiperparámetros (optuna, auto sklearn, pycaret, etc).

### 🟠 **Nivel Avanzado:**
✅ Un modelo que implemente redes neuronales y mejore significativamente los resultados frente a una solución de Machine Learning (RNN o LSTM).

✅ Una solución que permita introducir la url de un vídeo concreto y reconocer mensajes de odio haciendo seguimiento del video en tiempo real.

✅ Despliegue en un servidor accesible públicamente.

✅ Dockerizar la aplicación.

### 🔴 **Nivel Experto:**
✅ Utilizar un modelo basado en transformers.

✅ Guardar en base de datos los resultados de las predicciones.

✅ Trackear los experimentos realizados con MLFlow.

---

## 📊 Evaluación

Se considerarán los siguientes criterios técnicos:

| Criterio | Descripción |
|----------|-------------|
| 🧹 Preprocesamiento | Stemming, lematización, eliminación de stopwords |
| 🤖 Modelos | Clasificadores aplicados a texto |
| 🔢 Vectorización | TF-IDF, Bag of Words u otras técnicas clásicas |
| 🔍 Regex | Uso de expresiones regulares para limpieza y extracción |
| 🔄 Data Augmentation | Traducción, sinónimos u otras técnicas de aumento de datos |

📌 Más detalles en: [roadmap-mad-ai-p4.coderf5.es](https://roadmap-mad-ai-p4.coderf5.es/)

---

## � Cómo Ejecutar el Proyecto

### 🔧 Instalación local

```bash
# Clonar el repositorio
git clone https://github.com/tu-usuario/NLP_Analisis_Sentimientos.git
cd NLP_Analisis_Sentimientos

# Crear entorno virtual e instalar dependencias
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 🐳 Ejecución con Docker

```bash
docker-compose up --build
```

La aplicación estará disponible en `http://localhost:8501`

---

## �📁 Estructura del Proyecto

```
NLP_Analisis_Sentimientos/
├── data/
│   ├── raw/                  # Datos originales sin procesar
│   └── processed/            # Datos limpios y preprocesados
├── notebooks/
│   ├── 01_EDA.ipynb          # Análisis exploratorio de datos
│   ├── 02_preprocessing.ipynb# Preprocesamiento de texto
│   ├── 03_modeling.ipynb     # Entrenamiento de modelos
│   └── 04_evaluation.ipynb   # Evaluación y métricas
├── src/
│   ├── __init__.py
│   ├── preprocessing.py      # Funciones de limpieza y NLP
│   ├── vectorization.py      # TF-IDF, BoW, embeddings
│   ├── model.py              # Definición y entrenamiento del modelo
│   ├── predict.py            # Lógica de predicción
│   └── scraper.py            # Scraping de comentarios de YouTube
├── app/
│   ├── main.py               # App principal (Streamlit / FastAPI)
│   └── utils.py              # Utilidades para la app
├── models/
│   └── trained_model.pkl     # Modelo entrenado serializado
├── tests/
│   ├── test_preprocessing.py
│   ├── test_model.py
│   └── test_predict.py
├── mlruns/                   # Experimentos MLFlow
├── .gitignore
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```
