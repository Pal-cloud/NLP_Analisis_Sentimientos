# 🚀 PROYECTO NLP: Analisis de Sentimientos

![NLP](https://github.com/user-attachments/assets/aa8ae666-25aa-416c-8be4-58e1af7a00c6)

## 📜 Briefing: PROYECTO DATA SCIENTIST/AI DEVELOPER

### 🔍 Planteamiento

YouTube lleva un tiempo preocupado por el aumento de los mensajes de odio entre los
comentarios de sus vídeos y ha llegado a un punto donde un equipo de moderadores no da
a basto y aumentar ese equipo sería prohibitivamente caro, además de que no se hace
posible escalar el equipo al ritmo al que crece la plataforma y aumenta el volumen de estos
mensajes.
Por ese motivo han decidido subcontratar a una consultora, donde trabajáis, en busca de
una solución para poder detectar este tipo de mensajes de forma automática y para así
poder eliminarlos, banear al usuario o tomar las acciones necesarias.
Han hecho hincapié en que es importante la implementación de la solución que encontréis,
quieren una solución práctica por encima de una herramienta precisa.

---

## 🎯 Objetivos del Proyecto

* **Analizar un dataset de texto.**
* **Preprocesar los datos.**
* **Implementar técnicas clásicas de procesamiento del lenguaje natural.**
* **Entrenar un modelo de Machine Learning de clasificación.**
* **Evaluación de resultados.**
* **Tuneo de hiperparámetros.**

---

## 📦 Condiciones de Entrega

Para la fecha de entrega, los equipos deberán presentar:

✅ **Repositorio en GitHub** con el código fuente documentado.

✅ **Demo en vivo** mostrando el funcionamiento del modelo.

✅ **Presentación técnica**, explicando los objetivos, desarrollo y tecnologías utilizadas.

✅ **Tablero Kanban** con la gestión del proyecto (Trello, Jira, etc.).

---

## ⚙️ Tecnologías Recomendadas

- **Control de versiones:** Git / GitHub
- **Entorno de ejecución:** Docker
- **Lenguaje principal:** Python
- **Librerías útiles:** Scikit-learn, Pandas, Spacy, NLTK, Optuna, Huggingface, BeautifulSoup, requests, Scrapy
- **Visualización de Datos:** Streamlit
- **Gestión del proyecto:** Trello, Jira, Github

---

## 🏆 Datos

[Youtube Comments](https://drive.google.com/file/d/1bG7fA273jIBgJfc6YS1vsKfr1qRiNUTU/view?usp=sharing)

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

Se considerarán los siguientes criterios:

✅ Preprocesamiento (stemming, lematización, stopwords)
✅ Modelos de clasificación aplicados a texto
✅ Emplea técnicas de vectorización clásicas
✅ Uso de Expresiones regulares
✅ Implementa técnicas de data augmentation en texto (traducción, reemplazo por sinónimos, etc.)

Más detalles en: [roadmap-mad-ai-p4.coderf5.es](https://roadmap-mad-ai-p4.coderf5.es/)

---

## 📁 Estructura del Proyecto

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
