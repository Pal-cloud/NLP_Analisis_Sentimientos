"""
utils.py - Funciones de preprocesamiento para la app Streamlit.
Deben ser idénticas a las usadas durante el entrenamiento del modelo.
"""

import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Descargar recursos necesarios (solo la primera vez)
for resource in ["stopwords", "wordnet", "omw-1.4"]:
    try:
        nltk.data.find(f"corpora/{resource}")
    except LookupError:
        nltk.download(resource, quiet=True)

STOP_WORDS = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()


def clean_text(text: str) -> str:
    """
    Limpia y normaliza el texto de entrada.
    Pipeline: minúsculas → sin URLs → sin menciones → sin HTML
              → solo letras → sin stopwords → lematización.
    """
    if not isinstance(text, str):
        return ""

    # Minúsculas
    text = text.lower()
    # Eliminar URLs
    text = re.sub(r"http\S+|www\.\S+", "", text)
    # Eliminar menciones @usuario
    text = re.sub(r"@\w+", "", text)
    # Eliminar etiquetas HTML
    text = re.sub(r"<.*?>", "", text)
    # Eliminar caracteres especiales y números, mantener solo letras y espacios
    text = re.sub(r"[^a-z\s]", "", text)
    # Eliminar espacios múltiples
    text = re.sub(r"\s+", " ", text).strip()

    # Tokenización simple, eliminar stopwords y lematizar
    tokens = [
        lemmatizer.lemmatize(word)
        for word in text.split()
        if word not in STOP_WORDS and len(word) > 2
    ]

    return " ".join(tokens)
