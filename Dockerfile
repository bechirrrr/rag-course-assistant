# Image de l'application (API FastAPI et interface Streamlit)
FROM python:3.12-slim

WORKDIR /app

# 1. Dépendances d'abord : cette couche reste en cache tant que requirements.txt ne change pas
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 2. Puis le code
COPY . .

EXPOSE 8000
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
