FROM python:3.12-slim

WORKDIR /app

# Cria venv
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Instala dependências
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copia código
COPY . .