# Imagem base do Python
FROM python:3.11-slim

# Evita prompts interativos e melhora logs
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Diretório de trabalho dentro do container
WORKDIR /app

# Copia requirements e instala dependências
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copia todo o código do projeto para dentro do container
COPY . .

# Gera os artefatos (df_results.csv, df_analise_produtos.csv, xgb_model.json)
RUN mkdir -p artifacts && python train_model.py

# Expõe a porta do Dash (por padrão 8050)
EXPOSE 8050

# Comando para iniciar o dashboard
CMD ["python", "app.py"]
