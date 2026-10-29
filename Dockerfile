FROM python:3.11-slim
WORKDIR /app
COPY pyproject.toml README.md ./
COPY src/ ./src/
RUN pip install --no-cache-dir .
COPY artifacts/ ./artifacts/
ENV MODEL_DIR=/app/artifacts
EXPOSE 8000
CMD ["uvicorn", "ml_lab.api:app", "--host", "0.0.0.0", "--port", "8000"]
