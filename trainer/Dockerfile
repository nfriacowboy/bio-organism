FROM rocm/pytorch:latest

WORKDIR /app

# Instalar uv
RUN pip install uv --no-cache-dir

COPY pyproject.toml ./
RUN uv sync --no-cache

COPY src/ ./src/

CMD ["python", "-m", "src.training.trainer"]
