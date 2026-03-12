FROM rocm/pytorch:latest

WORKDIR /app

# Instalar uv
RUN pip install uv --no-cache-dir

COPY pyproject.toml ./
RUN uv sync --no-cache

COPY src/ ./src/

# Adicionar src/ ao PYTHONPATH para imports absolutos (ex: from world.vision import ...)
ENV PYTHONPATH=/app/src

CMD ["python", "-m", "training.trainer"]
