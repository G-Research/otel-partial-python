FROM python:3.13-slim

WORKDIR /app

COPY pyproject.toml README.md ./

COPY src/ src/

RUN pip install --upgrade pip setuptools wheel

RUN pip install .

COPY tests/ tests/

RUN pip install pytest coverage

CMD ["pytest", "--tb=short", "tests"]