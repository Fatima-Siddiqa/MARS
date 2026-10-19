FROM python:3.12-slim
WORKDIR /app
COPY pyproject.toml ./
COPY macro_agent ./macro_agent
COPY config ./config
RUN pip install --no-cache-dir ".[postgres]"
ENV MACRO_SCHEDULER=1
EXPOSE 8000
CMD ["macro-agent", "serve", "--port", "8000"]
