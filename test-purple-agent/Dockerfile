FROM python:3.13-slim

WORKDIR /app

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copy project files
COPY pyproject.toml .
COPY src/ src/

# Install dependencies
RUN uv sync --no-dev

# Expose port
EXPOSE 9010

# Run the agent
CMD ["uv", "run", "src/server.py", "--host", "0.0.0.0", "--port", "9010"]
