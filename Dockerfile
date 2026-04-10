# ── Build stage ───────────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

# System deps needed to compile some wheels (e.g. lxml, grpc)
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --upgrade pip \
 && pip install --no-cache-dir --prefix=/install -r requirements.txt


# ── Runtime stage ─────────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

LABEL org.opencontainers.image.title="GraphBuilder" \
      org.opencontainers.image.description="Knowledge graph builder with LLM-powered extraction" \
      org.opencontainers.image.source="https://github.com/VincentPit/GraphBuilder"

# Non-root user for security
RUN groupadd --gid 1001 graphbuilder \
 && useradd  --uid 1001 --gid graphbuilder --no-create-home graphbuilder

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application source
COPY --chown=graphbuilder:graphbuilder src/ ./src/
COPY --chown=graphbuilder:graphbuilder setup.py pyproject.toml README.md ./

# Install the package itself (no deps — already in /usr/local)
RUN pip install --no-cache-dir --no-deps -e .

# Create a writable logs directory
RUN mkdir -p /app/logs && chown graphbuilder:graphbuilder /app/logs

USER graphbuilder

# Default command: show CLI help. Override via docker-compose or `docker run`.
ENTRYPOINT ["python", "-m", "graphbuilder"]
CMD ["--help"]
