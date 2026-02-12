# ── Build stage ──────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential && \
    rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY pyproject.toml README.md ./
COPY grdl_rt/ ./grdl_rt/

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir .

# ── Runtime stage ────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1

# Non-root user
RUN useradd --create-home grdl
USER grdl
WORKDIR /home/grdl

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD test -f /tmp/grdl_rt_healthy

ENTRYPOINT ["grdl-rt"]
