# --------------------------------------------------------------------------
# Stage 1: builder - resolve and install dependencies into a virtualenv
# --------------------------------------------------------------------------
FROM python:3.13-slim AS builder

# Install uv (fast Python package/dependency manager)
RUN pip install --no-cache-dir uv

WORKDIR /app

# Copy only the dependency manifests first so this layer is cached
# and only rebuilt when dependencies actually change, not on every
# source code edit.
COPY pyproject.toml uv.lock README.md ./

# Copy app code so uv can build the local package
COPY src/ ./src

# --frozen: fail if uv.lock is out of date instead of silently updating it
# --no-dev: skip dev-only dependencies, keep the runtime image lean
RUN uv sync --frozen --no-dev

# --------------------------------------------------------------------------
# Stage 2: runtime - slim image with just the venv and app code
# --------------------------------------------------------------------------
FROM python:3.13-slim AS runtime

WORKDIR /app

# Bring in the already-built virtual environment from the builder stage,
# so this stage never needs uv, pip, or a compiler toolchain at all.
COPY --from=builder /app/.venv /app/.venv

# Application code
COPY src/ ./src

# Put the venv's binaries first on PATH so `uvicorn`, `python`, etc.
# resolve to the versions installed in the virtualenv.
ENV PATH="/app/.venv/bin:$PATH"

# Lets MLFLOW_TRACKING_URI be overridden at `docker run` time; serve.py
# already falls back to http://127.0.0.1:5000 if this isn't set.


EXPOSE 8000

ENTRYPOINT ["uvicorn", "src.food11.serve:app", "--host", "0.0.0.0", "--port", "8000"]
