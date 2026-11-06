# Parity CLI. Build with --build-arg WITH_REFERENCE=0 for a smaller image that can only use
# a URL reference (--reference http://...); the default includes CPU PyTorch for hf:local.
FROM python:3.12-slim

ARG WITH_REFERENCE=1
ENV PIP_NO_CACHE_DIR=1 PYTHONUNBUFFERED=1 HF_HOME=/cache/hf

RUN pip install uv

WORKDIR /app
COPY pyproject.toml README.md LICENSE ./
COPY src ./src

# torch comes from the PyTorch CPU index in its own step: with --extra-index-url uv would take
# every package that index also carries (e.g. an old `packaging`) from there and fail.
RUN if [ "$WITH_REFERENCE" = "1" ]; then \
      uv pip install --system --index-url https://download.pytorch.org/whl/cpu "torch>=2.4" && \
      uv pip install --system ".[reference]"; \
    else \
      uv pip install --system .; \
    fi

VOLUME ["/cache", "/work"]
WORKDIR /work
ENTRYPOINT ["parity"]
CMD ["--help"]
