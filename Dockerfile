# rummage with the pinned llama.cpp CPU build. The model files are fetched into the data volume by
# `rummage setup` on first start (checked by size and SHA-256), so the image stays small.
FROM python:3.12-slim-bookworm

LABEL org.opencontainers.image.licenses="AGPL-3.0-or-later" \
      org.opencontainers.image.source="https://github.com/soffits/rummage" \
      org.opencontainers.image.description="Local search by meaning and exact words over your files"

ARG LLAMA_TAG=b12387
ARG LLAMA_SHA256=9c58ebcd5a704e879cfadf2bbcb1fb1ec8ad4a5bec761fc58b3515864dd5e37a

RUN apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates curl libgomp1 libssl3 \
 && rm -rf /var/lib/apt/lists/* \
 && curl -fsSL -o /tmp/llama.tgz \
      "https://github.com/ggml-org/llama.cpp/releases/download/${LLAMA_TAG}/llama-${LLAMA_TAG}-bin-ubuntu-x64.tar.gz" \
 && echo "${LLAMA_SHA256}  /tmp/llama.tgz" | sha256sum -c - \
 && mkdir -p /opt/llama && tar -xzf /tmp/llama.tgz -C /opt/llama --strip-components=1 && rm /tmp/llama.tgz \
 && useradd --system --create-home --uid 10001 rummage

COPY pyproject.toml README.md LICENSE NOTICE /src/
COPY src /src/src
RUN pip install --no-cache-dir "/src[archives,seafile]" && rm -rf /src

ENV RUMMAGE_DATA_DIR=/data RUMMAGE_LLAMA_SERVER=/opt/llama/llama-server
USER rummage
VOLUME /data
EXPOSE 8765
HEALTHCHECK --interval=60s --timeout=5s CMD curl -fs http://127.0.0.1:8765/health || exit 1
ENTRYPOINT ["sh", "-c", "rummage setup && exec rummage serve --host 0.0.0.0 --port 8765 \"$@\"", "--"]
CMD ["--reindex-minutes", "60"]
