FROM postgres:18 AS psql_bm25s-build

ARG PSQL_BM25S_REF=v0.4.14

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        clang \
        git \
        libicu-dev \
        make \
        pkg-config \
        postgresql-server-dev-18 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src

RUN git clone --depth 1 --branch "${PSQL_BM25S_REF}" https://github.com/Intelligent-Internet/psql_bm25s.git . \
    && make clean PG_CONFIG=/usr/bin/pg_config \
    && make PG_CONFIG=/usr/bin/pg_config \
    && make install DESTDIR=/tmp/psql_bm25s-stage PG_CONFIG=/usr/bin/pg_config

FROM postgres:18

LABEL org.opencontainers.image.title="TurboSearch Postgres smoke database"
LABEL org.opencontainers.image.description="PostgreSQL 18 with psql_bm25s and pgvector for TurboSearch local production-mode smoke tests"

RUN apt-get update \
    && apt-get install -y --no-install-recommends postgresql-18-pgvector \
    && rm -rf /var/lib/apt/lists/*

COPY --from=psql_bm25s-build /tmp/psql_bm25s-stage/ /
COPY --from=psql_bm25s-build /src/packaging/docker/postgres18/initdb/010-create-psql_bm25s.sh \
    /docker-entrypoint-initdb.d/010-create-psql_bm25s.sh

RUN chmod 0755 /docker-entrypoint-initdb.d/010-create-psql_bm25s.sh
