# Postgres 18 + pgvector + pg_vectorsearch, built on the benchmark host so
# -march=native / SIMD detection matches the CPU that runs the benchmark.
#
#   docker compose -f postgres/compose.yaml up -d --build
#   docker compose -f postgres/compose.yaml exec pg cat /usr/share/pg_vectorsearch.commit
#
# pg_vectorsearch is alpha: after the first build that works, set PRISM_REF to
# the commit printed above so every run uses the same code.
ARG PG_MAJOR=18
FROM postgres:${PG_MAJOR}
ARG PG_MAJOR
ARG PGVECTOR_REF=v0.8.7
ARG PRISM_REF=4899ffb57e7752a766b714621844d0abb4e6cad1

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential git ca-certificates pkg-config ninja-build python3-pip \
        postgresql-server-dev-${PG_MAJOR} libopenblas-dev \
    && pip3 install --break-system-packages meson \
    && rm -rf /var/lib/apt/lists/*

RUN git clone --depth 1 --branch ${PGVECTOR_REF} https://github.com/pgvector/pgvector.git /tmp/pgvector \
    && make -C /tmp/pgvector && make -C /tmp/pgvector install \
    && rm -rf /tmp/pgvector

RUN git clone https://github.com/timescale/pg_vectorsearch.git /tmp/pgvs \
    && cd /tmp/pgvs && git checkout ${PRISM_REF} \
    && git rev-parse HEAD > /usr/share/pg_vectorsearch.commit \
    && meson setup builddir --buildtype=release \
    && meson install -C builddir \
    && rm -rf /tmp/pgvs
