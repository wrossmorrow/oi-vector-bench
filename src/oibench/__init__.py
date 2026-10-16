"""Filtered / grouped vector-search benchmark over an oilake sample.

Pipeline (all on the benchmark box):

  export    lake (Delta on S3) -> local numpy dataset        oibench.export
  workload  dataset -> queries + exact ground truth           oibench.workload, oibench.truth
  load      dataset -> Postgres tables + HNSW / PRISM         oibench.pg
  run       workload x parameter sweep -> raw results         oibench.run
  report    results + ground truth -> recall / latency tables oibench.report

Everything after `export` reads only the numpy dataset, so it needs no
pyarrow and no S3.
"""
