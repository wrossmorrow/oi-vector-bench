# oi-vector-bench: Open Images lake + PE-Core + CLIP embeddings

## Statement on AI use

As of committing, the human author/owner was a meat proxy. Claude (Opus 5.5 medium) did everything except run stuff and spend cloud dollars. 

## Introduction

A testbed for filtered and grouped vector search: pg_vectorsearch (PRISM) vs pgvector
HNSW in Postgres, with Pinot as a possible later comparison.

- **Phase 1 (`oilake`)** builds the data: Open Images metadata and image embeddings as
  Delta tables on S3, with deterministic, nested benchmark samples (1M ⊂ ~1.9M).
- **Phase 2 (`oibench`)** exports a sample, generates a workload with exact answers,
  loads Postgres, sweeps index parameters and reports recall and latency.

Scope: the 1,910,098 images whose pixels are in the public CVDF mirror
(`s3://open-images-dataset`). Labels and metadata are ingested for all 9.18M images, so a
later Flickr-crawl tier only needs new embeddings.

## Embedding models

| `model_id` | weights | dim | notes |
|---|---|---|---|
| `pe-core-l14-336` | `hf-hub:timm/PE-Core-L-14-336` | 1024 | the better model |
| `clip-l14-336` | OpenAI ViT-L/14@336 via `open_clip` | 768 | older and weaker, but under pg_vectorsearch's fastscan limit (below) |

Both cover the same images (decode errors don't depend on the model), so one set of
samples serves both and their benchmarks are directly comparable.

## Tables (`<lake>/delta/…`)

| table | rows | notes |
|---|---|---|
| `images` | 9.18M | `in_s3_mirror`, `s3_key`, `sample_key`, `rotation`, author/license/title, URLs |
| `image_labels` | ~100M+ | human + machine image-level labels, partitioned by `kind`. Human negatives (confidence 0) kept |
| `classes` | ~20.6k | MID, display name, dense `label_id` (for `int[]` in Postgres), boxable/trainable flags |
| `class_closure` | | (ancestor, descendant, depth) over the 600 boxable classes; Subcategory only, Parts ignored |
| `embeddings` | 1.91M per model | `image_id`, float32 L2-normalized `embedding`, `error`, partitioned by `model_id` |
| `samples` | | named nested samples: `queries`, `mirror_1m`, `mirror_all` |

Readable from DuckDB (`delta_scan('s3://…/delta/images')`), Polars, Spark, etc.

## Where things run

| piece | where | why |
|---|---|---|
| mirror images | `s3://open-images-dataset`, us-east-1 | public, read unsigned; egress is billed to the bucket owner, not you |
| lake | your bucket in us-east-1 | next to the mirror and the Postgres box |
| ingest | a cloud box | ~10 GB of CSVs down from Google, multi-GB Parquet files up to S3; a home uplink timed out on S3 multipart uploads |
| embedding | Lambda GPU instance | AWS GPU capacity was unavailable; this job only needs S3 over the internet |
| Postgres benchmark | EC2 x86 with AVX-512 in us-east-1 | CPU only |

## Setup (one time)

1. **Lake bucket** in us-east-1, Block Public Access on. Lake root: `s3://<bucket>/oi`.
   A lifecycle rule that aborts incomplete multipart uploads after a day is good hygiene.

2. **IAM user `oilake-writer`** with an access key (Lambda can't assume an instance role),
   and this inline policy. Delete is needed because delta-rs' rename on S3 is copy + delete.
   No permissions are needed for the mirror.

   ```json
   {
     "Version": "2012-10-17",
     "Statement": [
       {"Effect": "Allow", "Action": "s3:ListBucket",
        "Resource": "arn:aws:s3:::<bucket>"},
       {"Effect": "Allow", "Action": ["s3:GetObject", "s3:PutObject", "s3:DeleteObject"],
        "Resource": "arn:aws:s3:::<bucket>/*"}
     ]
   }
   ```

   Deactivate the key when the work is done. On EC2, an instance profile is the better
   practice.

3. **Python**: pin 3.12 with `uv python pin 3.12`. uv otherwise picks the newest
   interpreter it finds, and torch doesn't publish wheels for every new Python release
   right away (3.15 had none at the time).

## Phase 1: data (`oilake`)

```bash
uv sync --extra embed          # a plain `uv sync` removes the extras: always pass --extra
export AWS_REGION=us-east-1
export AWS_ACCESS_KEY_ID=...  AWS_SECRET_ACCESS_KEY=...   # oilake-writer
LAKE=s3://<bucket>/oi

uv run oilake ingest   --lake $LAKE                     # metadata -> Delta
uv run oilake plan     --lake $LAKE                     # mirror images in sample_key order, 20k/unit
uv run oilake embed    --lake $LAKE --max-units 1       # smoke test: check img/s and errors
uv run oilake embed    --lake $LAKE --num-workers 28    # the rest, in tmux/screen; rerun to resume
uv run oilake finalize --lake $LAKE

# second model: same commands with --model clip-l14-336 (plan, embed, finalize)
```

`ingest` reruns individual steps with `--only classes images labels`; downloads are cached
in `~/.cache/oilake`.

**Throughput.** One A100 (40 GB, 30 vCPUs) ran both ViT-L/14@336 models at about
190 img/s, about 2h 50m for 1.91M images. It was GPU-bound (100% util, 350 W); fetch and
decode had headroom. 190 img/s is well below what an A100 should do for this model in
fp16, so for a bigger tier, try fp16/bf16 weights instead of autocast, then
`torch.compile`, and check which attention kernel runs.
`scripts/preflight.py` measures fetch, decode and GPU separately before a run.

**Tuning loaders.** Each DataLoader process fetches its batch with `OILAKE_FETCH_THREADS`
concurrent GETs (default 16), then decodes.
- GPU idle, CPU busy: decode-bound. Set `--num-workers` to about the vCPU count.
- GPU idle, CPU idle: fetch-bound. Raise `OILAKE_FETCH_THREADS` (32–64 from outside AWS).
- GPU busy: done.

Units follow `sample_key` order, so stopping after the first ~50 units still gives a
complete `mirror_1m` (finalize with `--allow-missing`). Several machines can share the
work with `--worker 0/2` and `--worker 1/2`. `--lake` also takes a local directory, and
`--local-images DIR` reads JPEGs from an `aws s3 sync` copy.

### Samples

Hold out queries first, so no query image is ever in an indexed set:

```bash
uv run oilake sample --lake $LAKE --name queries    --n 5000 --tail
uv run oilake sample --lake $LAKE --name mirror_1m  --n 1000000 --exclude queries
uv run oilake sample --lake $LAKE --name mirror_all --exclude queries     # 1,905,098
```

### Decisions worth a second look

- **Rotation is applied** (`Rotation` = degrees CCW to the uploader's intent) before
  embedding, and recorded per row in `rotation_applied`. Pass `--no-rotation` to skip.
- **Some mirror images are full size**, despite the mirror's documented 1024 px cap
  (Pillow warned about a ~100 MP image). They embed fine; `Image.draft()` would make
  them cheap to decode if decode ever becomes the bottleneck.
- **Loader.** PE-Core is loaded through `open_clip` (`hf-hub:timm/PE-Core-L-14-336`). If
  you query with vectors from Meta's reference loader, compare the two on a few images
  first.
- **float32 storage**: Delta has no float16 type (or unsigned ints: `sample_key` is a
  63-bit signed value for that reason).

## Phase 2: the benchmark (`oibench`)

One CPU box in us-east-1: r7i.4xlarge (16 vCPU, 128 GiB, AVX-512), 300 GB gp3,
Docker with compose and buildx. The Postgres client and server share the box, so
latencies are comparable across methods, not absolute.

**Docker on Amazon Linux 2023**: `dnf install docker` comes without buildx, which
`compose build` requires. Install the buildx binary from its GitHub releases into
`~/.docker/cli-plugins/` (this CLI doesn't search `/usr/local/lib/docker/cli-plugins`).

```bash
uv sync --extra bench
docker compose -f postgres/compose.yaml up -d --build
docker compose -f postgres/compose.yaml exec pg cat /usr/share/pg_vectorsearch.commit   # pin as PRISM_REF

# optional: test the Postgres path end to end without real data (a few minutes)
uv run oibench synth --out data/syn
uv run oibench workload --data data/syn --out work/syn --n-per-cell 20
uv run oibench load --data data/syn --prefix syn
uv run oibench run --data data/syn --workload work/syn --sweep sweep.example.toml --prefix syn --out results/syn

# the real thing (PE-Core: data/m1, prefix m1; CLIP: --model clip-l14-336, data/c1, prefix c1)
uv run oibench export   --lake $LAKE --sample mirror_1m --out data/m1
uv run oibench workload --data data/m1 --out work/m1
uv run oibench load     --data data/m1 --prefix m1
uv run oibench run      --data data/m1 --workload work/m1 --sweep sweep.example.toml \
                        --out results/m1-a --methods exact hnsw prism
uv run oibench run      --data data/m1 --workload work/m1 --sweep sweep.example.toml \
                        --out results/m1-a --methods hnsw_iter      # slow; same --out appends
uv run oibench run      --data data/m1 --workload work/m1 --sweep sweep.example.toml \
                        --out results/m1-a --methods routed_prism routed_hnsw
uv run oibench plans    --workload work/m1 --results results/m1-a | tee results/m1-a/plans.txt
uv run oibench report   --workload work/m1 --results results/m1-a   # report.md, summary.csv
```

Don't run a `workload` or `load` while a `run` is going on the same box: they compete
for CPU and cache and distort the run's latencies.

`run` **resumes**: rerun the same command into the same `--out` and it skips configs that
already finished and drops the partial rows of an interrupted one. Check the
"N configs to run" line before walking away; a stale sweep file on the box shows up there.

### What gets measured

- **Workload** (`work/<name>/workload.jsonl`): held-out query images, three kinds.
  - `filter`: top-k among items with one label. Selectivity bins are half-decades from 1
    down to 10⁻⁵ (only labels with ≥ k items). `ftype` is how the label relates to the
    query image: `pos` (it has the label), `rand` (unrelated), `neg` (human-verified
    absent). Each query also records `lift`, the share of the query's unfiltered top-1000
    neighbors that carry the label divided by its overall selectivity. Lift below 1 is
    where post-filtering scans run out of candidates.
  - `group_label`: top-3 per label for G = 10 or 100 labels from one bin.
  - `diverse`: top-k with at most 1 or 2 images per photographer.
- **Ground truth** (`truth.jsonl`): exact answers by brute force over the same
  fp16-rounded vectors Postgres stores, using pgvector's cosine distance.
- **Tables**: `<prefix>_hnsw` and `<prefix>_prism` hold identical rows
  (`labels int4[]` with GIN, `author_id`, `embedding halfvec STORAGE PLAIN`), one vector
  index each, so no planner hints are needed.
- **Sweep** (`sweep.example.toml`): exact baseline; HNSW over `ef_search`; HNSW with
  iterative scan over order × `max_scan_tuples`; PRISM over `nprobe`. Each runs in
  `forced` mode (vector index only) and `planner` mode (Postgres may pick GIN + sort).
  Grouped queries run as `lateral` (one filtered subquery per group) and `overfetch:M`
  (top-M, then group in SQL). A method can narrow the query kinds it runs with `kinds`.
- **Routed** (`routed_prism`, `routed_hnsw`; filter queries only): probe-then-route.
  An unfiltered top-`probe` scan through the vector index counts how many of the query's
  nearest neighbors pass the filter. That count estimates the filter's lift around this
  query, which the planner can't see. If fewer than `factor` × k pass, the query runs on
  the exact plan (`enable_indexscan = off`, so GIN + sort); otherwise it runs on the
  index. Latency includes the probe. The report's `routed_exact` column is the share
  sent to the exact plan.
- **Report**: recall mean and 10th percentile, short-result rate, latency p50/p95 per
  config × cell. `explain.jsonl` keeps `EXPLAIN ANALYZE` plans; `oibench plans` summarizes
  which plan each config actually used.

### Measurement pitfalls (all handled; listed so they stay handled)

- **Cached plans.** psycopg prepares a statement after 5 executions, and Postgres keeps a
  prepared statement's plan even after `enable_*` or index GUCs change. Every config then
  silently reuses the first config's plan. `run` disables server-side prepares and runs
  `DEALLOCATE ALL` between configs, so planning time is included in latency for every
  method.
- **Forcing the index.** Turning off seq and bitmap scans isn't enough: the planner can
  walk the primary-key btree, filter and sort, which is an exact plan. Forced mode also
  turns off sorts, leaving the ordered vector index scan as the only plan.
- **Verify with `oibench plans`**: forced rows must show only `Index Scan:<prefix>_*_vec`.

### pg_vectorsearch fastscan and dimension

PRISM's packed SIMD posting layout (`fastscan = auto`) only turns on if a 32-vector group
fits on a cluster's first posting page. That page also holds a float32 copy of the
centroid, so the cutoff is about **945 dimensions** at 8 KB pages: 580 + 4·dim bytes per
group ≤ 8,144 − 4·dim usable bytes. At pinned commit `4899ffb`:

- PE-Core (1024-d): `fastscan: off`; PRISM scans the array-of-structs layout.
- CLIP (768-d): fastscan on.

The tuning guide's "always fits up to ~1900 dims" is the overflow-page limit, which doesn't
bind. Check `fastscan` in `data/<name>/load-<prefix>.json` after every load.

### Things to expect

- HNSW without iterative scan returns at most `ef_search` rows before filtering, so
  selective or anti-correlated filters, k=100 at ef_search=40, and `overfetch:M` with
  M > ef_search come back short.
- Iterative scan stops at `hnsw.max_scan_tuples` (default 20,000) or its memory budget
  (`work_mem` × `hnsw.scan_mem_multiplier`), whichever comes first.
- PRISM fetches about k / selectivity candidates and filters afterwards, so filters whose
  items sit far from the query (`neg`, low lift) come back short regardless of `nprobe`.
  Its top-k heap is capped by `work_mem` (256MB in `compose.yaml`).
- `planner` mode often rescues both by choosing the exact GIN plan, but not always: on
  synthetic data it picked HNSW at 1–3% selectivity and returned wrong answers.

## Tests

```bash
uv run --extra embed --extra bench python -m unittest discover -s tests
```

pyarrow/Delta and Pillow tests skip when those packages aren't installed.
