"""Where tables live, and how to talk to the storage behind them."""

from __future__ import annotations

import os
from dataclasses import dataclass, field

TABLES = ("images", "image_labels", "classes", "class_closure", "embeddings", "samples")


@dataclass(frozen=True)
class Lake:
    """A lake root: a local directory or an s3://bucket/prefix URI.

    Layout:
      <root>/delta/<table>/                    Delta tables
      <root>/staging/embeddings/<model_id>/    per-unit Parquet written by embed workers
    """

    root: str
    storage_options: dict = field(default_factory=dict)

    @classmethod
    def from_root(cls, root: str) -> Lake:
        root = root.rstrip("/")
        opts: dict = {}
        if root.startswith("s3://"):
            region = os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION")
            if region:
                opts["AWS_REGION"] = region
            # Every table here has exactly one writer at a time (ingest, finalize,
            # sample run serially; embed workers write plain Parquet, not Delta).
            # That makes delta-rs' unsafe rename acceptable and avoids a lock table.
            opts["AWS_S3_ALLOW_UNSAFE_RENAME"] = "true"
        return cls(root, opts)

    @property
    def is_s3(self) -> bool:
        return self.root.startswith("s3://")

    def table(self, name: str) -> str:
        if name not in TABLES:
            raise ValueError(f"unknown table {name!r}")
        return f"{self.root}/delta/{name}"

    def staging(self, model_id: str) -> str:
        return f"{self.root}/staging/embeddings/{model_id}"

    def filesystem(self):
        """(pyarrow filesystem, path) for the lake root, for plain-Parquet staging I/O."""
        import pyarrow.fs as pafs

        if self.is_s3:
            region = self.storage_options.get("AWS_REGION")
            fs = pafs.S3FileSystem(region=region) if region else pafs.S3FileSystem()
            return fs, self.root[len("s3://") :]
        os.makedirs(self.root, exist_ok=True)
        return pafs.LocalFileSystem(), os.path.abspath(self.root)

    def fs_path(self, uri: str) -> str:
        """Convert a lake URI to a path for `filesystem()`."""
        return uri[len("s3://") :] if uri.startswith("s3://") else os.path.abspath(uri)
