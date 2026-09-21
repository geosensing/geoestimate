"""Read annotated frames and run the stable inference API."""

import argparse
from pathlib import Path

import pandas as pd

from .designs import PointDesign
from .inference import estimate
from .types import InferenceResult

PARQUET_SUFFIXES = {".parquet", ".pq"}
CSV_SUFFIXES = {".csv", ".tsv"}
COMPRESSION_SUFFIXES = {".gz", ".bz2", ".zip", ".xz", ".zst"}


def read_frames(path: str | Path) -> pd.DataFrame:
    """Read an annotated-frame table based on its suffix.

    CSV and TSV files may use a compression suffix supported by pandas.
    Parquet files must end in ``.parquet`` or ``.pq`` because external
    compression obscures their format and is unnecessary.

    Args:
        path: Parquet, CSV, or TSV path.

    Returns:
        The table with the types stored by its format.

    Raises:
        ValueError: If the suffix does not name a supported format.
    """
    suffixes = [suffix.lower() for suffix in Path(path).suffixes]
    compressed = bool(suffixes and suffixes[-1] in COMPRESSION_SUFFIXES)
    format_suffixes = suffixes[:-1] if compressed else suffixes
    suffix = format_suffixes[-1] if format_suffixes else ""
    if suffix in PARQUET_SUFFIXES:
        if compressed:
            raise ValueError("externally compressed Parquet files are not supported")
        return pd.read_parquet(path)
    if suffix in CSV_SUFFIXES:
        return pd.read_csv(path, sep="\t" if suffix == ".tsv" else ",")
    expected = sorted(PARQUET_SUFFIXES | CSV_SUFFIXES)
    raise ValueError(
        f"cannot infer file format from {path!r}; expected one of {expected}"
    )


def estimate_from_file(
    path: str | Path,
    women_var: str = "n_women",
    people_var: str = "n_people",
    cluster_var: str | None = None,
    ci_level: float = 0.95,
    bootstrap: bool = False,
    bootstrap_reps: int = 2000,
    seed: int = 42,
    se_method: str | None = None,
    ci_method: str | None = None,
) -> InferenceResult:
    """Estimate proportions from an annotated-frame file.

    Args:
        path: Parquet, CSV, or TSV file.
        women_var: Column containing women counts.
        people_var: Column containing people counts.
        cluster_var: Optional column identifying independent clusters.
        ci_level: Confidence level strictly between zero and one.
        bootstrap: Compute pairs-bootstrap results over design units.
        bootstrap_reps: Number of bootstrap draws.
        seed: Random seed for bootstrap draws.
        se_method: Selected standard-error method.
        ci_method: Selected confidence-interval method.

    Returns:
        The result returned by :func:`geoinference.estimate`.
    """
    data = read_frames(path)
    return estimate(
        data,
        women_var,
        people_var,
        design=PointDesign(cluster_var=cluster_var),
        ci_level=ci_level,
        bootstrap=bootstrap,
        bootstrap_reps=bootstrap_reps,
        seed=seed,
        se_method=se_method,
        ci_method=ci_method,
    )


def main(argv: list[str] | None = None) -> None:
    """Run the command-line interface."""
    parser = argparse.ArgumentParser(prog="geoinference.io")
    subparsers = parser.add_subparsers(dest="command", required=True)
    command = subparsers.add_parser("estimate", help="estimate from a frame table")
    command.add_argument("path")
    command.add_argument("--women-var", default="n_women")
    command.add_argument("--people-var", default="n_people")
    command.add_argument("--cluster-var")
    command.add_argument("--ci-level", type=float, default=0.95)
    command.add_argument("--bootstrap", action="store_true")
    command.add_argument("--bootstrap-reps", type=int, default=2000)
    command.add_argument("--seed", type=int, default=42)
    command.add_argument("--se-method", choices=("naive", "cluster", "bootstrap"))
    command.add_argument("--ci-method", choices=("normal", "t", "bootstrap"))
    args = parser.parse_args(argv)

    result = estimate_from_file(
        args.path,
        women_var=args.women_var,
        people_var=args.people_var,
        cluster_var=args.cluster_var,
        ci_level=args.ci_level,
        bootstrap=args.bootstrap,
        bootstrap_reps=args.bootstrap_reps,
        seed=args.seed,
        se_method=args.se_method,
        ci_method=args.ci_method,
    )
    print(result.summary())


if __name__ == "__main__":
    main()
