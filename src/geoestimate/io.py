"""File input and command-line interface."""

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

from .sample import Sample

PARQUET_SUFFIXES = {".parquet", ".pq"}
CSV_SUFFIXES = {".csv", ".tsv"}
COMPRESSION_SUFFIXES = {".gz", ".bz2", ".zip", ".xz", ".zst"}


def _read_table(path: str | Path) -> pd.DataFrame:
    """Read a supported table based on its filename suffix."""
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


def _add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("path")
    parser.add_argument("--cluster")
    parser.add_argument(
        "--inference",
        choices=("design", "iid", "cluster", "bootstrap"),
        default="design",
    )
    parser.add_argument("--confidence-level", type=float, default=0.95)
    parser.add_argument("--bootstrap-reps", type=int, default=2000)
    parser.add_argument("--seed", type=int)


def _common_options(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "inference": args.inference,
        "confidence_level": args.confidence_level,
        "bootstrap_reps": args.bootstrap_reps,
        "seed": args.seed,
    }


def main(argv: list[str] | None = None) -> None:
    """Run the ``geoestimate`` command-line interface."""
    parser = argparse.ArgumentParser(prog="geoestimate")
    subparsers = parser.add_subparsers(dest="command", required=True)

    mean_parser = subparsers.add_parser("mean", help="estimate a population mean")
    _add_common_arguments(mean_parser)
    mean_parser.add_argument("--variable", required=True)

    total_parser = subparsers.add_parser("total", help="estimate a population total")
    _add_common_arguments(total_parser)
    total_parser.add_argument("--variable", required=True)
    total_parser.add_argument("--population-size", type=int, required=True)

    ratio_parser = subparsers.add_parser(
        "ratio", help="estimate a ratio of population totals"
    )
    _add_common_arguments(ratio_parser)
    ratio_parser.add_argument("--numerator", required=True)
    ratio_parser.add_argument("--denominator", required=True)

    mean_ratio_parser = subparsers.add_parser(
        "mean-of-ratios", help="estimate a mean of row-level ratios"
    )
    _add_common_arguments(mean_ratio_parser)
    mean_ratio_parser.add_argument("--numerator", required=True)
    mean_ratio_parser.add_argument("--denominator", required=True)

    args = parser.parse_args(argv)
    try:
        sample = Sample.from_file(args.path, cluster=args.cluster)
        common = _common_options(args)
        if args.command == "mean":
            result = sample.mean(args.variable, **common)
        elif args.command == "total":
            result = sample.total(
                args.variable, population_size=args.population_size, **common
            )
        elif args.command == "ratio":
            result = sample.ratio(args.numerator, args.denominator, **common)
        else:
            result = sample.mean_of_ratios(args.numerator, args.denominator, **common)
    except (TypeError, ValueError) as error:
        parser.error(str(error))
    print(result.summary())


if __name__ == "__main__":
    main()
