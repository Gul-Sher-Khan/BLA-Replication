import argparse
import hashlib
import logging
import os
import random
import sys
from collections import Counter
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LACE_ROOT = os.path.abspath(os.path.join(BASE_DIR, "..", "LACE"))
if LACE_ROOT not in sys.path:
    sys.path.append(LACE_ROOT)

from lace import lace1, lace2_simulator  # type: ignore[reportMissingImports]


logger = logging.getLogger("anon-lace-kmeans")


INPUT_CSV_PATH = "input.csv"
OUTPUT_CSV_PATH = "output.csv"

ID_FIELD = "commit_id"
DATE_FIELD = "author_date"
SENSITIVE = ["la", "ld"]
QIDS = ["nf", "nd", "ns", "ent", "ndev", "nuc", "age", "aexp", "asexp", "arexp"]


def setup_logging(log_level: str, log_file: str | None) -> None:
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    logger.handlers.clear()
    logger.propagate = False

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if log_file:
        log_dir = os.path.dirname(log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)


def _hash32(value: str) -> int:
    return int(hashlib.sha256(value.encode("utf-8")).hexdigest()[:8], 16)


def preprocess_df(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    out = df.copy()
    if date_col in out.columns:
        out[date_col] = pd.to_datetime(out[date_col], unit="s", errors="coerce")
        out = out.sort_values(by=date_col).reset_index(drop=True)
    return out


def add_derived(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    out = df.copy()
    derived_only_cols: List[str] = []
    la = pd.to_numeric(out["la"], errors="coerce").fillna(0)
    ld = pd.to_numeric(out["ld"], errors="coerce").fillna(0)
    out["churn"] = (la + ld).astype(int)

    if "bugcount" in out.columns:
        if "buggy" not in out.columns:
            derived_only_cols.append("buggy")
        out["buggy"] = (
            pd.to_numeric(out["bugcount"], errors="coerce").fillna(0) > 0
        ).astype(int)

    return out, derived_only_cols


def ensure_objective_available(df: pd.DataFrame, objective: str) -> None:
    if objective in df.columns:
        return
    if objective == "buggy":
        raise KeyError(
            "Objective 'buggy' is unavailable. Input must contain 'bugcount' or 'buggy'."
        )
    raise KeyError(f"Missing objective column: {objective}")


def _objective_has_diversity(
    cluster_df: pd.DataFrame, objective: str, objective_binary: bool
) -> bool:
    values = pd.to_numeric(cluster_df[objective], errors="coerce").fillna(0)
    if objective_binary:
        return int((values > 0).nunique()) > 1
    return int(values.nunique()) > 1


def _compute_n_clusters(
    n_rows: int, requested: int, min_cluster_size: int, max_clusters: int = 40
) -> int:
    if n_rows < 2:
        return 1

    if requested and requested > 1:
        return max(2, min(requested, n_rows))

    auto_clusters = n_rows // max(1, min_cluster_size)
    auto_clusters = max(2, min(max_clusters, auto_clusters))
    return min(auto_clusters, n_rows)


def assign_kmeans_clusters(
    df: pd.DataFrame,
    qids: List[str],
    n_clusters: int,
    seed: int,
) -> pd.Series:
    features = df[qids].apply(pd.to_numeric, errors="coerce")
    imputer = SimpleImputer(strategy="median")
    matrix = imputer.fit_transform(features)

    if n_clusters <= 1:
        labels = np.zeros(len(df), dtype=int)
    else:
        model = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
        labels = model.fit_predict(matrix)

    return pd.Series(labels, index=df.index, name="cluster_key")


def _apply_lace_to_cluster(
    cluster_df: pd.DataFrame,
    cluster_id: str,
    method: str,
    objective: str,
    objective_binary: bool,
    cliff: float,
    alpha: float,
    beta: float,
    morph_alpha: float,
    morph_beta: float,
    holders: int,
) -> Tuple[pd.DataFrame, str]:
    if len(cluster_df) < 2:
        logger.info(
            "Skipping LACE for cluster %s: only %d row.", cluster_id, len(cluster_df)
        )
        return cluster_df.copy(), "skipped_small"

    if not _objective_has_diversity(cluster_df, objective, objective_binary):
        logger.info(
            "Skipping LACE for cluster %s: objective '%s' has no class diversity.",
            cluster_id,
            objective,
        )
        return cluster_df.copy(), "skipped_no_diversity"

    work_df = cluster_df.copy()
    work_df["__row_id__"] = cluster_df.index.astype(int)
    attribute_names = list(work_df.columns)
    independent_attrs = [
        col for col in (SENSITIVE + QIDS) if col in work_df.columns and col != objective
    ]

    if not independent_attrs:
        logger.info(
            "Skipping LACE for cluster %s: no independent attributes available.",
            cluster_id,
        )
        return cluster_df.copy(), "skipped_no_independent_attrs"

    data_matrix = work_df[attribute_names].values.tolist()

    try:
        random.seed(_hash32(cluster_id))
        if method == "lace1":
            result_rows = lace1(
                attribute_names=attribute_names,
                data_matrix=data_matrix,
                independent_attrs=independent_attrs,
                objective_attr=objective,
                objective_as_binary=objective_binary,
                cliff_percentage=cliff,
                alpha=alpha,
                beta=beta,
            )
        else:
            result_rows = lace2_simulator(
                attribute_names=attribute_names,
                data_matrix=data_matrix,
                independent_attrs=independent_attrs,
                objective_attr=objective,
                objective_as_binary=objective_binary,
                cliff_percentage=cliff,
                morph_alpha=morph_alpha,
                morph_beta=morph_beta,
                number_of_holder=holders,
            )
            if result_rows:
                result_rows = list(result_rows)
                if result_rows and list(result_rows[0]) == attribute_names:
                    result_rows = result_rows[1:]

        result_df = pd.DataFrame(result_rows, columns=attribute_names)
        if result_df.empty:
            logger.warning(
                "LACE produced no rows for cluster %s; keeping original cluster.",
                cluster_id,
            )
            return cluster_df.copy(), "fallback_empty"

        result_df["__row_id__"] = pd.to_numeric(
            result_df["__row_id__"], errors="coerce"
        ).astype(int)
        result_df = result_df.sort_values("__row_id__").set_index("__row_id__")
        result_df.index.name = None
        result_df = result_df.reindex(sorted(result_df.index)).copy()
        return result_df[cluster_df.columns], "applied"
    except Exception as exc:
        logger.warning(
            "LACE failed for cluster %s; keeping original rows. Error: %s",
            cluster_id,
            exc,
        )
        return cluster_df.copy(), "fallback_error"


def anonymize_with_kmeans_lace(
    df: pd.DataFrame,
    method: str,
    objective: str,
    objective_binary: bool,
    cliff: float,
    alpha: float,
    beta: float,
    morph_alpha: float,
    morph_beta: float,
    holders: int,
    n_clusters: int,
    min_cluster_size: int,
    drop_columns: List[str],
) -> pd.DataFrame:
    ensure_objective_available(df, objective)

    for qid in QIDS:
        if qid not in df.columns:
            raise KeyError(f"Missing QID column: {qid}")

    resolved_clusters = _compute_n_clusters(
        n_rows=len(df), requested=n_clusters, min_cluster_size=min_cluster_size
    )
    logger.info("Assigning k-means clusters with k=%d ...", resolved_clusters)
    cluster_keys = assign_kmeans_clusters(
        df=df,
        qids=QIDS,
        n_clusters=resolved_clusters,
        seed=_hash32(f"kmeans|{len(df)}|{objective}"),
    )
    logger.info("Assigned rows to %d unique clusters.", len(cluster_keys.unique()))

    grouped: Dict[int, List[int]] = {}
    for index, key in cluster_keys.items():
        grouped.setdefault(int(key), []).append(index)

    lace_parts: List[pd.DataFrame] = []
    cluster_status_counts: Counter[str] = Counter()
    total_input_rows = len(df)
    total_output_rows = 0
    total_pruned_rows = 0

    for key, indexes in grouped.items():
        cluster_id = f"kmeans|{key}"
        cluster_df = df.loc[indexes].copy()
        logger.info(
            "Applying %s to cluster %s (rows=%d)...",
            method,
            cluster_id,
            len(cluster_df),
        )
        result_df, status = _apply_lace_to_cluster(
            cluster_df=cluster_df,
            cluster_id=cluster_id,
            method=method,
            objective=objective,
            objective_binary=objective_binary,
            cliff=cliff,
            alpha=alpha,
            beta=beta,
            morph_alpha=morph_alpha,
            morph_beta=morph_beta,
            holders=holders,
        )
        cluster_status_counts[status] += 1
        lace_parts.append(result_df)
        total_output_rows += len(result_df)
        total_pruned_rows += max(len(cluster_df) - len(result_df), 0)

    out_df = pd.concat(lace_parts).sort_index().reset_index(drop=True)
    out_df["la"] = (
        pd.to_numeric(out_df["la"], errors="coerce")
        .fillna(0)
        .round()
        .clip(lower=0)
        .astype(int)
    )
    out_df["ld"] = (
        pd.to_numeric(out_df["ld"], errors="coerce")
        .fillna(0)
        .round()
        .clip(lower=0)
        .astype(int)
    )
    out_df["churn"] = (out_df["la"] + out_df["ld"]).astype(int)
    if "buggy" in out_df.columns and "bugcount" in out_df.columns:
        out_df["buggy"] = (
            pd.to_numeric(out_df["bugcount"], errors="coerce").fillna(0) > 0
        ).astype(int)
    if drop_columns:
        out_df = out_df.drop(columns=drop_columns, errors="ignore")

    logger.info(
        "K-means clustered %s complete: input rows=%d, output rows=%d.",
        method,
        total_input_rows,
        total_output_rows,
    )
    logger.info(
        "Run summary: clusters=%d, applied=%d, skipped_small=%d, skipped_no_diversity=%d, skipped_no_independent_attrs=%d, fallback_empty=%d, fallback_error=%d, pruned_rows=%d",
        len(grouped),
        cluster_status_counts["applied"],
        cluster_status_counts["skipped_small"],
        cluster_status_counts["skipped_no_diversity"],
        cluster_status_counts["skipped_no_independent_attrs"],
        cluster_status_counts["fallback_empty"],
        cluster_status_counts["fallback_error"],
        total_pruned_rows,
    )
    return out_df


def parse_cli_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="K-means cluster-first anonymization using bundled LACE"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        help="Directory containing input.csv (and where output will be written)",
    )
    parser.add_argument(
        "--input-csv", type=str, help="Explicit input CSV path (overrides --data-dir)"
    )
    parser.add_argument(
        "--output-csv", type=str, help="Explicit output CSV path (overrides --data-dir)"
    )
    parser.add_argument(
        "--method",
        choices=["lace1", "lace2"],
        default="lace1",
        help="Which LACE variant to apply inside each cluster",
    )
    parser.add_argument(
        "--n-clusters",
        type=int,
        default=0,
        help="K for k-means clustering; 0 means automatic from data size",
    )
    parser.add_argument(
        "--min-cluster-size",
        type=int,
        default=25,
        help="Used for automatic k selection when --n-clusters is 0",
    )
    parser.add_argument(
        "--objective",
        default="buggy",
        help="Objective column used by LACE (default: buggy, derived from bugcount > 0)",
    )
    parser.add_argument(
        "--objective-binary",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Treat the objective as binary (default: true)",
    )
    parser.add_argument(
        "--cliff",
        type=float,
        default=0.8,
        help="cliff_percentage passed to LACE",
    )
    parser.add_argument("--alpha", type=float, default=0.15, help="LACE1 alpha")
    parser.add_argument("--beta", type=float, default=0.35, help="LACE1 beta")
    parser.add_argument(
        "--morph-alpha", type=float, default=0.15, help="LACE2 morph_alpha"
    )
    parser.add_argument(
        "--morph-beta", type=float, default=0.35, help="LACE2 morph_beta"
    )
    parser.add_argument(
        "--holders",
        type=int,
        default=5,
        help="Number of holders for LACE2 simulator",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging verbosity",
    )
    parser.add_argument(
        "--log-file",
        type=str,
        help="Optional log file path",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_cli_args()
    global INPUT_CSV_PATH, OUTPUT_CSV_PATH

    data_dir = args.data_dir
    if args.input_csv:
        INPUT_CSV_PATH = args.input_csv
    elif data_dir:
        INPUT_CSV_PATH = os.path.join(data_dir, "input.csv")

    if args.output_csv:
        OUTPUT_CSV_PATH = args.output_csv
    elif data_dir:
        OUTPUT_CSV_PATH = os.path.join(data_dir, f"output_{args.method}_kmeans.csv")

    log_file = args.log_file
    if not log_file and data_dir:
        log_file = os.path.join(data_dir, f"anonymizer_lace_kmeans_{args.method}.log")

    setup_logging(args.log_level, log_file)

    logger.info("Starting k-means + LACE anonymization")
    logger.info("Input=%s, Output=%s", INPUT_CSV_PATH, OUTPUT_CSV_PATH)
    logger.info(
        "Method=%s, n_clusters=%d, min_cluster_size=%d, Objective=%s, ObjectiveBinary=%s, Cliff=%s",
        args.method,
        args.n_clusters,
        args.min_cluster_size,
        args.objective,
        args.objective_binary,
        args.cliff,
    )

    if not INPUT_CSV_PATH or not OUTPUT_CSV_PATH:
        raise EnvironmentError("Set INPUT_CSV_PATH and OUTPUT_CSV_PATH.")

    df = pd.read_csv(INPUT_CSV_PATH)
    if ID_FIELD not in df.columns or "la" not in df.columns or "ld" not in df.columns:
        raise ValueError("Input must include commit_id, la, ld.")

    df = preprocess_df(df, DATE_FIELD)
    df, derived_only_cols = add_derived(df)

    anonymized_df = anonymize_with_kmeans_lace(
        df=df,
        method=args.method,
        objective=args.objective,
        objective_binary=args.objective_binary,
        cliff=args.cliff,
        alpha=args.alpha,
        beta=args.beta,
        morph_alpha=args.morph_alpha,
        morph_beta=args.morph_beta,
        holders=args.holders,
        n_clusters=args.n_clusters,
        min_cluster_size=args.min_cluster_size,
        drop_columns=derived_only_cols,
    )

    anonymized_df.to_csv(OUTPUT_CSV_PATH, index=False)
    logger.info(
        "Done. Writing k-means %s output to %s (rows=%d).",
        args.method,
        OUTPUT_CSV_PATH,
        len(anonymized_df),
    )


if __name__ == "__main__":
    main()
