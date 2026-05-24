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


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LACE_ROOT = os.path.abspath(os.path.join(BASE_DIR, "..", "LACE"))
if LACE_ROOT not in sys.path:
    sys.path.append(LACE_ROOT)

from lace import lace1, lace2_simulator


logger = logging.getLogger("anon-lace")


INPUT_CSV_PATH = "input.csv"
OUTPUT_CSV_PATH = "output.csv"

ID_FIELD = "commit_id"
DATE_FIELD = "author_date"
SENSITIVE = ["la", "ld"]
DERIVED = ["churn"]
QIDS = ["nf", "nd", "ns", "ent", "ndev", "nuc", "age", "aexp", "asexp", "arexp"]

NUM_BINS = 20


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


def build_bins(
    df: pd.DataFrame, cols: List[str], q: int = NUM_BINS
) -> Dict[str, np.ndarray]:
    bins_map: Dict[str, np.ndarray] = {}
    for col in cols:
        series = pd.to_numeric(df[col], errors="coerce")
        _, bins = pd.qcut(series, q, duplicates="drop", retbins=True)
        bins_map[col] = bins
    return bins_map


def bin_index(value, bins: np.ndarray) -> int:
    try:
        numeric_value = float(value)
        index = np.digitize([numeric_value], bins, right=True)[0] - 1
        if 0 <= index < len(bins) - 1:
            return int(index)
    except Exception:
        pass
    return -1


def pick_primary_qid(commit_id: str, qids: List[str] = QIDS) -> str:
    return qids[_hash32(commit_id) % len(qids)]


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


def assign_clusters(df: pd.DataFrame, bins_map: Dict[str, np.ndarray]) -> pd.Series:
    keys = []
    for _, row in df.iterrows():
        commit_id = str(row[ID_FIELD])
        qid = pick_primary_qid(commit_id)
        bucket = bin_index(row[qid], bins_map[qid])
        keys.append((qid, bucket))
    return pd.Series(keys, index=df.index, name="cluster_key")


def preprocess_df(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    out = df.copy()
    if date_col in out.columns:
        out[date_col] = pd.to_datetime(out[date_col], unit="s", errors="coerce")
        out = out.sort_values(by=date_col).reset_index(drop=True)
    return out


def _objective_has_diversity(
    cluster_df: pd.DataFrame, objective: str, objective_binary: bool
) -> bool:
    values = pd.to_numeric(cluster_df[objective], errors="coerce").fillna(0)
    if objective_binary:
        return int((values > 0).nunique()) > 1
    return int(values.nunique()) > 1


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


def anonymize_with_lace(
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
    drop_columns: List[str],
) -> pd.DataFrame:
    ensure_objective_available(df, objective)

    logger.info("Building %d quantile bins for QID columns...", NUM_BINS)
    for qid in QIDS:
        if qid not in df.columns:
            raise KeyError(f"Missing QID column: {qid}")
    bins_map = build_bins(df, QIDS, q=NUM_BINS)

    logger.info("Assigning rows to clusters...")
    cluster_keys = assign_clusters(df, bins_map)
    logger.info("Assigned rows to %d unique clusters.", len(cluster_keys.unique()))

    grouped: Dict[Tuple[str, int], List[int]] = {}
    for index, key in cluster_keys.items():
        grouped.setdefault(key, []).append(index)

    lace_parts: List[pd.DataFrame] = []
    cluster_status_counts: Counter[str] = Counter()
    total_input_rows = len(df)
    total_output_rows = 0
    total_pruned_rows = 0

    for key, indexes in grouped.items():
        qid, bucket = key
        cluster_id = f"{qid}|{bucket}"
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
        logger.info(
            "Cluster %s result: status=%s, input_rows=%d, output_rows=%d",
            cluster_id,
            status,
            len(cluster_df),
            len(result_df),
        )

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
        "Clustered %s complete: input rows=%d, output rows=%d.",
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
        description="Cluster-first anonymization using bundled LACE"
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
        "--bins", type=int, help="Number of quantile bins for QIDs (default 20)"
    )
    parser.add_argument(
        "--method",
        choices=["lace1", "lace2"],
        default="lace1",
        help="Which LACE variant to apply inside each cluster",
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
    global INPUT_CSV_PATH, OUTPUT_CSV_PATH, NUM_BINS

    data_dir = args.data_dir
    if args.input_csv:
        INPUT_CSV_PATH = args.input_csv
    elif data_dir:
        INPUT_CSV_PATH = os.path.join(data_dir, "input.csv")

    if args.output_csv:
        OUTPUT_CSV_PATH = args.output_csv
    elif data_dir:
        OUTPUT_CSV_PATH = os.path.join(data_dir, "output.csv")

    if args.bins:
        NUM_BINS = args.bins

    log_file = args.log_file
    if not log_file and data_dir:
        log_file = os.path.join(data_dir, f"anonymizer_lace_{args.method}.log")

    setup_logging(args.log_level, log_file)

    logger.info("Starting clustered LACE anonymization")

    logger.info("Input=%s, Output=%s", INPUT_CSV_PATH, OUTPUT_CSV_PATH)
    logger.info(
        "Method=%s, Bins=%d, Objective=%s, ObjectiveBinary=%s, Cliff=%s",
        args.method,
        NUM_BINS,
        args.objective,
        args.objective_binary,
        args.cliff,
    )
    if log_file:
        logger.info("LogFile=%s", log_file)

    if not INPUT_CSV_PATH or not OUTPUT_CSV_PATH:
        raise EnvironmentError("Set INPUT_CSV_PATH and OUTPUT_CSV_PATH.")

    logger.info("Loading data from %s...", INPUT_CSV_PATH)
    df = pd.read_csv(INPUT_CSV_PATH)
    logger.info("Loaded %d rows.", len(df))
    if ID_FIELD not in df.columns or "la" not in df.columns or "ld" not in df.columns:
        raise ValueError("Input must include commit_id, la, ld.")

    df = preprocess_df(df, DATE_FIELD)

    logger.info(
        "Adding derived columns (churn, buggy from bugcount > 0 when available)..."
    )
    df, derived_only_cols = add_derived(df)

    anonymized_df = anonymize_with_lace(
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
        drop_columns=derived_only_cols,
    )

    logger.info(
        "Done. Writing clustered %s output to %s (rows=%d).",
        args.method,
        OUTPUT_CSV_PATH,
        len(anonymized_df),
    )
    anonymized_df.to_csv(OUTPUT_CSV_PATH, index=False)


if __name__ == "__main__":
    main()
