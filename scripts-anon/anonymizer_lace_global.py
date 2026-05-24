import argparse
import hashlib
import logging
import os
import random
import sys

import pandas as pd


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LACE_ROOT = os.path.abspath(os.path.join(BASE_DIR, "..", "LACE"))
if LACE_ROOT not in sys.path:
    sys.path.append(LACE_ROOT)

from lace import lace1, lace2_simulator


logger = logging.getLogger("anon-lace-global")


INPUT_CSV_PATH = "input.csv"
OUTPUT_CSV_PATH = "output.csv"

ID_FIELD = "commit_id"
DATE_FIELD = "author_date"
SENSITIVE = ["la", "ld"]
QIDS = ["nf", "nd", "ns", "ent", "ndev", "nuc", "age", "aexp", "asexp", "arexp"]


def _hash32(value: str) -> int:
    return int(hashlib.sha256(value.encode("utf-8")).hexdigest()[:8], 16)


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


def preprocess_df(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    out = df.copy()
    if date_col in out.columns:
        out[date_col] = pd.to_datetime(out[date_col], unit="s", errors="coerce")
        out = out.sort_values(by=date_col).reset_index(drop=True)
    return out


def add_derived(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    derived_only_cols: list[str] = []
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


def apply_lace_global(
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
    drop_columns: list[str],
) -> pd.DataFrame:
    ensure_objective_available(df, objective)

    independent_attrs = [
        col for col in (SENSITIVE + QIDS) if col in df.columns and col != objective
    ]
    if not independent_attrs:
        raise ValueError("No independent attributes available for LACE.")

    input_rows = len(df)
    attribute_names = list(df.columns)
    data_matrix = df[attribute_names].values.tolist()

    logger.info(
        "Applying global %s: input_rows=%d, independent_attrs=%s",
        method,
        input_rows,
        ",".join(independent_attrs),
    )

    random.seed(_hash32(f"global|{method}|{objective}|{input_rows}"))
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
        raise RuntimeError(f"{method} produced no rows.")

    result_df["la"] = (
        pd.to_numeric(result_df["la"], errors="coerce")
        .fillna(0)
        .round()
        .clip(lower=0)
        .astype(int)
    )
    result_df["ld"] = (
        pd.to_numeric(result_df["ld"], errors="coerce")
        .fillna(0)
        .round()
        .clip(lower=0)
        .astype(int)
    )
    result_df["churn"] = (result_df["la"] + result_df["ld"]).astype(int)
    if "buggy" in result_df.columns and "bugcount" in result_df.columns:
        result_df["buggy"] = (
            pd.to_numeric(result_df["bugcount"], errors="coerce").fillna(0) > 0
        ).astype(int)
    if drop_columns:
        result_df = result_df.drop(columns=drop_columns, errors="ignore")

    output_rows = len(result_df)
    logger.info(
        "Global %s complete: input_rows=%d, output_rows=%d, pruned_rows=%d",
        method,
        input_rows,
        output_rows,
        max(input_rows - output_rows, 0),
    )
    return result_df


def parse_cli_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Global anonymization using bundled LACE without clustering"
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
        help="Which LACE variant to apply globally",
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
        OUTPUT_CSV_PATH = os.path.join(data_dir, f"output_{args.method}_global.csv")

    log_file = args.log_file
    if not log_file and data_dir:
        log_file = os.path.join(data_dir, f"anonymizer_lace_global_{args.method}.log")

    setup_logging(args.log_level, log_file)

    logger.info("Starting global LACE anonymization")
    logger.info("Input=%s, Output=%s", INPUT_CSV_PATH, OUTPUT_CSV_PATH)
    logger.info(
        "Method=%s, Objective=%s, ObjectiveBinary=%s, Cliff=%s",
        args.method,
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

    anonymized_df = apply_lace_global(
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
        "Done. Writing global %s output to %s (rows=%d).",
        args.method,
        OUTPUT_CSV_PATH,
        len(anonymized_df),
    )
    anonymized_df.to_csv(OUTPUT_CSV_PATH, index=False)


if __name__ == "__main__":
    main()
