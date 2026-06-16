# BSA Replication

This repository contains the code, input data, and precomputed outputs needed to reproduce the experiments for the paper.

The anonymization stage can be slow, especially for the full set of projects and the RQ5 sensitivity runs. For practical replication, this repository includes a precomputed artifact zip with already anonymized outputs and final result folders. Reviewers can use those artifacts to recreate tables and figures quickly, or rerun the anonymization and evaluation pipeline from scratch.

## Contents

```text
repo-root/
  README.md
  MANIFEST.md
  requirements.txt
  artifacts/precomputed_outputs.zip
  projects/
  scripts-anon/
  scripts-results/
  LACE/
```

- `projects/`: per-project input CSVs used by the experiment.
- `scripts-anon/`: anonymization scripts used by the pipeline.
- `scripts-results/`: orchestration, evaluation, statistics, and plotting scripts.
- `LACE/`: vendored LACE implementation used for comparison/reference.
- `requirements.txt`: Python dependencies.
- `artifacts/precomputed_outputs.zip`: precomputed anonymized outputs, LACE outputs, and final results.

## Precomputed Artifact

The artifact zip contains these top-level folders:

```text
anon_results/
lace-results/
results/
```

Purpose of each folder:

- `anon_results/`: precomputed anonymized CSV outputs. This avoids requiring reviewers to wait for the slow anonymization stage.
- `lace-results/`: imported or computed LACE comparison outputs used by the analysis scripts.
- `results/`: final evaluation outputs, statistical test outputs, and generated figures.

After unzipping the artifact into the repository root, the layout should include:

```text
repo-root/
  anon_results/
  lace-results/
  results/
  projects/
  scripts-anon/
  scripts-results/
  LACE/
```

## Environment Setup

The main runner expects a local virtual environment at `.venv/`.

From the repository root:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

If `py -3.12` is not available, use a compatible Python 3 version and keep the virtual environment path as `.venv`.

## Fast Reproduction Path

Use this path to regenerate the analysis outputs from the supplied precomputed artifact.

1. Unzip `artifacts/precomputed_outputs.zip` into the repository root.

2. Rebuild the statistical outputs:

```powershell
.\.venv\Scripts\python.exe scripts-results\stats_tests.py --results-root results
```

3. Rebuild the figures:

```powershell
.\.venv\Scripts\python.exe scripts-results\visualize_results.py --results-root results
```

4. Optional: rebuild the RQ5 combined heatmap:

```powershell
.\.venv\Scripts\python.exe scripts-results\plot_rq5_combined_heatmap_v2.py
```

This path verifies the analysis layer using the supplied anonymized outputs and final result files.

## Full Reproduction Path

Use this path to rerun anonymization, model evaluation, statistics, and visualizations.

First unzip the precomputed artifact zip, because the pipeline expects the graph-anonymizer outputs to exist under `anon_results/`.

Then run:

```powershell
.\scripts-results\run_all_experiments_cla.ps1
```

The full runner performs these steps:

1. Generates or reuses anonymized CSVs for the active methods.
2. Reads graph-anonymizer outputs from `anon_results/`.
3. Runs privacy and Random Forest utility evaluation.
4. Runs statistical tests.
5. Generates plots.
6. Runs RQ5 sensitivity analysis by default.

The process can take a long time. To reduce runtime during a check run, disable RQ5:

```powershell
.\scripts-results\run_all_experiments_cla.ps1 -IncludeRQ5 $false
```

To force regeneration of existing anonymized outputs:

```powershell
.\scripts-results\run_all_experiments_cla.ps1 -SkipExisting:$false
```

To limit project-level parallelism:

```powershell
.\scripts-results\run_all_experiments_cla.ps1 -MaxParallelProjects 2
```

## Expected Outputs

After a successful run, `results/` should include:

```text
results/
  cassandra/
  flink/
  groovy/
  ignite/
  openstack/
  qt/
  stats/
  visualizations/
```

Each project folder contains evaluation summaries such as:

- `results.json`
- `summary.txt`
- `runs.txt`
- `feature_importance_avg.txt`
- `feature_importance_runs.txt`

The `stats/` and `visualizations/` folders contain the statistical tables and paper figures.

## Notes for Reviewers

- The anonymization process is intentionally included for transparency, but it is slow for the full experiment.
- The supplied artifact zip contains already anonymized outputs so reviewers can reproduce the analysis and figures without rerunning the slowest stage.
- Rerunning the full pipeline may produce small numeric differences because Random Forest training uses repeated randomized runs.
- The main experiment entry point is `scripts-results/run_all_experiments_cla.ps1`.
- The main analysis scripts are `scripts-results/stats_tests.py` and `scripts-results/visualize_results.py`.

## Troubleshooting

- If the runner reports `Missing venv python`, create `.venv/` as shown in the setup section.
- If it reports missing graph method CSVs, confirm that `anon_results/` was extracted from the artifact zip into the repository root.
- If Python imports fail, reinstall dependencies with `.\.venv\Scripts\python.exe -m pip install -r requirements.txt`.
- If figures are missing, rerun `stats_tests.py` first and then `visualize_results.py`.
