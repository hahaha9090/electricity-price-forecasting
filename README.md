# Probabilistic electricity price forecasting

Code and seven market datasets for multi-quantile forecasting and asymmetric feedback with quantile ordering.

## Setup

Use Python 3.10 in a separate environment:

```sh
python -m venv .venv
# Activate the environment using your operating system's activation command.
pip install -r code/requirements.txt
cd code
python run_qnbm_research.py --help
python run_qnbm_research.py audit
```

TensorFlow 2.13 uses CPU on native Windows; GPU execution requires a compatible Linux environment. No server configuration is included.

## Data

`data/datasets/` contains the seven complete CSV files copied from the study assets. `manifest.json` records dates, row counts and SHA256 hashes. Dataset provenance: [PefCodeBench](https://github.com/bruale/PefCodeBench/tree/main/data/datasets). Its Apache-2.0 license is retained; no separate data-specific license has been identified. This repository does not grant additional rights to underlying market data.

## Training

Run from `code/`. The existing `run_qnbm_research.py` runner provides `tune`, `forecast`, `calibrate`, and `evaluate` commands; `--help` lists options. Use a unique `--run-id`. The runner records fitted parameters and generated outputs locally and refuses inconsistent resumes. Its development selection gates are retained. It is a historical experiment runner; its calibration command uses the earlier timing and sorting protocol.

Data loading, market dates, sample construction and preprocessing boundaries are defined in `tools/research_protocol.py`. The `legacy` protocol fits preprocessing using the historical validation pool as well as training observations. `chronological` uses a separate fitting boundary. Dataset timestamps alone do not establish external-variable publication times.

## Feedback and ordering

`postprocess.py` provides the delayed feedback replay separately from the historical runner. Prepare an NPZ with `predictions` shaped [day,24,9] and `targets` shaped [day,24], in chronological order and original price units. Quantiles are 0.1 through 0.9. Only locally trusted inputs should be used with the historical training runner's pickle artifacts.

```sh
python postprocess.py --help
```

Supply `--lr`, `--ki`, `--burnin`, `--warmup` and `--ordering` (`sort`, `anchor`, or `sweep`) explicitly. Select parameters using development observations only. Feedback on day t uses observations through day t-2. Warmup observations are excluded from reported scores. Outputs must use a new filename. Optimal fitted settings, weights, logs and experimental results are not distributed.

## Scope and verification

This release supplies method implementations, training utilities, datasets and a replay/evaluation entry point. It does not automatically reproduce every manuscript table. Full training and numerical agreement with manuscript results have not been verified for this standalone layout. The minimal dependency check and synthetic replay do not establish reproduction of the paper's results.

## License

See LICENSE, NOTICE, licenses/ and inline third-party notices. Retain applicable attribution when redistributing.
