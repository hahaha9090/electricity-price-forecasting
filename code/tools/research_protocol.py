"""Frozen EPF-QNBM-001 data, time and run-identity boundaries.

The historical CSVs have artificial 24-hour days; no DST/issue timestamp claim
is inferred from that representation. Legacy scaling includes the historical
validation pool, as the original code does. Chronological scaling does not.
"""
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

QUANTILES = [i / 10 for i in range(1, 10)]
SEEDS = [11, 22, 33, 44]
MARKETS = ['DE_price', 'MGP_NORD', 'MGP_CNOR', 'MGP_CSUD', 'MGP_SUD', 'MGP_SARD', 'MGP_SICI']


def partition_dates(dates,index=0,count=1):
    """Disjoint complete dates; never split or change the four member seeds."""
    if count<1 or not 0<=index<count:
        raise ValueError('Invalid date partition')
    return dates[index::count]


def market_spec(market):
    if market not in MARKETS:
        raise ValueError('Unknown market: ' + market)
    if market == 'DE_price':
        return dict(market=market, dataset='DE_market__2015-01-01__2020-12-31.csv',
                    train_start='2015-01-01', origin='2018-12-27', lag_days=2,
                    tune_start='2019-06-27', tune_end='2019-09-30',
                    screen_start='2019-10-01', screen_end='2019-12-31',
                    test_start='2020-01-01', test_end='2020-12-31')
    return dict(market=market, dataset=market+'_market__2015-01-01__2019-08-31.csv',
                train_start='2015-01-10', origin='2018-03-01', lag_days=7,
                tune_start='2018-08-30', tune_end='2018-10-15',
                screen_start='2018-10-16', screen_end='2018-11-30',
                test_start='2018-12-01', test_end='2019-08-31')


def require_stage(market, day, stage, frozen=False):
    spec = market_spec(market)
    day = pd.Timestamp(day).strftime('%Y-%m-%d')
    if stage == 'development':
        valid = spec['origin'] <= day <= spec['screen_end']
    elif stage == 'evaluation':
        valid = frozen and spec['test_start'] <= day <= spec['test_end']
    else:
        raise ValueError('Stage must be development or evaluation')
    if not valid:
        raise ValueError('Date/stage is outside the frozen protocol')


def data_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode('utf-8')).hexdigest()


def write_json(path, value):
    """Atomic complete-file replacement, only for the current run's state."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    temporary.replace(path)


def create_run(root, run_id, identity):
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,95}', run_id):
        raise ValueError('Run ID must be a single safe directory name')
    path = Path(root) / run_id
    path.mkdir(parents=True, exist_ok=False)
    write_json(path/'identity.json', dict(identity=identity, sha256=canonical_hash(identity)))
    return path


def assert_run_identity(path, identity):
    stored = json.loads((Path(path)/'identity.json').read_text(encoding='utf-8'))
    if stored['identity'] != identity or stored['sha256'] != canonical_hash(identity):
        raise ValueError('Run identity mismatch; use a new run ID')


def load_market(root, market):
    path = Path(root)/'data'/'datasets'/market_spec(market)['dataset']
    frame = pd.read_csv(path)
    # No reading a model result is needed to audit the data.
    dates = pd.to_datetime(frame['Date'])
    hours = frame['Hour'].to_numpy()
    if len(frame) % 24 or not np.array_equal(hours, np.tile(np.arange(24), len(frame)//24)):
        raise ValueError('Expected complete ordered 24-hour days')
    daily = dates.iloc[::24]
    if daily.duplicated().any() or not np.all(np.diff(daily.to_numpy()).astype('timedelta64[D]') == np.timedelta64(1, 'D')):
        raise ValueError('Dates must be consecutive and unique')
    if not np.array_equal(dates.to_numpy(), np.repeat(daily.to_numpy(), 24)):
        raise ValueError('Date/Hour rows disagree')
    return frame


def prepare_day(frame, market, day, protocol='legacy', split_seed=2026):
    """Prepare one daily fit without ever exposing the forecast-day target.

    Returns float32 windows, training labels, raw test labels (for *subsequent*
    scoring only), fitted target location/scale and auditable date membership.
    The target column in x_test's future block is explicitly masked to zero.
    """
    spec = market_spec(market)
    day = pd.Timestamp(day).normalize()
    origin = pd.Timestamp(spec['origin'])
    if day < origin or day > pd.Timestamp(spec['test_end']):
        raise ValueError('Forecast origin outside market range')
    if protocol not in ('legacy', 'chronological'):
        raise ValueError('Unknown validation protocol')
    start = pd.Timestamp(spec['train_start']) + (day-origin)
    selected = frame.loc[(frame.Date >= str(start.date())) & (frame.Date <= str(day.date()))].copy()
    target = 'TARG__'+market
    prefixes = ('PAST__', 'FUTU__', 'CONST__', 'F_L-1__', 'CONST_L-2__')
    columns = [target] + [c for c in frame if c.startswith(prefixes)]
    values = selected[columns].to_numpy(dtype=np.float64)
    expected = ((day-start).days+1)*24
    if len(values) != expected or not np.array_equal(selected.Hour.to_numpy(), np.tile(np.arange(24), expected//24)):
        raise ValueError('Incomplete or unordered daily window')
    if not np.isfinite(values).all():
        raise ValueError('Non-finite dataset values')
    lag = spec['lag_days']
    label_dates = pd.date_range(start+pd.Timedelta(days=lag), day-pd.Timedelta(days=1))
    count = len(label_dates)
    n_val = 291 if protocol == 'legacy' else 28
    if count <= n_val:
        raise ValueError('Insufficient training/validation history')
    indices = np.arange(count)
    if protocol == 'legacy':
        indices = np.random.RandomState(split_seed + day.toordinal()).permutation(indices)
    train_idx, val_idx = indices[:-n_val], indices[-n_val:]
    fit_end = len(values)-24 if protocol == 'legacy' else (lag+int(val_idx[0]))*24
    target_scaler = StandardScaler().fit(values[:fit_end, :1])
    feature_scaler = StandardScaler().fit(values[:fit_end, 1:])
    transformed = np.concatenate([target_scaler.transform(values[:, :1]),
                                  feature_scaler.transform(values[:, 1:])], axis=1).astype(np.float32)
    # Mask after fitting/transforming: no current answer in the predictor input.
    transformed[-24:, 0] = 0.
    total = (lag+1)*24
    windows = np.stack([transformed[i*24:i*24+total] for i in range(count+1)])
    # Historical labels are scaled; evaluation labels stay in actual price units.
    labels = windows[:-1, -24:, 0].copy()
    names = [x.strftime('%Y-%m-%d') for x in label_dates]
    return dict(train_x=windows[train_idx].copy(), train_y=labels[train_idx],
                val_x=windows[val_idx].copy(), val_y=labels[val_idx],
                x_test=windows[-1:].copy(), y_test=values[-24:, 0].copy(),
                target_mean=float(target_scaler.mean_[0]), target_scale=float(target_scaler.scale_[0]),
                feature_scaler=feature_scaler, target_scaler=target_scaler,
                x_columns_names=columns,
                metadata=dict(day=str(day.date()), protocol=protocol, window_start=str(start.date()),
                              train_dates=[names[i] for i in train_idx], validation_dates=[names[i] for i in val_idx],
                              scaler_fit_last_date=str((start+pd.Timedelta(days=fit_end//24-1)).date()),
                              split_seed=split_seed, future_target_masked=True))
