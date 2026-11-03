"""Evaluation for EPF-QNBM-001, without training or method selection on final data.

All forecast inputs have shape [day, hour, quantile]; targets are [day, hour].
The frozen core is nine equally weighted quantiles 0.1,...,0.9. Pinball is
their mean, not a full CRPS estimate. Intervals use miscoverage alpha as keys:
"0.2", "0.4", "0.6", "0.8" mean 80%, 60%, 40%, 20% coverage.
Coverage errors and their changes are proportions (0.01 is one percentage
point). Relative score changes are fractions (0.05 is five percent).

Evaluation never sorts forecasts, fits a volatility threshold, or determines
dates. Supply already frozen, causally constructed high-volatility labels.
Reversed interval endpoints have status ``invalid_intervals``; their scores
are None rather than misleading negative widths. All metric dictionaries are
strict-JSON compatible; unavailable statistics are None with an explicit status.

Screening is a per-market development decision. The caller must require both
development markets to pass; averaging their gates is not allowed. Locked
evaluation only assesses the already frozen method and never selects a fallback.
"""

import numpy as np


CORE_QUANTILES = np.arange(1, 10, dtype=float) / 10.0
INTERVAL_ALPHAS = (0.2, 0.4, 0.6, 0.8)
_INTERVAL_COLUMNS = ((0, 8), (1, 7), (2, 6), (3, 5))
_INTERVAL_FIELDS = ('coverage', 'coverage_error', 'width', 'winkler',
                    'lower_miss', 'upper_miss')


def _finite_array(values, name):
    try:
        result = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError(name + ' must be a finite numeric array') from error
    if not np.isfinite(result).all():
        raise ValueError(name + ' must contain only finite values')
    return result


def _positive_integer(value, name, minimum=1):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(name + ' must be an integer')
    if value < minimum:
        raise ValueError(name + ' must be at least ' + str(minimum))
    return int(value)


def _validate_inputs(targets, predictions, quantiles):
    y = _finite_array(targets, 'targets')
    predictions = _finite_array(predictions, 'predictions')
    quantiles = _finite_array(quantiles, 'quantiles')
    if y.ndim != 2 or min(y.shape) == 0:
        raise ValueError('targets must have nonempty shape [day, hour]')
    if predictions.shape != y.shape + (9,):
        raise ValueError('predictions must have shape [day, hour, 9] aligned with targets')
    if quantiles.shape != (9,) or not np.allclose(
            quantiles, CORE_QUANTILES, rtol=0.0, atol=1e-12):
        raise ValueError('quantiles must be the ordered core grid 0.1,...,0.9')
    return y, predictions, quantiles


def _volatility_mask(high_volatility, shape):
    if high_volatility is None:
        return np.zeros(shape, dtype=bool)
    labels = _finite_array(high_volatility, 'high_volatility')
    if not np.logical_or(labels == 0, labels == 1).all():
        raise ValueError('high_volatility must contain boolean or 0/1 labels')
    if labels.shape == (shape[0],):
        labels = np.broadcast_to(labels[:, None], shape)
    elif labels.shape != shape:
        raise ValueError('high_volatility must have shape [day] or [day, hour]')
    return labels.astype(bool)


def _observations(y, predictions, quantiles):
    error = y[:, :, None] - predictions
    pinball = np.maximum(quantiles * error, (quantiles - 1.0) * error).mean(axis=2)
    intervals = {}
    for alpha, (lower_column, upper_column) in zip(INTERVAL_ALPHAS, _INTERVAL_COLUMNS):
        lower = predictions[:, :, lower_column]
        upper = predictions[:, :, upper_column]
        width = upper - lower
        intervals[str(alpha)] = {
            'coverage': (y >= lower) & (y <= upper),
            'width': width,
            'winkler': width + (2.0 / alpha) * (
                np.maximum(lower - y, 0.0) + np.maximum(y - upper, 0.0)),
            'lower_miss': y < lower,
            'upper_miss': y > upper,
            'inverted': lower > upper,
        }
    return {'pinball': pinball,
            'mae': np.abs(y - predictions[:, :, 4]),
            'crossing': np.diff(predictions, axis=2) < 0.0,
            'intervals': intervals}


def _interval_summary(values, mask, alpha):
    count = int(mask.sum())
    days = int(np.any(mask, axis=1).sum())
    inverted_count = int(np.logical_and(values['inverted'], mask).sum())
    result = {'status': 'ok', 'count': count, 'days': days,
              'nominal_coverage': 1.0 - alpha, 'inverted_count': inverted_count}
    if count == 0 or inverted_count:
        result['status'] = 'invalid_intervals' if inverted_count else 'evidence_insufficient'
        result.update({name: None for name in _INTERVAL_FIELDS})
        return result
    for name in ('coverage', 'width', 'winkler', 'lower_miss', 'upper_miss'):
        result[name] = float(values[name][mask].mean())
    result['coverage_error'] = abs(result['coverage'] - (1.0 - alpha))
    return result


def evaluate_quantiles(targets, predictions, quantiles, high_volatility=None):
    """Return overall, hourly, fixed-high-volatility, and daily metrics.

    ``crossing_rate`` is the fraction of day/hour rows with at least one
    adjacent quantile inversion; ``crossing_pair_rate`` uses all adjacent pairs.
    ``daily`` keeps one entry per original day, suitable for paired auditing.
    Missing high-volatility labels mean an unavailable group, not all days high.
    """
    y, predictions, quantiles = _validate_inputs(targets, predictions, quantiles)
    high_mask = _volatility_mask(high_volatility, y.shape)
    values = _observations(y, predictions, quantiles)
    full_mask = np.ones(y.shape, dtype=bool)
    result = {
        'status': 'ok', 'days': y.shape[0], 'hours': y.shape[1],
        'quantiles': quantiles.tolist(), 'pinball_definition': 'mean_of_nine_quantiles',
        'pinball': float(values['pinball'].mean()),
        'mae': float(values['mae'].mean()),
        'crossing_rate': float(values['crossing'].any(axis=2).mean()),
        'crossing_pair_rate': float(values['crossing'].mean()),
        'intervals': {},
        'daily': {'pinball': values['pinball'].mean(axis=1).tolist(),
                  'mae': values['mae'].mean(axis=1).tolist(), 'intervals': {}},
    }
    for alpha in INTERVAL_ALPHAS:
        key = str(alpha)
        interval_values = values['intervals'][key]
        summary = _interval_summary(interval_values, full_mask, alpha)
        summary['hourly'] = []
        for hour in range(y.shape[1]):
            hour_mask = np.zeros(y.shape, dtype=bool)
            hour_mask[:, hour] = True
            hourly = _interval_summary(interval_values, hour_mask, alpha)
            hourly['hour'] = hour
            summary['hourly'].append(hourly)
        summary['high_volatility'] = _interval_summary(interval_values, high_mask, alpha)
        result['intervals'][key] = summary
        daily = {name: [] for name in _INTERVAL_FIELDS}
        for day in range(y.shape[0]):
            day_values = {name: array[day:day + 1] for name, array in interval_values.items()}
            day_summary = _interval_summary(day_values, full_mask[day:day + 1], alpha)
            for name in _INTERVAL_FIELDS:
                daily[name].append(day_summary[name])
        result['daily']['intervals'][key] = daily
    return result


def past_volatility(prices, lookback_days=7):
    """Population std over all hours of the previous complete calendar rows.

    Row d uses prices[d-lookback_days:d], excluding the current row. The first
    lookback_days entries are NaN because that much history is unavailable.
    This helper does not fit the frozen 80th-percentile calibration threshold.
    """
    prices = _finite_array(prices, 'prices')
    lookback_days = _positive_integer(lookback_days, 'lookback_days')
    if prices.ndim != 2 or min(prices.shape) == 0:
        raise ValueError('prices must have nonempty shape [day, hour]')
    result = np.full(prices.shape[0], np.nan)
    for day in range(lookback_days, prices.shape[0]):
        result[day] = prices[day - lookback_days:day].std(ddof=0)
    return result


def moving_block_day_indices(n_days, block_days=7, n_boot=2000, seed=2026):
    """Draw circular moving blocks of whole days, returning [replicate, day].

    A block starts uniformly at an observed day and continues chronologically,
    wrapping at the end. Concatenation is truncated to n_days. Callers use the
    same indices for targets, every method, every hour, and frozen labels.
    """
    n_days = _positive_integer(n_days, 'n_days')
    block_days = _positive_integer(block_days, 'block_days')
    n_boot = _positive_integer(n_boot, 'n_boot', minimum=2)
    seed = _positive_integer(seed, 'seed', minimum=0)
    if block_days > n_days:
        raise ValueError('block_days cannot exceed the observed day count')
    rng = np.random.default_rng(seed)
    block_count = (n_days + block_days - 1) // block_days
    starts = rng.integers(0, n_days, size=(n_boot, block_count))
    indices = (starts[:, :, None] + np.arange(block_days)) % n_days
    return indices.reshape(n_boot, -1)[:, :n_days]


def _relative_change(candidate, reference):
    candidate = np.asarray(candidate, dtype=float)
    reference = np.asarray(reference, dtype=float)
    result = np.full(np.broadcast(candidate, reference).shape, np.nan)
    np.divide(candidate - reference, reference, out=result, where=reference > 0.0)
    result = np.where((reference == 0.0) & (candidate == 0.0), 0.0, result)
    return result


def _bootstrap_summary(point, samples, enough_days, reason=None):
    point = float(point)
    finite = np.isfinite(samples)
    valid_samples = int(finite.sum())
    sufficient = enough_days and np.isfinite(point) and bool(finite.all())
    result = {
        'status': 'ok' if sufficient else 'evidence_insufficient',
        'point': point if np.isfinite(point) else None,
        'ci95': np.percentile(samples, [2.5, 97.5]).tolist() if sufficient else None,
        'valid_bootstrap_samples': valid_samples,
    }
    if not sufficient:
        result['reason'] = reason or ('undefined_point_or_replicate' if enough_days
                                      else 'fewer_than_two_blocks_of_days')
    return result


def paired_block_bootstrap(targets, candidate, reference, quantiles,
                           high_volatility=None, block_days=7, n_boot=2000, seed=2026):
    """Compare one frozen candidate/reference pair using paired day blocks.

    Statistics use the complete 80% interval. Coverage-error increase is
    |coverage(candidate)-.8|-|coverage(reference)-.8|; high-volatility improvement
    reverses that subtraction within the supplied group. Ratios recompute the
    denominator for each replicate. CI is a two-sided 95% percentile interval.

    At least two blocks of observed days are required for statistical support;
    the high-volatility comparison additionally needs two blocks of labelled
    days. A replicate without group members invalidates its group CI rather
    than silently dropping the replicate. Undefined relative change against a
    zero reference is insufficient evidence; two exact zero scores give zero.
    No threshold is fitted and no method is selected here.
    """
    y, candidate, quantiles = _validate_inputs(targets, candidate, quantiles)
    _, reference, _ = _validate_inputs(y, reference, quantiles)
    mask = _volatility_mask(high_volatility, y.shape)
    block_days = _positive_integer(block_days, 'block_days')
    n_boot = _positive_integer(n_boot, 'n_boot', minimum=2)
    seed = _positive_integer(seed, 'seed', minimum=0)
    # Even short histories retain a point estimate, but cannot claim support.
    actual_block = min(block_days, y.shape[0])
    indices = moving_block_day_indices(y.shape[0], actual_block, n_boot, seed)
    enough_days = y.shape[0] >= 2 * block_days
    can = _observations(y, candidate, quantiles)
    ref = _observations(y, reference, quantiles)
    can_pinball = can['pinball'].mean(axis=1)
    ref_pinball = ref['pinball'].mean(axis=1)
    pinball_point = _relative_change(can_pinball.mean(), ref_pinball.mean())
    pinball_samples = _relative_change(can_pinball[indices].mean(axis=1),
                                       ref_pinball[indices].mean(axis=1))
    result = {
        'method': 'paired_circular_moving_day_blocks', 'days': y.shape[0],
        'hours': y.shape[1], 'block_days': block_days, 'n_boot': n_boot,
        'seed': seed, 'confidence': 0.95, 'interval_alpha': 0.2,
        'high_volatility_days': int(mask.any(axis=1).sum()),
        'high_volatility_count': int(mask.sum()),
        'pinball_relative_change': _bootstrap_summary(
            pinball_point, pinball_samples, enough_days),
    }
    ci = can['intervals']['0.2']
    ri = ref['intervals']['0.2']
    valid_intervals = not (ci['inverted'].any() or ri['inverted'].any())
    coverage_point = abs(ci['coverage'].mean() - 0.8) - abs(ri['coverage'].mean() - 0.8)
    coverage_samples = (abs(ci['coverage'].mean(axis=1)[indices].mean(axis=1) - 0.8)
                        - abs(ri['coverage'].mean(axis=1)[indices].mean(axis=1) - 0.8))
    winkler_point = _relative_change(ci['winkler'].mean(), ri['winkler'].mean())
    winkler_samples = _relative_change(ci['winkler'].mean(axis=1)[indices].mean(axis=1),
                                       ri['winkler'].mean(axis=1)[indices].mean(axis=1))
    interval_reason = None if valid_intervals else 'inverted_interval_endpoints'
    if not valid_intervals:
        coverage_point = winkler_point = np.nan
        coverage_samples[:] = np.nan
        winkler_samples[:] = np.nan
    result['coverage_error_increase'] = _bootstrap_summary(
        coverage_point, coverage_samples, enough_days, interval_reason)
    result['winkler_relative_change'] = _bootstrap_summary(
        winkler_point, winkler_samples, enough_days, interval_reason)
    group_count = mask.sum(axis=1)
    counts = group_count[indices].sum(axis=1)
    can_hits = np.logical_and(ci['coverage'], mask).sum(axis=1)
    ref_hits = np.logical_and(ri['coverage'], mask).sum(axis=1)
    can_group = np.full(n_boot, np.nan)
    ref_group = np.full(n_boot, np.nan)
    np.divide(can_hits[indices].sum(axis=1), counts, out=can_group, where=counts > 0)
    np.divide(ref_hits[indices].sum(axis=1), counts, out=ref_group, where=counts > 0)
    group_samples = abs(ref_group - 0.8) - abs(can_group - 0.8)
    group_point = (abs(ref_hits.sum() / mask.sum() - 0.8)
                   - abs(can_hits.sum() / mask.sum() - 0.8)) if mask.any() else np.nan
    enough_group = enough_days and int(mask.any(axis=1).sum()) >= 2 * block_days
    if not valid_intervals:
        group_point = np.nan
        group_samples[:] = np.nan
    result['high_vol_coverage_error_improvement'] = _bootstrap_summary(
        group_point, group_samples, enough_group and valid_intervals,
        interval_reason or ('fewer_than_two_blocks_of_high_volatility_days'
                            if not enough_group else None))
    return result


def _number(value):
    if isinstance(value, (bool, np.bool_)):
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if np.isfinite(value) else None


def _get_metric(metrics, *path):
    current = metrics
    for key in path:
        if not isinstance(current, dict):
            return None
        if current.get('status', 'ok') != 'ok':
            return None
        current = current.get(key)
    return _number(current)


def _point_gate(values, rules):
    if any(value is None for value in values.values()):
        status = 'evidence_insufficient'
    else:
        status = 'pass' if all(
            (values[name] <= margin + 1e-12 if direction == 'upper'
             else values[name] >= margin - 1e-12)
            for name, direction, margin in rules) else 'fail'
    return dict(status=status, **values)


def _relative_number(candidate, reference):
    if candidate is None or reference is None:
        return None
    return _number(_relative_change(candidate, reference))


def screening_decision(candidate_metrics, reference_metrics, normalized_metrics=None):
    """Apply frozen per-market development point-estimate gates.

    candidate is QNBM+original OCQR; reference is the same-protocol QR-DNN+OCQR;
    normalized is the optional QNBM+normalized OCQR. Require main pinball <=5%
    and absolute 80%-coverage-error increase <=.01. Normalization must improve
    high-volatility absolute coverage error by >=.01, with overall 80%-Winkler
    increase <=5%, relative to candidate. Both screening markets must pass.
    This function is for development data only, never locked evaluation.
    """
    can_pinball = _get_metric(candidate_metrics, 'pinball')
    ref_pinball = _get_metric(reference_metrics, 'pinball')
    can_error = _get_metric(candidate_metrics, 'intervals', '0.2', 'coverage_error')
    ref_error = _get_metric(reference_metrics, 'intervals', '0.2', 'coverage_error')
    main = _point_gate({
        'pinball_relative_change': _relative_number(can_pinball, ref_pinball),
        'coverage_error_increase': (can_error - ref_error
                                    if can_error is not None and ref_error is not None else None),
    }, [('pinball_relative_change', 'upper', 0.05),
        ('coverage_error_increase', 'upper', 0.01)])
    normalized = {'status': 'not_evaluated'}
    if normalized_metrics is not None:
        old_error = _get_metric(candidate_metrics, 'intervals', '0.2',
                                'high_volatility', 'coverage_error')
        new_error = _get_metric(normalized_metrics, 'intervals', '0.2',
                                'high_volatility', 'coverage_error')
        normalized = _point_gate({
            'high_vol_coverage_error_improvement': (
                old_error - new_error if old_error is not None and new_error is not None else None),
            'winkler_relative_change': _relative_number(
                _get_metric(normalized_metrics, 'intervals', '0.2', 'winkler'),
                _get_metric(candidate_metrics, 'intervals', '0.2', 'winkler')),
        }, [('high_vol_coverage_error_improvement', 'lower', 0.01),
            ('winkler_relative_change', 'upper', 0.05)])
    action = 'stop_expansion'
    if main['status'] == 'pass':
        action = ('advance_normalized_candidate' if normalized['status'] == 'pass'
                  else 'advance_original_ocqr')
    return {'phase': 'development_point_estimates', 'scope': 'one_market',
            'main_gate': main, 'normalization_gate': normalized, 'action': action}


def _confidence_gate(comparison, rules):
    statuses = []
    for name, direction, margin in rules:
        metric = comparison.get(name, {}) if isinstance(comparison, dict) else {}
        ci = metric.get('ci95')
        if metric.get('status') != 'ok' or not isinstance(ci, (tuple, list)) or len(ci) != 2:
            statuses.append('evidence_insufficient')
            continue
        lower, upper = _number(ci[0]), _number(ci[1])
        if lower is None or upper is None or lower > upper:
            statuses.append('evidence_insufficient')
        elif direction == 'upper':
            statuses.append('supported' if upper <= margin + 1e-12
                            else 'failed' if lower > margin + 1e-12
                            else 'evidence_insufficient')
        else:
            statuses.append('supported' if lower >= margin - 1e-12
                            else 'failed' if upper < margin - 1e-12
                            else 'evidence_insufficient')
    status = ('failed' if 'failed' in statuses else 'supported'
              if all(value == 'supported' for value in statuses) else 'evidence_insufficient')
    return {'status': status,
            'criteria': {rule[0]: status for rule, status in zip(rules, statuses)}}


def locked_evaluation_decision(main_comparison, normalized_comparison=None):
    """Assess 95% CI support for one market's already frozen method.

    Arguments are paired_block_bootstrap results: main QNBM+original versus
    QR-DNN+OCQR, optional normalized versus QNBM+original. A noninferiority
    bound must lie wholly below its margin, and improvement wholly above .01.
    An interval wholly beyond the wrong side is ``failed``; overlap or missing
    evidence is ``evidence_insufficient``. The result has no selection action.
    Call only for comparisons fixed before observing locked-evaluation data.
    """
    main = _confidence_gate(main_comparison, [
        ('pinball_relative_change', 'upper', 0.05),
        ('coverage_error_increase', 'upper', 0.01)])
    normalized = {'status': 'not_evaluated'}
    if normalized_comparison is not None:
        normalized = _confidence_gate(normalized_comparison, [
            ('high_vol_coverage_error_improvement', 'lower', 0.01),
            ('winkler_relative_change', 'upper', 0.05)])
    supported = main['status'] == 'supported' and (
        normalized_comparison is None or normalized['status'] == 'supported')
    return {'phase': 'locked_evaluation', 'scope': 'one_market',
            'main_gate': main, 'normalization_gate': normalized,
            'frozen_method_supported': supported}
