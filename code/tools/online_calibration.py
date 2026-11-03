"""Independent, auditable asymmetric OCQR on daily quantile predictions.

The controller follows ``cp_pi.cts_pid``: zero initial thresholds, separate
tails, a past-score-range proportional step, and its lagged log integrator.
The first ``warmup`` days are excluded from evaluation. For scaled modes,
their prediction widths define fixed constants before these known days are
replayed; that replay is not claimed to be an online warmup experiment.
"""

import numpy as np

from tools.cp_pi import saturation_fn_log


def _track_tail(scores, alpha, lr, first_lr, csat, ki, burnin):
    """Mirror the ahead=1 controller, allowing a separately scaled first step."""
    thresholds = np.zeros(len(scores), dtype=float)
    quantile = 0.
    misses_before = 0.
    for day in range(len(scores)):
        step = (lr * (scores[max(day - burnin, 0):day].max() -
                      scores[max(day - burnin, 0):day].min())
                if day > 0 else first_lr)
        covered = thresholds[day] >= scores[day]
        gradient = alpha if covered else -(1. - alpha)
        # The reference intentionally excludes today's coverage from I[t+1].
        integral = saturation_fn_log(misses_before - day * alpha,
                                      day, csat, ki)
        if day < len(scores) - 1:
            quantile -= step * gradient
            thresholds[day + 1] = quantile + integral
            if not np.isfinite(thresholds[day + 1]):
                raise ValueError('online controller produced non-finite corrections')
        misses_before += not covered
    return thresholds


def calibrate_quantiles(predictions, targets, quantiles, warmup=182,
                        mode='original', options=None):
    """Calibrate arrays ``[day, hour, quantile]`` without mutating any input.

    ``original`` uses price-unit scores. ``constant`` divides each tail score
    by its hour/interval's fixed positive warmup-width median ``c0``.
    ``normalized`` uses ``max(current_width, 0.01*c0)`` instead. Each historical
    score retains its own scale; its threshold is expanded by today's scale.
    KI and the first fallback step are divided by c0 in both scaled modes;
    later proportional steps use the range of their already scaled scores.

    ``corrections`` are signed price-unit thresholds (lower, upper), before
    each interval's reference-compatible ordering repair. Negative thresholds
    are allowed. ``pre_sort`` has that pair repair but no final quantile sort;
    its median equals the sorted raw median. ``final`` sorts all quantiles and
    can change their labels, including the median. Metadata quantifies this.
    """
    raw = np.array(predictions, dtype=float, copy=True)
    observed = np.array(targets, dtype=float, copy=True)
    grid = np.asarray(quantiles, dtype=float)
    if raw.ndim != 3 or observed.shape != raw.shape[:2] or min(raw.shape) == 0:
        raise ValueError('predictions [day,hour,quantile] and targets [day,hour] required')
    if (grid.ndim != 1 or len(grid) != raw.shape[-1] or
            not np.isfinite(grid).all() or np.any(np.diff(grid) <= 0) or
            np.any((grid <= 0) | (grid >= 1))):
        raise ValueError('quantiles must be an increasing finite grid within (0,1)')
    if not np.isfinite(raw).all() or not np.isfinite(observed).all():
        raise ValueError('predictions and targets must be finite')
    if mode not in ('original', 'constant', 'normalized'):
        raise ValueError('mode must be original, constant, or normalized')
    if (isinstance(warmup, (bool, np.bool_)) or not isinstance(warmup, (int, np.integer))
            or warmup < 0 or warmup > len(raw) or (mode != 'original' and warmup == 0)):
        raise ValueError('warmup must fit available days and be positive for scaled modes')

    defaults = {'lr': 0.01, 'KI': 10., 'T_burnin': 7, 'Tin': 1e9,
                'delta': 0.05, 'target_alpha': [0.2, 0.4, 0.6, 0.8]}
    settings = dict(defaults)
    if options is not None:
        unknown = set(options) - set(defaults)
        if unknown:
            raise ValueError('unknown calibration options: ' + ', '.join(sorted(unknown)))
        settings.update(options)
    for key in ['lr', 'KI', 'Tin', 'delta']:
        if not np.isscalar(settings[key]) or not np.isfinite(settings[key]):
            raise ValueError(key + ' must be a finite scalar')
        settings[key] = float(settings[key])
    if settings['lr'] < 0 or settings['KI'] < 0 or settings['Tin'] <= 1 or settings['delta'] <= 0:
        raise ValueError('invalid controller lr, KI, Tin, or delta')
    burnin = settings['T_burnin']
    if isinstance(burnin, (bool, np.bool_)) or not isinstance(burnin, (int, np.integer)) or burnin < 1:
        raise ValueError('T_burnin must be a positive integer')
    settings['T_burnin'] = burnin = int(burnin)
    alphas = np.asarray(settings['target_alpha'], dtype=float)
    if (alphas.ndim != 1 or not len(alphas) or not np.isfinite(alphas).all() or
            np.any((alphas <= 0) | (alphas >= 1)) or len(np.unique(alphas)) != len(alphas)):
        raise ValueError('target_alpha must contain distinct values within (0,1)')

    def column(value):
        positions = np.flatnonzero(np.isclose(grid, value, atol=1e-12, rtol=0.))
        if len(positions) != 1:
            raise ValueError('quantile grid must contain exactly one column for ' + str(value))
        return int(positions[0])

    median_column = column(0.5)
    pairs = [(column(alpha / 2.), column(1. - alpha / 2.)) for alpha in alphas]
    used_columns = [position for pair in pairs for position in pair]
    if len(set(used_columns)) != len(used_columns):
        raise ValueError('target_alpha intervals must use distinct quantile columns')
    csat = 2. / np.pi * (np.ceil(np.log(settings['Tin']) * settings['delta']) -
                         1. / np.log(settings['Tin']))
    if not np.isfinite(csat) or csat <= 0:
        raise ValueError('Tin and delta must give a positive finite Csat')

    ordered = np.sort(raw, axis=-1)
    corrected = ordered.copy()
    days, hours, _ = raw.shape
    count_alpha = len(alphas)
    scales = np.ones((days, hours, count_alpha))
    corrections = np.zeros((days, hours, count_alpha, 2))
    controller_corrections = np.zeros_like(corrections)
    references = np.ones((hours, count_alpha))
    floors = np.zeros_like(references)
    pair_sort_changed = 0

    with np.errstate(over='raise', invalid='raise', divide='raise'):
        try:
            for interval, (alpha, (low, high)) in enumerate(zip(alphas, pairs)):
                for hour in range(hours):
                    lower, upper = ordered[:, hour, low], ordered[:, hour, high]
                    if mode != 'original':
                        widths = upper - lower
                        positive = widths[:warmup][widths[:warmup] > 0]
                        if not len(positive):
                            raise ValueError('no positive warmup width for hour {} alpha {}'.format(hour, alpha))
                        c0 = float(np.median(positive))
                        floor = 0.01 * c0
                        references[hour, interval], floors[hour, interval] = c0, floor
                        scales[:, hour, interval] = (c0 if mode == 'constant' else
                                                     np.maximum(widths, floor))
                    else:
                        c0 = 1.
                    scale = scales[:, hour, interval]
                    # Compute scores once: later scale changes never rescale history.
                    scores = np.stack((lower - observed[:, hour],
                                       observed[:, hour] - upper), axis=-1) / scale[:, None]
                    if not np.isfinite(scores).all():
                        raise ValueError('non-finite calibration scores')
                    thresholds = np.column_stack([
                        _track_tail(scores[:, tail], alpha / 2., settings['lr'],
                                    settings['lr'] / c0, csat, settings['KI'] / c0, burnin)
                        for tail in range(2)])
                    controller_corrections[:, hour, interval] = thresholds
                    price_thresholds = thresholds * scale[:, None]
                    corrections[:, hour, interval] = price_thresholds
                    new_lower = lower - price_thresholds[:, 0]
                    new_upper = upper + price_thresholds[:, 1]
                    pair_sort_changed += int(np.count_nonzero(new_lower > new_upper))
                    corrected[:, hour, low] = np.minimum(new_lower, new_upper)
                    corrected[:, hour, high] = np.maximum(new_upper, new_lower)
        except FloatingPointError as error:
            raise ValueError('non-finite calibration arithmetic: ' + str(error)) from error

    final = np.sort(corrected, axis=-1)
    if not all(np.isfinite(value).all() for value in [scales, corrections, corrected, final]):
        raise ValueError('calibration produced non-finite outputs')
    metadata = {
        'mode': mode,
        'target_alpha': alphas.tolist(),
        'quantiles': grid.tolist(),
        'interval_columns': [list(pair) for pair in pairs],
        'median_column': median_column,
        'warmup_days': int(warmup),
        'warmup_replay_days': int(warmup) if mode != 'original' else 0,
        'warmup_is_scored': False,
        'warmup_is_online_causal': mode == 'original',
        'evaluation_start_index': int(warmup),
        'evaluation_days': days - int(warmup),
        'constants_fit_slice': [0, int(warmup)] if mode != 'original' else None,
        'constants_frozen_before_replay': mode != 'original',
        'reference_scales': references.tolist() if mode != 'original' else None,
        'scale_floors': floors.tolist() if mode != 'original' else None,
        'correction_units': 'price',
        'controller_score_units': 'price' if mode == 'original' else 'normalized',
        'options': {key: value for key, value in settings.items() if key != 'target_alpha'},
        'Csat': float(csat),
        'raw_sort_changed_count': int(np.count_nonzero(raw != ordered)),
        'pair_sort_changed_count': pair_sort_changed,
        'final_sort_changed_count': int(np.count_nonzero(corrected != final)),
        'final_sort_changed_rows': int(np.count_nonzero(np.any(corrected != final, axis=-1))),
        'median_changed_rows': int(np.count_nonzero(corrected[..., median_column] != final[..., median_column])),
        'final_sort_max_abs_change': float(np.max(np.abs(corrected - final))),
        'prediction_update_order': 'predict day t, observe targets[t], update for day t+1',
        'historical_scores_keep_their_prediction_scale': True,
    }
    return {'raw': raw, 'raw_for_calibration': ordered, 'pre_sort': corrected,
            'final': final, 'scales': scales, 'corrections': corrections,
            'normalized_corrections': controller_corrections if mode != 'original' else None,
            'metadata': metadata}
