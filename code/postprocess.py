"""Replay feedback and evaluate locally generated forecasts (no training)."""
import argparse
from pathlib import Path
import numpy as np
from calibration_replay import calibrate
from ordering import anchor_project, median_sweep
from tools.research_evaluation import evaluate_quantiles

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', help='NPZ containing predictions [day,24,9] and targets [day,24] in price units')
    parser.add_argument('output', help='New NPZ output; existing files are refused')
    parser.add_argument('--warmup', type=int, default=182)
    parser.add_argument('--lr', type=float, required=True)
    parser.add_argument('--ki', type=float, required=True)
    parser.add_argument('--burnin', type=int, required=True)
    parser.add_argument('--ordering', choices=['sort','anchor','sweep'], required=True)
    args = parser.parse_args()
    with np.load(args.input, allow_pickle=False) as data:
        p, y = data['predictions'], data['targets']
    if p.ndim != 3 or p.shape[1:] != (24,9) or y.shape != p.shape[:2]:
        raise ValueError('Expected aligned [day,24,9] forecasts and [day,24] targets')
    if not np.isfinite(p).all() or not np.isfinite(y).all() or (np.diff(p,axis=-1)<0).any():
        raise ValueError('Inputs must be finite and quantiles ordered')
    if not 0 <= args.warmup < len(y):
        raise ValueError('Warmup must leave evaluation observations')
    result = calibrate(p,y,'ocqr',dict(lr=args.lr,KI=args.ki,T_burnin=args.burnin),warmup=args.warmup)
    final = result['final']
    if args.ordering == 'anchor': final = anchor_project(result['pre_sort'],p[...,4])
    if args.ordering == 'sweep': final = median_sweep(result['pre_sort'],p[...,4])
    import json
    metrics = evaluate_quantiles(y[args.warmup:],final[args.warmup:],np.arange(1,10)/10)
    with Path(args.output).open('xb') as stream:
        np.savez_compressed(stream,predictions=final,targets=y,metrics_json=json.dumps(metrics),warmup=args.warmup)
    print(json.dumps(metrics,indent=2))
if __name__ == '__main__': main()
