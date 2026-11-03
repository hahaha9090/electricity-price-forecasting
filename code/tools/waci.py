"""Width-Adaptive Conformal Inference on existing quantile forecasts.

Adapted from CCaribe9/HQR-WACI models.py/WidthAdaptiveConformalInference,
https://github.com/CCaribe9/HQR-WACI and arXiv:2406.14904v2, section 3.2.
This is the WACI calibration component, not a reproduction of HQR forecasts.
Its symmetric max-tail score, 500-day rolling bag, grid-indexed alpha vector,
Gaussian weights normalized by their maximum, and alpha boundary behavior
follow the public implementation. Stable exponent centering avoids 0/0.
No finite clipping of infinite conformal sets is introduced.

MIT License
Copyright (c) 2024 CCaribe9

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
import numpy as np


def waci_quantiles(predictions,targets,quantiles,warmup=182,options=None):
    raw=np.array(predictions,dtype=float,copy=True)
    y=np.asarray(targets,dtype=float)
    grid=np.asarray(quantiles,dtype=float)
    if (raw.ndim!=3 or raw.shape[-1]!=9 or y.shape!=raw.shape[:2] or
        grid.shape!=(9,) or not np.allclose(grid,np.arange(1,10)/10,rtol=0,atol=1e-12)):
        raise ValueError('Expected aligned [D,H,9], [D,H] and nine core quantiles')
    if not np.isfinite(raw).all() or not np.isfinite(y).all(): raise ValueError('Finite inputs required')
    if isinstance(warmup,(bool,np.bool_)) or not isinstance(warmup,(int,np.integer)) or not 1<=warmup<len(y):
        raise ValueError('Positive warmup and later evaluation observations required')
    settings=dict(gamma=.02,sigma=3.,width_grid=np.arange(0.,500.,.1),window=500)
    if options:
        if set(options)-set(settings): raise ValueError('Unknown WACI option')
        settings.update(options)
    gamma=float(settings['gamma']); sigma=float(settings['sigma']); width_grid=np.asarray(settings['width_grid'],dtype=float)
    window=settings['window']
    if (not np.isfinite([gamma,sigma]).all() or gamma<0 or sigma<=0 or
        width_grid.ndim!=1 or not len(width_grid) or not np.isfinite(width_grid).all() or
        (np.diff(width_grid)<=0).any() or isinstance(window,bool) or
        not isinstance(window,(int,np.integer)) or window<1): raise ValueError('Invalid WACI parameters')
    ordered=np.sort(raw,axis=-1); result=ordered.copy()
    alphas=np.array([.2,.4,.6,.8]); pairs=[(0,8),(1,7),(2,6),(3,5)]
    states=np.broadcast_to(alphas[None,:,None],(raw.shape[1],4,len(width_grid))).copy()
    chosen=np.full((len(y),raw.shape[1],4),np.nan)
    for day in range(warmup,len(y)):
        start=max(0,day-int(window)); n=day-start
        for hour in range(raw.shape[1]):
            for interval,(lower,upper) in enumerate(pairs):
                low,high=ordered[day,hour,lower],ordered[day,hour,upper]
                width=high-low
                distance=np.abs(width_grid-width); closest=int(np.argmin(distance))
                alpha=states[hour,interval,closest]; chosen[day,hour,interval]=alpha
                if 0<alpha<1:
                    scores=np.maximum(ordered[start:day,hour,lower]-y[start:day,hour],
                                      y[start:day,hour]-ordered[start:day,hour,upper])
                    level=np.clip((n+1)*(1-alpha)/n,0,1)
                    qhat=np.quantile(scores,level,method='linear')
                    low-=qhat; high+=qhat
                elif alpha<=0:
                    low=-np.inf; high=np.inf
                # The author's alpha>=1 branch returns the uncorrected interval.
                result[day,hour,lower]=low; result[day,hour,upper]=high
                error=float(not (low<=y[day,hour]<=high))
                weights=np.exp(-(distance**2-distance.min()**2)/(2*sigma**2))
                states[hour,interval]+=gamma*weights*(alphas[interval]-error)
    final=np.sort(result,axis=-1)
    bad=~np.isfinite(result[warmup:])
    return dict(raw=raw,pre_sort=result,final=final,adaptive_alphas=chosen,
                final_alpha_grids=states,width_grid=width_grid,
                metadata=dict(method='WACI_component_adapter',source='https://github.com/CCaribe9/HQR-WACI',
                    warmup=int(warmup),gamma=gamma,sigma=sigma,window=int(window),
                    width_grid_size=len(width_grid),nonfinite_interval_count=int(bad.sum()//2),
                    final_sort_changed_rows=int(np.any(result!=final,axis=-1).sum()),
                    infinite_sets_clipped=False,hqr_reproduced=False,
                    alpha_selection='nearest_absolute_width_distance',quantile_method='clipped_level_linear',
                    prediction_update_order='predict current day before updating each hour/interval state'))
