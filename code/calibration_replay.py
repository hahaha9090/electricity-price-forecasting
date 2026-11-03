# Modified: standalone source root; reference formulas retained.
"""V3-0 delayed baseline adapters, not a new calibration algorithm.
Existing author/license declarations remain in tools modules. Function extraction
avoids importing unavailable plotting/training dependencies; formulas unchanged.
"""
import ast
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent
def functions(path,names,namespace):
 tree=ast.parse((ROOT/path).read_text(encoding='utf-8-sig'));nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
 if {n.name for n in nodes}!=set(names):raise ValueError('Missing reference function')
 exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),namespace)
 return namespace
NS={'np':np,'Dict':dict,'List':list}
functions('tools/cp_pi.py',['mytan','saturation_fn_log'],NS)
functions('tools/online_calibration.py',['_track_tail','calibrate_quantiles'],NS)
functions('tools/prediction_quantiles_tools.py',['fix_quantile_crossing','exec_cqr'],NS)

def delayed_waci(p,y,warmup,options):
 # Minimal source adapter: retain exact reference formula/grid/alpha handling,
 # move feedback to beginning of d using interval issued on d-2; bag ends d-2.
 tree=ast.parse((ROOT/'tools/waci.py').read_text());node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='waci_quantiles');s=ast.get_source_segment((ROOT/'tools/waci.py').read_text(),node)
 old='''        start=max(0,day-int(window)); n=day-start'''
 new='''        feedback=day-2
        if feedback>=warmup:
            for hh in range(raw.shape[1]):
                for ii,(ll,uu) in enumerate(pairs):
                    width=ordered[feedback,hh,uu]-ordered[feedback,hh,ll]
                    distance=np.abs(width_grid-width)
                    error=float(not (result[feedback,hh,ll]<=y[feedback,hh]<=result[feedback,hh,uu]))
                    weights=np.exp(-(distance**2-distance.min()**2)/(2*sigma**2))
                    states[hh,ii]+=gamma*weights*(alphas[ii]-error)
        end=day-1
        start=max(0,end-int(window)); n=end-start'''
 if old not in s:raise ValueError('Reference WACI changed')
 s=s.replace(old,new).replace('ordered[start:day','ordered[start:end').replace('y[start:day','y[start:end')
 old_update='''                error=float(not (low<=y[day,hour]<=high))
                weights=np.exp(-(distance**2-distance.min()**2)/(2*sigma**2))
                states[hour,interval]+=gamma*weights*(alphas[interval]-error)'''
 if old_update not in s:raise ValueError('Reference update changed')
 s=s.replace(old_update,'');ns={'np':np};exec(compile(s,'delayed_reference_waci','exec'),ns)
 return ns['waci_quantiles'](p,y,np.arange(1,10)/10,warmup=warmup,options=options)

def calibrate(p,y,method,options=None,warmup=182):
 p=np.array(p,float,copy=True);y=np.array(y,float,copy=True);options=options or {};q=np.arange(1,10)/10;pre=p.copy();extra={}
 if method=='waci':
  r=delayed_waci(p,y,warmup,options);pre=r['pre_sort'];extra={'adaptive_alphas':r['adaptive_alphas']}
 elif method=='cqr':
  settings=dict(target_alpha=[.2,.4,.6,.8],q_alpha_map={a:dict(l=i,u=8-i) for i,a in enumerate([.2,.4,.6,.8])})
  # Call exact exec_cqr, also reconstruct unsorted endpoints for sorting audit.
  for d in range(warmup+1,len(y)):
   end=d-1;start=end-warmup
   for i,a in enumerate([.2,.4,.6,.8]):
    pre[d,:,i]=p[d,:,i]-np.quantile(p[start:end,:,i]-y[start:end],1-a/2,axis=0)
    pre[d,:,8-i]=p[d,:,8-i]+np.quantile(y[start:end]-p[start:end,:,8-i],1-a/2,axis=0)
   exact=NS['exec_cqr'](p[start:end],y[start:end],p[d:d+1],settings)
   if not np.allclose(exact,np.sort(pre[d],axis=-1)):raise ValueError('CQR reference mismatch')
 else:
  mode='normalized' if method=='normalized' else 'original';r=NS['calibrate_quantiles'](p,y,q,warmup=warmup,mode=mode,options=options)
  # threshold at d-1 has consumed scores through d-2; apply at d with d's scale.
  t=np.zeros_like(r['corrections']);t[1:]=r['corrections'][:-1]/r['scales'][:-1,:,:,None]*r['scales'][1:,:,:,None]
  for i in range(4):
   lo=p[:,:,i]-t[:,:,i,0];hi=p[:,:,8-i]+t[:,:,i,1];pre[:,:,i]=np.minimum(lo,hi);pre[:,:,8-i]=np.maximum(lo,hi)
  extra={'corrections':t}
 final=np.sort(pre,axis=-1)
 return dict(final=final,pre_sort=pre,**extra)
