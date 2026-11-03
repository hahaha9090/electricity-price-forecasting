# Standalone release copy; data location and environment reporting adapted.
"""Frozen, resumable EPF-QNBM-001 experiments; import never trains.

Run --help. A smoke run cannot be used as a tuned development result. Locked
dates require a source/data/protocol-bound selection from both pilot markets.
"""
import argparse
import contextlib
import json
import os
from pathlib import Path
import platform
import subprocess
import shutil
import sys
import tempfile
import time

import numpy as np
import pandas as pd

from tools.research_protocol import (QUANTILES, SEEDS, MARKETS, market_spec,
    prepare_day, load_market, require_stage, data_hash, canonical_hash,
    write_json, create_run, assert_run_identity, partition_dates)
from tools.research_artifacts import (series_digest,save_calibration,load_calibration,
                                      evidence_hashes,verify_evidence,experiment_lock)

ROOT = Path(__file__).resolve().parent
PILOT_MARKETS = ['DE_price', 'MGP_NORD']


def source_files():
    return ([ROOT/'run_qnbm_research.py',ROOT/'requirements.txt',ROOT/'LICENSE']+
            sorted((ROOT/'tools').rglob('*.py'))+sorted((ROOT/'scripts').rglob('*.py')))


def source_hash():
    return canonical_hash({p.relative_to(ROOT).as_posix():data_hash(p) for p in source_files()})


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def price_explanation(explanation, location, scale):
    result = dict(explanation)
    result['raw'] = np.asarray(result['raw'])*scale+location
    result['reference_prediction'] = np.asarray(result['reference_prediction'])*scale+location
    result['contributions'] = np.asarray(result['contributions'])*scale
    if not np.allclose(result['reference_prediction']+result['contributions'].sum(-1), result['raw'], rtol=2e-5, atol=2e-4):
        raise ValueError('Feature contributions do not reconstruct raw price prediction')
    return result


def forecast_identity(market, model, protocol, epochs, data_sha, source_sha, hyperparameters,runtime_policy_sha=None):
    return dict(market=market, model=model, protocol=protocol, epochs=epochs,
                data_sha=data_sha, source_sha=source_sha, hyperparameters=hyperparameters,
                seeds=SEEDS, quantiles=QUANTILES,runtime_policy_sha=runtime_policy_sha)


def freeze_selection(path, screens):
    if set(screens) != set(PILOT_MARKETS):
        raise ValueError('Both pilot-market screening results are required')
    if not all(x['replacement_pass'] for x in screens.values()):
        raise ValueError('Backbone replacement failed; seven-market expansion is stopped')
    method = 'normalized' if all(x['normalized_pass'] for x in screens.values()) else 'original'
    path = Path(path)
    identity = read_json(path/'identity.json')['sha256']
    if any(s.get('identity') != identity for s in screens.values()):
        raise ValueError('Screening evidence belongs to another run')
    for market,screen_record in screens.items(): _verify_screen(path,market,screen_record)
    selection = dict(method=method, screens=screens, selected_from='screening_only', identity=identity,
                     calibration_hashes={m:_parameter_hashes(path,m) for m in PILOT_MARKETS},
                     frozen_at_utc=pd.Timestamp.now(tz='UTC').isoformat())
    target = Path(path)/'selection.json'
    if target.exists():
        raise FileExistsError('Selection is already frozen')
    write_json(target, selection)
    return selection


def _parameter_hashes(run,market):
    return {name:data_hash(run/market/name/'calibration'/'parameters.json') for name in ['DNN','QNBM']}


def _verify_screen(run,market,record):
    if record.get('parameter_hashes') != _parameter_hashes(run,market):
        raise ValueError('Parameters changed after screening')
    for name in ['DNN','QNBM']:
        verify_evidence(run/market/name/'calibration',record['artifact_hashes'][name])


def _verify_selection(run,market=None):
    selection=read_json(run/'selection.json')
    if selection.get('identity') != read_json(run/'identity.json')['sha256']:
        raise ValueError('Frozen selection identity mismatch')
    if selection['method'] not in ('original','normalized'):
        raise ValueError('Unknown frozen method')
    for m,expected in selection['calibration_hashes'].items():
        if expected != _parameter_hashes(run,m): raise ValueError('Frozen calibration parameters were changed')
        _verify_screen(run,m,selection['screens'][m])
    if market is not None and market not in PILOT_MARKETS:
        record=read_json(run/market/'expansion_frozen.json')
        if record['identity']!=selection['identity'] or record['calibration_hashes']!=_parameter_hashes(run,market):
            raise ValueError('Expansion calibration identity mismatch')
    return selection


def _open_run(args):
    identity = dict(study='EPF-QNBM-001', source_sha=source_hash(), protocol=args.protocol,
                    max_epochs=args.max_epochs, network_trials=20, seeds=SEEDS,
                    data_sha={m:data_hash(ROOT.parent/'data'/'datasets'/market_spec(m)['dataset']) for m in MARKETS})
    policy=getattr(args,'runtime_policy',None)
    identity['runtime_policy_sha']=data_hash(policy) if policy else None
    base = Path(args.output_root).resolve()
    if not args.run_id:
        raise ValueError('--run-id is required for experiment writes')
    # Validate the name even when resuming.
    import re
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,95}', args.run_id):
        raise ValueError('Unsafe run ID')
    run = base/args.run_id
    if run.exists():
        assert_run_identity(run, identity)
    else:
        create_run(base, args.run_id, identity)
        snapshot=run/'source_snapshot'
        for file in source_files():
            destination=snapshot/file.relative_to(ROOT)
            destination.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(str(file),str(destination))
        write_json(run/'source_manifest.json',{p.relative_to(snapshot).as_posix():data_hash(p) for p in sorted(snapshot.rglob('*')) if p.is_file()})
        if source_hash()!=identity['source_sha']: raise ValueError('Source changed during snapshot creation')
    return run, identity


def _settings(model, batch, epochs, hyperparameters):
    settings = dict(model_class='DNN' if model == 'JSU' else model,
                PF_method='JSU' if model == 'JSU' else 'qr',
                pred_horiz=24, target_quantiles=QUANTILES,
                x_columns_names=batch['x_columns_names'], hidden_size=64,
                n_hidden_layers=2, activation='softplus' if model != 'QNBM' else 'relu',
                lr=1e-4, max_epochs=epochs, batch_size=64, patience=50)
    settings.update(hyperparameters)
    return settings


def _runtime(threads):
    os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
    os.environ.setdefault('OMP_NUM_THREADS', str(threads))
    import tensorflow as tf
    if os.environ.get('EPF_REQUIRE_GPU')=='1' and not tf.config.list_physical_devices('GPU'):
        raise RuntimeError('GPU required by the frozen runtime policy')
    try:
        tf.config.threading.set_intra_op_parallelism_threads(threads)
        tf.config.threading.set_inter_op_parallelism_threads(threads)
    except RuntimeError:
        pass
    return tf


@contextlib.contextmanager
def _fit_directory(run):
    previous = Path.cwd()
    scratch=Path(os.environ.get('EPF_SCRATCH_ROOT',str(run))).resolve()
    scratch.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='fit-', dir=str(scratch)) as folder:
        try:
            os.chdir(folder)
            yield
        finally:
            os.chdir(str(previous))


def training_summary(history):
    loss=np.asarray(history.history.get('loss',[]),dtype=float)
    val=np.asarray(history.history.get('val_loss',[]),dtype=float)
    if (not len(loss) or loss.shape!=val.shape or len(history.epoch)!=len(loss)
            or not np.isfinite(loss).all() or not np.isfinite(val).all()):
        raise FloatingPointError('Training history must contain finite matching losses')
    best=int(np.argmin(val))
    return dict(trained_epochs=len(loss),best_epoch=best+1,best_monitored_validation_loss=float(val[best]),
        last_training_loss=float(loss[-1]),last_validation_loss=float(val[-1]))


def _train(run, model_name, batch, hp, epochs, seed, tf):
    from tools.models.models_tools import TensorflowRegressor
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(seed)
    environment_path=Path(os.environ.get('EPF_ENVIRONMENT_RECORD',str(run/'environment.json')))
    if not environment_path.exists():
        import sklearn,optuna,tensorflow_probability
        write_json(environment_path,dict(python=platform.python_version(),platform=platform.platform(),
            
            numpy=np.__version__,pandas=pd.__version__,tensorflow=tf.__version__,
            tensorflow_probability=tensorflow_probability.__version__,sklearn=sklearn.__version__,optuna=optuna.__version__,
            devices=[dict(name=d.name,type=d.device_type) for d in tf.config.list_physical_devices()],
            intra_threads=tf.config.threading.get_intra_op_parallelism_threads(),
            inter_threads=tf.config.threading.get_inter_op_parallelism_threads()))
    settings = _settings(model_name, batch, epochs, hp)
    with _fit_directory(run):
        model = TensorflowRegressor(settings, batch['x_test'])
        started=time.monotonic()
        history=model.fit(batch['train_x'], batch['train_y'], batch['val_x'], batch['val_y'])
        model.training_summary=training_summary(history)
        model.training_summary['monitor_objective']=('raw_quantile_pinball' if model_name=='QNBM' else
            'negative_log_likelihood' if model_name=='JSU' else 'sorted_quantile_pinball')
        model.training_summary['fit_seconds']=time.monotonic()-started
    devices=[v.device for v in model.regressor.model.trainable_variables]
    if os.environ.get('EPF_REQUIRE_GPU')=='1' and (not devices or any('GPU:' not in d for d in devices)):
        raise RuntimeError('Trainable model variables must be on GPU')
    model.training_summary['device']=devices[0] if devices else 'no_trainable_variables'
    telemetry=os.environ.get('EPF_TRAINING_RECORD')
    if telemetry:
        write_json(telemetry,dict(model.training_summary,model=model_name,seed=seed,
            membership=canonical_hash(batch['metadata']),updated_utc=pd.Timestamp.now(tz='UTC').isoformat()))
    return model


def _pinball(y, predictions):
    residual = y[..., None]-predictions
    return float(np.maximum(np.asarray(QUANTILES)*residual,
                            (np.asarray(QUANTILES)-1)*residual).mean())


def audit(args):
    report = dict(study='EPF-QNBM-001', source_sha=source_hash(), markets={},
                  features_release_time='unverified; CSV names are not publication timestamps')
    for market in MARKETS:
        frame = load_market(ROOT.parent, market)
        spec = market_spec(market)
        batch = prepare_day(frame, market, spec['origin'], args.protocol)
        report['markets'][market] = dict(spec=spec, rows=len(frame),
            first_date=frame.Date.iloc[0], last_date=frame.Date.iloc[-1],
            input_shape=list(batch['x_test'].shape), features=batch['x_columns_names'],
            train_days=len(batch['train_y']), validation_days=len(batch['val_y']),
            sha256=data_hash(ROOT.parent/'data'/'datasets'/spec['dataset']))
    if args.run_id:
        run, _ = _open_run(args)
        write_json(run/'audit.json', report)
    return report


def smoke(args, run, deadline):
    tf = _runtime(args.threads)
    outcomes = []
    for market in PILOT_MARKETS:
        batch = prepare_day(load_market(ROOT.parent, market), market, market_spec(market)['origin'], args.protocol)
        # Explicitly marked engineering probes: not the 20-trial/full-epoch run.
        for name in ('DNN', 'QNBM'):
            for seed in SEEDS:
                if time.monotonic() >= deadline:
                    return dict(status='budget_exhausted', purpose='engineering_only', outcomes=outcomes)
                begin = time.monotonic()
                model = _train(run, name, batch, {}, min(args.max_epochs, 2), seed, tf)
                pred = np.asarray(model.predict(batch['x_test']))[:, :, 0, :]
                raw = np.asarray(model.regressor.predict_raw(batch['x_test'])) if name == 'QNBM' else pred
                if not np.isfinite(pred).all():
                    raise ValueError('Non-finite smoke forecast')
                item = dict(market=market, model=name, seed=seed,
                            seconds=time.monotonic()-begin, price_pinball=_pinball(batch['y_test'][None], pred*batch['target_scale']+batch['target_mean']))
                if name == 'QNBM':
                    reference = np.median(model.regressor.build_model_input_from_series(batch['train_x'], batch['x_columns_names'], 24), axis=0, keepdims=True)
                    explanation = price_explanation(model.regressor.explain(batch['x_test'], reference), batch['target_mean'], batch['target_scale'])
                    item['explanation_max_error'] = float(np.max(np.abs(explanation['reference_prediction']+explanation['contributions'].sum(-1)-explanation['raw'])))
                outcomes.append(item)
                write_json(run/'smoke.json', dict(status='running', purpose='engineering_only', outcomes=outcomes))
    return dict(status='complete', purpose='engineering_only', max_epochs=min(args.max_epochs, 2), outcomes=outcomes)


def tune(args, run, deadline):
    import optuna
    tf = _runtime(args.threads)
    from tools.models.models_tools import get_model_class_from_conf
    market, name = args.market, args.model
    spec = market_spec(market)
    batch = prepare_day(load_market(ROOT.parent, market), market, spec['origin'], args.protocol)
    folder = run/market/name
    folder.mkdir(parents=True, exist_ok=True)
    from tools.tuning_state import open_study,checkpoint_sampler,close_study
    tuning_identity=canonical_hash(dict(run=read_json(run/'identity.json')['sha256'],
        market=market,model=name,validation=canonical_hash(batch['metadata'])))
    study = open_study(folder,tuning_identity)
    try:
        model_class = get_model_class_from_conf('DNN' if name == 'JSU' else name)
        complete = sum(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials)
        if any(t.state == optuna.trial.TrialState.RUNNING for t in study.trials):
            raise ValueError('Interrupted or active trial exists; inspect it before resuming')
        def objective(trial):
            if time.monotonic() >= deadline:
                raise optuna.TrialPruned('Wall-clock budget reached')
            settings = model_class.get_hyperparams_trial(trial, {})
            model = _train(run, name, batch, settings, args.max_epochs, SEEDS[0], tf)
            pred = np.asarray(model.predict(batch['val_x']))
            if name == 'JSU':
                from tools.models.models_tools import Ensemble
                pred = Ensemble(model.settings).get_preds_test_quantiles(pred).reshape(-1,24,9)
            else:
                pred = pred[:, :, 0, :]
            return _pinball(batch['val_y'], pred)
        if len(study.trials)>20: raise ValueError('Trial budget was exceeded')
        if len(study.trials) < 20:
            study.optimize(objective, n_trials=20-len(study.trials),
                           timeout=None if not np.isfinite(deadline) else max(0.1, deadline-time.monotonic()),
                           catch=(ValueError,FloatingPointError),callbacks=[checkpoint_sampler(folder,tuning_identity)])
        complete = sum(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials)
        result = dict(status='complete' if len(study.trials)==20 and complete>0 else 'failed_trials' if len(study.trials)>=20 else 'budget_exhausted',
                      complete_trials=complete,attempted_trials=len(study.trials),
                      origin=spec['origin'], latest_label=max(batch['metadata']['train_dates']+batch['metadata']['validation_dates']),
                      best_parameters=study.best_trial.params if complete else None,
                      best_validation_pinball=study.best_value if complete else None,
                      validation_identity=canonical_hash(batch['metadata']), protocol=args.protocol)
        write_json(folder/'tuned.json', result)
        return result
    finally:
        close_study(study)


def forecast(args, run, identity, deadline):
    tf = _runtime(args.threads)
    spec = market_spec(args.market)
    folder = run/args.market/args.model
    tuning = read_json(folder/'tuned.json')
    if tuning['status'] != 'complete' or tuning.get('attempted_trials') != 20 or tuning['complete_trials']<1:
        raise ValueError('Formal forecasts require the full 20-attempt budget and a valid result')
    frozen = (run/'selection.json').exists()
    end = spec['screen_end'] if args.stage == 'development' else spec['test_end']
    start = spec['origin'] if args.stage == 'development' else spec['test_start']
    if args.stage == 'evaluation' and not frozen:
        raise ValueError('Locked evaluation requires a frozen selection')
    if args.stage == 'evaluation': _verify_selection(run,args.market)
    frame = load_market(ROOT.parent, args.market)
    dates = pd.date_range(start, end)
    dates = partition_dates(dates,getattr(args,'shard_index',0),getattr(args,'shards',1))
    output = folder/'daily'
    output.mkdir(exist_ok=True)
    completed = 0
    # Fixed raw reference window obtained only from pre-origin training inputs.
    initial = prepare_day(frame, args.market, spec['origin'], args.protocol)
    reference_window = np.median(initial['train_x'], axis=0, keepdims=True).astype(np.float64)
    reference_window[:, :, 0] = reference_window[:, :, 0]*initial['target_scale']+initial['target_mean']
    reference_window[:, :, 1:] = initial['feature_scaler'].inverse_transform(reference_window[0, :, 1:])[None]
    from tools.models.DNN import DNNRegressor
    for day in dates:
        ds = str(day.date())
        require_stage(args.market, ds, args.stage, frozen=frozen)
        batch = None
        for seed in SEEDS:
            dest = output/(ds+'_'+str(seed)+'.npz')
            if dest.exists():
                with np.load(dest, allow_pickle=False) as saved:
                    if str(saved['identity']) != canonical_hash(forecast_identity(args.market,args.model,args.protocol,args.max_epochs,identity['data_sha'][args.market],identity['source_sha'],tuning['best_parameters'],identity.get('runtime_policy_sha'))):
                        raise ValueError('Cached prediction identity mismatch')
                continue
            if time.monotonic() >= deadline or completed >= args.days*len(SEEDS):
                return dict(status='budget_exhausted', new_member_days=completed, next_day=ds)
            if batch is None:
                batch = prepare_day(frame, args.market, ds, args.protocol)
            begin = time.monotonic()
            model = _train(run,args.model,batch,tuning['best_parameters'],args.max_epochs,seed+day.toordinal(),tf)
            if args.model == 'JSU':
                from tools.models.models_tools import Ensemble
                pred = Ensemble(model.settings).get_preds_test_quantiles(model.predict(batch['x_test'])).reshape(1,24,9)
                raw = pred
            else:
                pred = np.asarray(model.predict(batch['x_test']))[:, :, 0, :]
                raw = np.asarray(model.regressor.predict_raw(batch['x_test'])) if args.model == 'QNBM' else pred
            raw = raw*batch['target_scale']+batch['target_mean']
            pred = pred*batch['target_scale']+batch['target_mean']
            if not np.isfinite(pred).all() or not np.isfinite(raw).all():
                raise ValueError('Non-finite daily predictions')
            payload=dict(predictions=pred[0],raw=raw[0],targets=batch['y_test'],
                day=ds,seed=seed,seconds=time.monotonic()-begin,
                actual_seed=seed+day.toordinal(),
                identity=canonical_hash(forecast_identity(args.market,args.model,args.protocol,args.max_epochs,identity['data_sha'][args.market],identity['source_sha'],tuning['best_parameters'],identity.get('runtime_policy_sha'))),
                data_membership=canonical_hash(batch['metadata']))
            payload.update(model.training_summary)
            if args.model == 'QNBM' and ds in {spec['origin'],spec['tune_start'],spec['screen_start'],spec['screen_end'],spec['test_start'],spec['test_end']}:
                ref = reference_window.copy()
                ref[:, :, 0] = (ref[:,:,0]-batch['target_mean'])/batch['target_scale']
                ref[:, :, 1:] = batch['feature_scaler'].transform(ref[0,:,1:])[None]
                flat = DNNRegressor.build_model_input_from_series(ref,batch['x_columns_names'],24)
                explanation = price_explanation(model.regressor.explain(batch['x_test'],flat),batch['target_mean'],batch['target_scale'])
                payload.update({'explanation_'+key:np.asarray(value) for key,value in explanation.items()})
            temporary = dest.with_suffix('.tmp.npz')
            np.savez_compressed(temporary,**payload)
            temporary.replace(dest)
            completed += 1
            progress=Path(getattr(args,'progress_path',None) or folder/'forecast_status.json')
            write_json(progress,dict(status='running',latest_day=ds,seed=seed,new_member_days=completed,
                shard_index=getattr(args,'shard_index',0),shards=getattr(args,'shards',1),
                training=model.training_summary))
    return dict(status='complete', new_member_days=completed, stage=args.stage)


def _series(run, market, name, include_test=False):
    spec = market_spec(market)
    end = spec['test_end'] if include_test else spec['screen_end']
    days = pd.date_range(spec['origin'],end)
    root_identity=read_json(run/'identity.json')['identity']
    hp=read_json(run/market/name/'tuned.json')['best_parameters']
    expected_identity=canonical_hash(forecast_identity(market,name,root_identity['protocol'],root_identity['max_epochs'],root_identity['data_sha'][market],root_identity['source_sha'],hp,root_identity.get('runtime_policy_sha')))
    preds, raw, labels = [], [], []
    for day in days:
        ds = str(day.date())
        members = []
        raw_members = []
        expected_target = None
        membership = None
        for seed in SEEDS:
            with np.load(run/market/name/'daily'/(ds+'_'+str(seed)+'.npz'),allow_pickle=False) as saved:
                if str(saved['identity']) != expected_identity or str(saved['day'])!=ds or int(saved['seed'])!=seed:
                    raise ValueError('Daily artifact identity/date/seed mismatch')
                if saved['predictions'].shape!=(24,9) or saved['raw'].shape!=(24,9) or saved['targets'].shape!=(24,):
                    raise ValueError('Invalid daily artifact shape')
                if not all(np.isfinite(saved[k]).all() for k in ['predictions','raw','targets']):
                    raise ValueError('Non-finite daily artifact')
                if expected_target is None:
                    expected_target=saved['targets'].copy(); membership=str(saved['data_membership'])
                elif not np.array_equal(saved['targets'],expected_target) or str(saved['data_membership']) != membership:
                    raise ValueError('Ensemble targets or validation memberships differ')
                members.append(saved['predictions']); raw_members.append(saved['raw'])
        preds.append(np.mean(members,axis=0)); raw.append(np.mean(raw_members,axis=0)); labels.append(expected_target)
    return days,np.asarray(preds),np.asarray(raw),np.asarray(labels)


def _volatility(frame, market, dates, tune_mask):
    from tools.research_evaluation import past_volatility
    prices=frame['TARG__'+market].to_numpy().reshape(-1,24)
    all_days=pd.DatetimeIndex(pd.to_datetime(frame.Date.iloc[::24]))
    volatility=past_volatility(prices)[all_days.get_indexer(dates)]
    threshold=float(np.quantile(volatility[tune_mask],0.8))
    return volatility >= threshold, threshold


def select_controller(pred,y,warmup,tune_mask,mode,grid):
    """Only the declared tuning slice can influence ranking or rejection."""
    from tools.online_calibration import calibrate_quantiles
    from tools.research_evaluation import evaluate_quantiles
    from tools.waci import waci_quantiles
    if not np.any(tune_mask): raise ValueError('No calibration tuning observations')
    tune_end=int(np.flatnonzero(tune_mask)[-1])+1
    mask=np.asarray(tune_mask[:tune_end],dtype=bool)
    best=None; attempted=[]
    for options in grid:
        try:
            if mode=='waci':
                result=waci_quantiles(pred[:tune_end],y[:tune_end],QUANTILES,warmup=warmup,options=options)
            else:
                result=calibrate_quantiles(pred[:tune_end],y[:tune_end],QUANTILES,
                    warmup=warmup,mode=mode,options=options)
            if not np.isfinite(result['final'][mask]).all(): raise ValueError('Non-finite tuning intervals')
        except ValueError as error:
            attempted.append(dict(options=options,status='failed',error=str(error)))
            continue
        metric=evaluate_quantiles(y[:tune_end][mask],result['final'][mask],QUANTILES)
        coverage=metric['intervals']['0.2']['coverage_error']
        score=(coverage>0.01,metric['pinball'])
        if best is None or score<best[0]: best=(score,dict(options))
        attempted.append(dict(options=options,status='complete',coverage_error=coverage,pinball=metric['pinball']))
    if best is None: raise ValueError('All calibration candidates failed')
    return best[1],attempted


def calibrate(args, run, include_test=False):
    from tools.online_calibration import calibrate_quantiles
    from tools.research_evaluation import evaluate_quantiles
    market=args.market; spec=market_spec(market)
    if include_test: _verify_selection(run,market)
    days,pred,raw,y=_series(run,market,args.model,include_test)
    warmup=int((pd.Timestamp(spec['tune_start'])-days[0]).days)
    tune_mask=(days>=spec['tune_start']) & (days<=spec['tune_end'])
    modes=['original','constant','normalized','waci']
    folder=run/market/args.model/'calibration'
    folder.mkdir(exist_ok=True)
    choices_path=folder/'parameters.json'
    choices=read_json(choices_path) if choices_path.exists() else {}
    if include_test and set(choices)!=set(modes):
        raise ValueError('All calibration parameters must be frozen before evaluation')
    # Same 20-entry grid for original and normalized. Constant is an exact
    # unit-change ablation of the selected original controller.
    grid=[dict(lr=lr,KI=ki) for lr in [.001,.003,.01,.03,.1] for ki in [0.,1.,3.,10.]]
    summary={}
    for mode in modes:
        if mode not in choices:
            if include_test:
                raise ValueError('Evaluation must use development-frozen calibration parameters')
            trials=([dict(gamma=gamma,sigma=sigma) for gamma in [.001,.003,.01,.03,.1] for sigma in [1.,3.,5.,10.]]
                    if mode=='waci' else grid if mode!='constant' else [choices['original']])
            selected,attempted=select_controller(pred,y,warmup,tune_mask,mode,trials)
            write_json(folder/(mode+'_tuning.json'),dict(period=[spec['tune_start'],spec['tune_end']],trials=attempted))
            choices[mode]=selected
            write_json(choices_path,choices)
    context=dict(identity=read_json(run/'identity.json')['sha256'],parameters_hash=data_hash(choices_path),
                 inputs_hash=series_digest(days,pred,y))
    for mode in modes:
        if mode=='waci':
            from tools.waci import waci_quantiles
            result=waci_quantiles(pred,y,QUANTILES,warmup=warmup,options=choices[mode])
        else:
            result=calibrate_quantiles(pred,y,QUANTILES,warmup=warmup,mode=mode,options=choices[mode])
        save_calibration(folder/(mode+('_test' if include_test else '')+'.npz'),
            {k:v for k,v in result.items() if isinstance(v,np.ndarray)},result['metadata'],**context)
        summary[mode]=choices[mode]
    # Fixed rolling asymmetric CQR is a separate baseline; copies prevent the
    # historical in-place mutation from altering subsequent calibration bags.
    from tools.prediction_quantiles_tools import exec_cqr,build_alpha_quantiles_map
    alphas=[.2,.4,.6,.8]
    cqr=np.sort(pred,axis=-1).copy()
    for day in range(warmup,len(pred)):
        cqr[day]=exec_cqr(pred[day-warmup:day],y[day-warmup:day],pred[day:day+1],
            dict(target_alpha=alphas,q_alpha_map=build_alpha_quantiles_map(alphas,QUANTILES))).reshape(24,9)
    save_calibration(folder/('cqr'+('_test' if include_test else '')+'.npz'),{'final':cqr},
                     dict(method='rolling_asymmetric_cqr',warmup=warmup,window_days=182),**context)
    return dict(status='complete',parameters=summary)


def screen(args, run):
    from tools.research_evaluation import evaluate_quantiles
    spec=market_spec(args.market)
    frame=load_market(ROOT.parent,args.market)
    days,dnn,_,y=_series(run,args.market,'DNN')
    other_days,qnbm,_,other_y=_series(run,args.market,'QNBM')
    if not days.equals(other_days) or not np.array_equal(y,other_y): raise ValueError('Model dates/targets differ')
    tune_mask=(days>=spec['tune_start']) & (days<=spec['tune_end'])
    mask=(days>=spec['screen_start']) & (days<=spec['screen_end'])
    high,threshold=_volatility(frame,args.market,days,tune_mask)
    metrics={}
    for name,base in [('DNN',dnn),('QNBM',qnbm)]:
        metrics[name]=evaluate_quantiles(y[mask],base[mask],QUANTILES,high[mask])
        folder=run/args.market/name/'calibration'
        context=dict(identity=read_json(run/'identity.json')['sha256'],parameters_hash=data_hash(folder/'parameters.json'),
                     inputs_hash=series_digest(days,base,y))
        for mode in ['original','normalized','constant','cqr','waci']:
            saved=load_calibration(folder/(mode+'.npz'),**context)
            if np.isfinite(saved['final'][mask]).all():
                metrics[name+'_'+mode]=evaluate_quantiles(y[mask],saved['final'][mask],QUANTILES,high[mask])
            else:
                metrics[name+'_'+mode]=dict(status='non_finite_intervals',
                    nonfinite_values=int((~np.isfinite(saved['final'][mask])).sum()),
                    note='No finite clipping or case deletion; scores unavailable')
    reference=metrics['DNN_original']; candidate=metrics['QNBM_original']; normalized=metrics['QNBM_normalized']
    from tools.research_evaluation import screening_decision
    decision=screening_decision(candidate,reference,normalized)
    replacement=decision['main_gate']['status']=='pass'
    normalized_pass=(decision['normalization_gate']['status']=='pass' and
                     screening_decision(normalized,reference)['main_gate']['status']=='pass')
    result=dict(market=args.market,status='complete',period=[spec['screen_start'],spec['screen_end']],
        replacement_pass=bool(replacement),normalized_pass=normalized_pass,volatility_threshold=threshold,
        metrics=metrics,decision=decision,identity=read_json(run/'identity.json')['sha256'],
        parameter_hashes=_parameter_hashes(run,args.market),
        artifact_hashes={name:evidence_hashes(run/args.market/name/'calibration') for name in ['DNN','QNBM']},
        purpose='development_point_estimates_only')
    write_json(run/args.market/'screen.json',result)
    return {k:v for k,v in result.items() if k!='metrics'}


def evaluate(args, run):
    from tools.research_evaluation import evaluate_quantiles,paired_block_bootstrap
    selection=_verify_selection(run,args.market); spec=market_spec(args.market)
    days,dnn,_,y=_series(run,args.market,'DNN',True)
    other_days,qnbm,_,other_y=_series(run,args.market,'QNBM',True)
    if not days.equals(other_days) or not np.array_equal(y,other_y): raise ValueError('Locked model dates/targets differ')
    mask=(days>=spec['test_start']) & (days<=spec['test_end'])
    frame=load_market(ROOT.parent,args.market)
    from tools.research_evaluation import past_volatility
    vd=past_volatility(frame['TARG__'+args.market].to_numpy().reshape(-1,24))
    frame_days=pd.DatetimeIndex(pd.to_datetime(frame.Date.iloc[::24]))
    threshold=(selection['screens'][args.market]['volatility_threshold'] if args.market in selection['screens'] else
               read_json(run/args.market/'expansion_frozen.json')['volatility_threshold'])
    high=vd[frame_days.get_indexer(days)]>=threshold
    def artifact(name,base,mode):
        folder=run/args.market/name/'calibration'
        return load_calibration(folder/(mode+'_test.npz'),identity=read_json(run/'identity.json')['sha256'],
            parameters_hash=data_hash(folder/'parameters.json'),inputs_hash=series_digest(days,base,y))['final'][mask]
    reference=artifact('DNN',dnn,'original')
    candidate=artifact('QNBM',qnbm,selection['method'])
    backbone=artifact('QNBM',qnbm,'original')
    uncertainty=paired_block_bootstrap(y[mask],candidate,reference,QUANTILES,high[mask])
    from tools.research_evaluation import locked_evaluation_decision
    backbone_uncertainty=paired_block_bootstrap(y[mask],backbone,reference,QUANTILES,high[mask])
    normalization_uncertainty=(paired_block_bootstrap(y[mask],candidate,backbone,QUANTILES,high[mask])
                               if selection['method']=='normalized' else None)
    result=dict(period=[spec['test_start'],spec['test_end']],method=selection['method'],
        candidate=evaluate_quantiles(y[mask],candidate,QUANTILES,high[mask]),
        reference=evaluate_quantiles(y[mask],reference,QUANTILES,high[mask]),
        uncertainty=uncertainty,
        backbone_uncertainty=backbone_uncertainty, normalization_uncertainty=normalization_uncertainty,
        decision=locked_evaluation_decision(uncertainty,normalization_uncertainty),
        backbone_decision=locked_evaluation_decision(backbone_uncertainty),
        warning='Locked evaluation is not a method-selection input')
    write_json(run/args.market/'locked_evaluation.json',result)
    return result


def develop(args,run,identity,deadline):
    """Pilot development only; no locked dates are launched by this worker."""
    from types import SimpleNamespace
    events=[]
    status_path=run/'development_progress.json'
    def record(event):
        events.append(event)
        write_json(status_path,dict(status='running',events=events,locked_evaluation_started=False))
    for market in PILOT_MARKETS:
        for model in ['DNN','QNBM']:
            step=SimpleNamespace(**vars(args)); step.market=market; step.model=model; step.stage='development'
            if time.monotonic()>=deadline: return dict(status='budget_exhausted',events=events)
            result=tune(step,run,deadline)
            record(dict(phase='network_tuning',market=market,model=model,result=result))
            if result['status']!='complete': return dict(status=result['status'],events=events)
    remaining={(m,n) for m in PILOT_MARKETS for n in ['DNN','QNBM']}
    while remaining:
        for market,model in sorted(remaining):
            if time.monotonic()>=deadline: return dict(status='budget_exhausted',events=events)
            step=SimpleNamespace(**vars(args)); step.market=market; step.model=model; step.stage='development'
            result=forecast(step,run,identity,deadline)
            record(dict(phase='rolling_forecasts',market=market,model=model,result=result))
            if result['status']=='complete': remaining.remove((market,model))
    screens={}
    for market in PILOT_MARKETS:
        step=SimpleNamespace(**vars(args)); step.market=market
        for model in ['DNN','QNBM']:
            if time.monotonic()>=deadline: return dict(status='budget_exhausted',events=events)
            step.model=model
            record(dict(phase='calibration',market=market,model=model,result=calibrate(step,run)))
        record(dict(phase='screening',market=market,result=screen(step,run)))
        screens[market]=read_json(run/market/'screen.json')
    if not all(item['replacement_pass'] for item in screens.values()):
        return dict(status='stopped_backbone_gate',events=events,
                    note='Pilot replacement did not pass; no expansion or locked evaluation')
    selection=(_verify_selection(run) if (run/'selection.json').exists() else freeze_selection(run,screens))
    return dict(status='development_complete',method=selection['method'],events=events,
                locked_evaluation_started=False,note='Frozen pilot method; locked evaluation and expansion remain separate stages')


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['audit','smoke','tune','forecast','calibrate','screen','freeze','evaluate','develop'])
    parser.add_argument('--run-id')
    parser.add_argument('--market',choices=MARKETS,default='DE_price')
    parser.add_argument('--model',choices=['DNN','QNBM','JSU'],default='QNBM')
    parser.add_argument('--protocol',choices=['legacy','chronological'],default='legacy')
    parser.add_argument('--stage',choices=['development','evaluation'],default='development')
    parser.add_argument('--max-epochs',type=int,default=800)
    parser.add_argument('--budget-seconds',type=float,default=None,help='Only set an explicit wall budget when requested; default has no time limit')
    parser.add_argument('--days',type=int,default=1,help='Maximum new forecast days per bounded invocation')
    parser.add_argument('--threads',type=int,default=2)
    parser.add_argument('--shards',type=int,default=1)
    parser.add_argument('--shard-index',type=int,default=0)
    parser.add_argument('--runtime-policy')
    parser.add_argument('--output-root',default=str(ROOT/'experiments'/'qnbm'))
    args=parser.parse_args(argv)
    if min(args.max_epochs,args.days,args.threads)<=0: parser.error('Epochs, chunks and threads must be positive')
    if args.shards<1 or not 0<=args.shard_index<args.shards:parser.error('Invalid date partition')
    if args.budget_seconds is not None and (not np.isfinite(args.budget_seconds) or args.budget_seconds<0):
        parser.error('Optional wall budget must be finite and nonnegative')
    start=time.monotonic(); deadline=(float('inf') if args.budget_seconds in (None,0) else start+args.budget_seconds)
    if args.command=='audit':
        result=audit(args)
    else:
        run,identity=_open_run(args)
        with experiment_lock(run):
            status_path=run/(args.command+'_'+args.market+'_'+args.model+'_status.json')
            write_json(status_path,dict(status='running',command=vars(args),started_utc=pd.Timestamp.now(tz='UTC').isoformat(),
                python=platform.python_version(),source_sha=identity['source_sha']))
            try:
                if args.command=='smoke': result=smoke(args,run,deadline); write_json(run/'smoke.json',result)
                elif args.command=='tune': result=tune(args,run,deadline)
                elif args.command=='develop': result=develop(args,run,identity,deadline)
                elif args.command=='forecast': result=forecast(args,run,identity,deadline)
                elif args.command=='calibrate':
                    if args.stage=='evaluation' and not (run/'selection.json').exists(): raise ValueError('Selection not frozen')
                    result=calibrate(args,run,args.stage=='evaluation')
                elif args.command=='screen': result=screen(args,run)
                elif args.command=='freeze':
                    if not (run/'selection.json').exists():
                        screens={m:read_json(run/m/'screen.json') for m in PILOT_MARKETS if (run/m/'screen.json').exists()}
                        result=freeze_selection(run,screens)
                    else:
                        _verify_selection(run)
                        if args.market in PILOT_MARKETS: raise ValueError('Pilot selection is already frozen')
                        record=read_json(run/args.market/'screen.json')
                        if record['identity']!=read_json(run/'identity.json')['sha256']: raise ValueError('Expansion screening identity mismatch')
                        _verify_screen(run,args.market,record)
                        dest=run/args.market/'expansion_frozen.json'
                        if dest.exists(): raise FileExistsError('Expansion is already frozen')
                        result=dict(identity=record['identity'],volatility_threshold=record['volatility_threshold'],
                            calibration_hashes=_parameter_hashes(run,args.market),method=_verify_selection(run)['method'],
                            note='Threshold/parameters frozen; global method remains unchanged')
                        write_json(dest,result)
                else: result=evaluate(args,run)
                write_json(status_path,dict(status=result.get('status','complete'),elapsed_seconds=time.monotonic()-start,result=result))
            except BaseException as error:
                write_json(status_path,dict(status='failed',elapsed_seconds=time.monotonic()-start,error=repr(error)))
                raise
    print(json.dumps(result,ensure_ascii=False,allow_nan=False,indent=2))
    return result


if __name__=='__main__':
    main()
