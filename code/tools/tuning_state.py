"""Persist TPE RNG together with trial history; never silently reseed a study.

Sampler pickles are created only in this run's own directory. They are loaded
only after identity, Optuna/Python version, history and byte hashes agree.
Missing/interrupted checkpoints fail closed rather than repeating proposals.
"""
import json
import pickle
from pathlib import Path
import platform

import optuna

from tools.research_protocol import canonical_hash,data_hash,write_json


def _history(study):
    return canonical_hash([dict(number=t.number,state=t.state.name,params=t.params,values=t.values)
                           for t in study.get_trials(deepcopy=False)])


def close_study(study):
    """Release this study's RDB connections (also needed on Windows)."""
    storage=study._storage
    storage.remove_session()
    backend=getattr(storage,'_backend',storage)
    backend.engine.dispose()


def open_study(folder,identity):
    folder=Path(folder); folder.mkdir(parents=True,exist_ok=True)
    study=optuna.create_study(study_name='network',storage='sqlite:///'+(folder/'tuning.sqlite3').as_posix(),
        direction='minimize',load_if_exists=True,sampler=optuna.samplers.TPESampler(seed=2026))
    try:
        if any(t.state==optuna.trial.TrialState.RUNNING for t in study.trials):
            raise ValueError('Interrupted or active trial: inspect before resuming')
        state=folder/'sampler.pkl'; metadata=folder/'sampler.json'
        if not study.trials:
            if state.exists() or metadata.exists(): raise ValueError('Orphan sampler checkpoint')
            return study
        if not state.is_file() or not metadata.is_file():
            raise ValueError('Sampler state missing; do not restart TPE with the initial seed')
        record=json.loads(metadata.read_text(encoding='utf-8'))
        expected=dict(identity=identity,history=_history(study),optuna=optuna.__version__,
                      python='.'.join(platform.python_version_tuple()[:2]),sha256=data_hash(state))
        if any(record.get(k)!=v for k,v in expected.items()):
            raise ValueError('Sampler checkpoint identity/history/version/hash mismatch')
        with state.open('rb') as stream: sampler=pickle.load(stream)
        if not isinstance(sampler,optuna.samplers.TPESampler): raise ValueError('Unexpected sampler checkpoint type')
        study.sampler=sampler
        return study
    except BaseException:
        close_study(study)
        raise


def checkpoint_sampler(folder,identity):
    folder=Path(folder)
    def callback(study,trial):
        temporary=folder/'sampler.tmp'
        with temporary.open('wb') as stream: pickle.dump(study.sampler,stream,protocol=pickle.HIGHEST_PROTOCOL)
        target=folder/'sampler.pkl'; temporary.replace(target)
        write_json(folder/'sampler.json',dict(identity=identity,history=_history(study),
            optuna=optuna.__version__,python='.'.join(platform.python_version_tuple()[:2]),
            sha256=data_hash(target),last_trial=trial.number))
    return callback
