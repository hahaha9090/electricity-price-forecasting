"""Bind scientific caches to their inputs, parameters and immutable run ID."""
import hashlib
import contextlib
import json
import os
import socket
import uuid
from pathlib import Path

import numpy as np

from tools.research_protocol import data_hash, canonical_hash, write_json


@contextlib.contextmanager
def experiment_lock(run):
    """Never infer that a previous experiment died merely from elapsed time."""
    path=Path(run)/'.active.lock'
    token=uuid.uuid4().hex
    payload=dict(pid=os.getpid(),host=socket.gethostname(),token=token)
    with path.open('x',encoding='utf-8') as stream:
        json.dump(payload,stream)
    try:
        yield
    finally:
        if path.exists() and _read(path).get('token')==token:
            path.unlink()


def series_digest(dates, predictions, targets):
    parts = {'dates': [str(x) for x in dates]}
    for key, value in [('predictions',predictions),('targets',targets)]:
        array=np.ascontiguousarray(value,dtype=np.float64)
        parts[key]=dict(shape=list(array.shape),sha256=hashlib.sha256(array.tobytes()).hexdigest())
    return canonical_hash(parts)


def _read(path):
    import json
    return json.loads(path.read_text(encoding='utf-8'))


def save_calibration(path, arrays, metadata, identity, parameters_hash, inputs_hash):
    path=Path(path)
    context=dict(identity=identity,parameters_hash=parameters_hash,inputs_hash=inputs_hash)
    if path.exists():
        existing=load_calibration(path,**context)
        if set(existing)!=set(arrays) or any(not np.array_equal(existing[k],arrays[k],equal_nan=True) for k in arrays):
            raise ValueError('Existing scientific result differs; use a new run ID')
        return
    if path.with_suffix('.artifact.json').exists():
        raise ValueError('Orphan artifact metadata; inspect before resuming')
    temporary=path.with_suffix('.tmp.npz')
    np.savez_compressed(temporary,**arrays)
    temporary.replace(path)
    write_json(path.with_suffix('.artifact.json'),dict(context,sha256=data_hash(path),metadata=metadata))


def load_calibration(path, identity, parameters_hash, inputs_hash):
    path=Path(path)
    record=_read(path.with_suffix('.artifact.json'))
    expected=dict(identity=identity,parameters_hash=parameters_hash,inputs_hash=inputs_hash)
    if any(record.get(k)!=v for k,v in expected.items()) or record['sha256']!=data_hash(path):
        raise ValueError('Calibration artifact identity/parameters/inputs/bytes mismatch')
    with np.load(path,allow_pickle=False) as archive:
        return {k:archive[k].copy() for k in archive.files}


def evidence_hashes(root):
    root=Path(root)
    return {p.relative_to(root).as_posix():data_hash(p) for p in sorted(root.rglob('*'))
            if p.is_file() and (p.suffix=='.json' or p.suffix=='.npz')}


def verify_evidence(root, hashes):
    root=Path(root)
    if not hashes:
        raise ValueError('Screening evidence must not be empty')
    for relative,expected in hashes.items():
        target=(root/relative).resolve()
        if root.resolve() not in target.parents or not target.is_file() or data_hash(target)!=expected:
            raise ValueError('Screening evidence changed after scoring: '+relative)
