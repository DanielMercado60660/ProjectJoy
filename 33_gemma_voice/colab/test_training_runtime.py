"""CPU-only checks of frozen identity and session-independent posterior sampling."""
import ast
import json
import glob
import hashlib
import os
from pathlib import Path
import numpy as np
import pytest


def functions(*names):
    path = Path(__file__).with_name('training_runtime.py')
    tree = ast.parse(path.read_text())
    selected = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.Assign)) and
                (getattr(n, 'name', None) in names or
                 isinstance(n, ast.Assign) and any(getattr(t, 'id', '') in ('TRAINING_VERSION','TRAINING_KEYS') for t in n.targets))]
    ns = dict(json=json, np=np, os=os, glob=glob, hashlib=hashlib)
    exec(compile(ast.Module(body=selected,type_ignores=[]), str(path), 'exec'), ns)
    return ns


def identity_fixture(tmp_path):
    ns = functions('stable_hash','frozen_data')
    cfg = {k: None for k in ns['TRAINING_KEYS']}
    cfg.update(CODEC='ming', SETS=['lj'], TRAIN_EPOCHS=2, BUDGET=6000, SEED=7)
    data = tmp_path/'data'/'ming'/'lj'; data.mkdir(parents=True)
    (data/'part.json').write_text('{"version":1}')
    (data/'part.f16').write_bytes(b'1234')
    ck = tmp_path/'ckpt'; ck.mkdir()
    ns.update(CFG=cfg, WORK=str(tmp_path), ckpt_dir=lambda:str(ck),
        load_data=lambda tok:([((1,2),0,4,0)],[],{},None,{'lj:lj':0},np.zeros(2),np.ones(2)),
        batches_for_epoch=lambda lens,budget,seed,ep:[[0]])
    class Tokenizer:
        bos_token_id=2
        vocab={'hello':1}
        def get_vocab(self): return self.vocab
    return ns, Tokenizer(), data


def test_identical_resume_identity(tmp_path):
    ns,tok,_=identity_fixture(tmp_path)
    a=ns['frozen_data'](tok)
    b=ns['frozen_data'](tok)
    assert a['identity']==b['identity'] and ns['CFG']['STEPS']==2


@pytest.mark.parametrize('change',['metadata','audio_size','batch','vocabulary'])
def test_resume_rejects_changed_contract(tmp_path,change):
    ns,tok,data=identity_fixture(tmp_path)
    ns['frozen_data'](tok)
    if change=='metadata': (data/'part.json').write_text('{"version":2}')
    elif change=='audio_size': (data/'part.f16').write_bytes(b'12345')
    elif change=='batch': ns['CFG']['BUDGET']=9000
    else: tok.vocab={'hello':2}
    with pytest.raises(AssertionError,match='identity changed'):
        ns['frozen_data'](tok)


def test_latent_sample_matches_after_new_session_and_epoch_boundary():
    ns=functions('next_training_batch')
    ns.update(CFG={'TRAIN_EPOCHS':3,'BUDGET':6000,'SEED':41},
        batches_for_epoch=lambda lens,budget,seed,ep: [[1],[0],[2]] if ep==0 else [[2],[1]],
        make_batch=lambda items,store,D,mu,sd,rng:(items,rng.standard_normal(20)))
    class Model: D=64
    data=dict(items=['a','b','c'],lens=[1,2,3],store=None,mu=None,sd=None)
    ctx=dict(data=data,model=Model(),epoch=1,batch=1,step=4,order_epoch=None)
    a=ns['next_training_batch'](ctx)
    fresh=dict(ctx,order=None,order_epoch=None)
    b=ns['next_training_batch'](fresh)
    assert a[0]==b[0]==['b']
    np.testing.assert_array_equal(a[1],b[1])
    fresh['step']+=1
    assert not np.array_equal(a[1],ns['next_training_batch'](fresh)[1])
