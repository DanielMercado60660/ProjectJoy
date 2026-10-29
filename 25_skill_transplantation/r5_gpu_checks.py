"""Runner-only GPU smoke, including SIGKILL/restart. Never run in Astra's sandbox."""
import os
from pathlib import Path
import subprocess
import sys
import time

for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[name] = '1'

import numpy as np
import mlx.core as mx
import mlx.optimizers as optim

from r5_core import (Deadline, add_adapter, deployment_params, forward, init_params, inputs, read_pickle,
                     train_keys, write_json)
from r5_train import safe_clip, train


def main():
    started = time.monotonic()
    deadline = Deadline(1.85)
    tag = 'smoke_' + time.strftime('%Y%m%d_%H%M%S') + '_' + str(os.getpid())
    ck, res, logs = Path('checkpoints')/f'r5_{tag}', Path('results/r5')/tag, Path('logs/r5')/tag
    ck.mkdir(parents=True)
    res.mkdir(parents=True)
    logs.mkdir(parents=True)
    rng = np.random.default_rng(8)
    p = add_adapter(init_params(0,16),1,2,1)
    ids = np.arange(12,dtype=np.int32)%2
    x = inputs(rng.normal(size=(12,17)),ids)
    mp = {k:mx.array(v) for k,v in p.items()}
    # Exercise nonzero routing, not just the easy zero-B identity.
    mp['d1_a2B'] = mx.array(rng.normal(size=p['d1_a2B'].shape).astype(np.float32)*.01)
    p['d1_a2B'] = np.array(mp['d1_a2B'])
    for value in (False,True):
        np.testing.assert_allclose(np.array(forward(mp,mx.array(x),mx.array(ids),mx,value)),
                                   forward(p,x,ids,value=value),atol=5e-3,rtol=1e-3)
    zero = {'g':mx.zeros((3,4))}
    clipped = safe_clip(zero)
    assert np.isfinite(np.array(clipped['g'])).all()
    zero_grad = mx.grad(lambda z: mx.sqrt(mx.sum(z*z)+1e-12))(mx.zeros((3,)))
    assert np.isfinite(np.array(zero_grad)).all()
    # Frozen adapter leaves, changed active route, finite zero-target MSE gradients.
    keys = train_keys(p,'adapter',1)
    variables = {k:mp[k] for k in keys}
    frozen = {k:v for k,v in mp.items() if k not in keys}
    xx, ii = mx.array(x), mx.array(ids)
    def loss(v):
        return mx.mean((forward(dict(frozen,**v),xx,ii,mx)-.2)**2)
    grad = mx.value_and_grad(loss)
    opt = optim.Adam(learning_rate=.01)
    first = float(loss(variables))
    for _ in range(12):
        l,g = grad(variables)
        assert all(np.isfinite(np.array(v)).all() for v in g.values())
        variables = opt.apply_gradients(safe_clip(g),variables)
        mx.eval(variables,opt.state,l)
    last = float(loss(variables))
    assert last < first, (first,last)
    assert all(np.array_equal(np.array(frozen[k]),p[k]) for k in frozen)
    old = np.array(forward(mp,xx,ii,mx))[ids==0]
    new = np.array(forward(dict(frozen,**variables),xx,ii,mx))[ids==0]
    np.testing.assert_array_equal(old,new)
    print(f'R5_GPU_ALGEBRA_OK finite zero gradients; fixed-batch MSE {first:.6f} -> {last:.6f}',flush=True)

    basecmd = [sys.executable,'r5.py']
    opts = ['--profile','tiny_t2','--tag',tag,'--max-minutes','1']
    kill_log = logs/'killed.log'
    with kill_log.open('w') as f:
        child = subprocess.Popen(basecmd+['train']+opts+['--pause-after-checkpoint'],stdout=f,stderr=subprocess.STDOUT)
        try:
            until = min(deadline.end-2,time.monotonic()+25)
            while time.monotonic() < until:
                if 'R5_PAUSED_AFTER_CHECKPOINT' in kill_log.read_text():
                    break
                if child.poll() is not None:
                    raise RuntimeError(kill_log.read_text())
                time.sleep(.05)
            else:
                raise TimeoutError(f'no saved-checkpoint pause; inspect {kill_log}')
            first_state = read_pickle(ck/'base.pkl')
            assert first_state['update']==1 and not first_state['complete']
            child.kill()
            child.wait(timeout=5)
        finally:
            if child.poll() is None:
                child.kill()
                child.wait(timeout=5)

    def command(name, extra=(), target_tag=None, log_name=None):
        options = ['--profile','tiny_t2','--tag',target_tag or tag,'--max-minutes','1']
        with (logs/f'{log_name or name}.log').open('w') as log:
            remaining = deadline.end-time.monotonic()-1
            if remaining <= 0:
                raise TimeoutError('smoke exceeded wall budget')
            subprocess.run(basecmd+[name]+options+list(extra),stdout=log,stderr=subprocess.STDOUT,
                           check=True,timeout=remaining)
    command('train')
    final = read_pickle(ck/'base.pkl')
    assert final['complete'] and final['update']==2
    assert final['history'][0] == first_state['history'][0]
    assert 'r5 resume' in (logs/'train.log').read_text()
    cfg = {k:v for k,v in final['config'].items() if k not in ('start_sha256','schema')}
    reference = train(init_params(0,16),ck/'resume_reference.pkl',cfg,deadline)
    max_error = max(float(np.max(np.abs(final['params'][k]-reference['params'][k]))) for k in final['params'])
    assert max_error < 1e-6, max_error
    assert int(final['optimizer']['step']) == int(reference['optimizer']['step'])
    assert all(np.isfinite(r['loss']) for r in final['history'])
    assert final['history'][-1]['lr'] < final['history'][0]['lr']
    assert final['selection']['step'] == reference['selection']['step']
    assert final['validation_history'] == reference['validation_history']
    assert final['cost']['validation_env_steps'] > 0
    for k in final['params']:
        np.testing.assert_allclose(deployment_params(final)[k],deployment_params(reference)[k],atol=1e-6,rtol=0)
    # DAgger replay at stage 3 contains examples from every skill.
    dagger = read_pickle(ck/'dagger_s3.pkl')
    assert np.all(dagger['replay'].count>0)
    command('cpg')
    command('cpg',extra=['--reuse-cpg',tag],target_tag=tag+'_reuse',log_name='cpg_reuse')
    import json
    original_cpg = json.loads((res/'cpg.json').read_text())
    reused_cpg = json.loads((Path('results/r5')/(tag+'_reuse')/'cpg.json').read_text())
    assert original_cpg['theta'] == reused_cpg['theta'] and original_cpg['reset_phase'] == reused_cpg['reset_phase']
    command('evaluate')
    command('replay-demo')
    command('report')
    evaluation = json.loads((res/'eval.json').read_text())
    assert evaluation['primary_seed_bank']=='smoke'
    assert {r['split'] for r in evaluation['rows']}=={'test','previous_test','legacy'}
    assert Path(f'media/r5_{tag}_demo.html').is_file()
    assert Path(f'media/r5_{tag}_transitions.png').is_file()
    result = dict(status='pass',tag=tag,wall_s=time.monotonic()-started,
                  sigkill_resume_max_weight_error=max_error,fixed_batch_mse_before=first,
                  fixed_batch_mse_after=last,replay_counts=dagger['replay'].count.tolist(),
                  checks=['MLX/NumPy parity','finite zero gradients','frozen old route','SIGKILL resume',
                          'optimizer continuity','all three methods/stages','CPG','held-out switching',
                          'parameter/cost reporting','offline demo','checkpoint demo reload',
                          'validation-selected deployment/resume','LR annealing','CPG unchanged reuse',
                          'fresh/previous/legacy test separation'])
    write_json(res/'smoke.json',result)
    print(f'R5_SMOKE_OK {result}; logs={logs}',flush=True)


if __name__=='__main__':
    main()
