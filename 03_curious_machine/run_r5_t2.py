"""Turn-2 scheduler. One gpu_run slot, at most two isolated child processes, two
CPU threads each. Three bounded waves resume the same job list; no shared checkpoints.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
from run_r5 import code_hashes, exploration_score

METHODS = ('lpm','rnd_ep','disagree','ngu','frontier','random')
SEEDS = (610,611,612)
VAL_SEEDS = tuple(range(64600,64604))
TEST_SEEDS = tuple(range(85400,85416))
SECONDARY = tuple(range(72500,72504))
STEPS = 5996544  # exactly the original round's 366 * 128 * 128 steps
ACTIVE = {}  # owned children only, for explicit scheduler interruption


def interrupt(signum, frame):
    for proc, group in list(ACTIVE.values()):
        try:
            if group:os.killpg(proc.pid,signal.SIGKILL)
            else:proc.kill()
        except ProcessLookupError:pass
    raise SystemExit(128+signum)


def child(args, log, deadline, group=False):
    remaining=deadline-time.monotonic()
    if remaining<8:return False
    log=Path(log);log.parent.mkdir(parents=True,exist_ok=True)
    with log.open('a') as output:
        proc=subprocess.Popen([sys.executable,*map(str,args)],stdout=output,stderr=subprocess.STDOUT,
                              start_new_session=group)
        ACTIVE[proc.pid]=(proc,group)
        try:code=proc.wait(timeout=max(1,remaining-2))
        except BaseException as error:
            # The batch owns a whole cell process group; its nested trainer cannot
            # outlive a timed-out cell and retain GPU work after the lock is released.
            try:
                if group:os.killpg(proc.pid,signal.SIGTERM)
                else:proc.terminate()
                proc.wait(timeout=1)
            except subprocess.TimeoutExpired:
                if group:os.killpg(proc.pid,signal.SIGKILL)
                else:proc.kill()
                proc.wait()
            except ProcessLookupError:pass
            finally:
                if group:
                    # Root may exit on TERM before a descendant does. Clean the
                    # entire owned group even if waiting for the root succeeded.
                    try:os.killpg(proc.pid,signal.SIGKILL)
                    except ProcessLookupError:pass
                ACTIVE.pop(proc.pid,None)
            if isinstance(error,subprocess.TimeoutExpired):return False
            raise
        ACTIVE.pop(proc.pid,None)
    if code==2:return False  # a clean wall-guard stop is resumable, not a model error
    if code:raise RuntimeError(f'Child exit {code}: {log}')
    return True


def hashes():
    out = code_hashes()
    for name in ('planner_r5_t2.py','progress_r5_t2.py','run_r5_t2.py'):
        out[name] = hashlib.sha256(Path(name).read_bytes()).hexdigest()
    return out


def paths(tag, method, world, seed):
    stem=f'{method}_{world}_s{seed}'
    return (Path('checkpoints') / f'r5_{tag}' / f'{stem}.pkl',
            Path('logs/r5') / tag / stem, Path('results/r5') / tag / f'{stem}.json')


def train_cell(tag, method, world, seed, steps, selected, deadline):
    ck, logs, _ = paths(tag,method,world,seed)
    remaining = deadline-time.monotonic()
    if remaining < 8: return False,ck
    minutes = min(44.,(remaining-4)/60)
    if method in ('frontier','random'):
        selection=Path('results/r5')/tag/'cell_selection.json'
        # Parent freezes this before parallel workers start; no concurrent writes.
        args=['classical_r5.py','--selection',selection]
    else:
        args=['train_r5.py','--lpm-window',selected['lpm_window'],
              '--ent-coef',selected['ent_coef'] if method=='lpm' else .001]
        if ck.exists():args+=['--resume']
    args+=['--method',method,'--world',world,'--seed',seed,'--steps',steps,'--n-envs',128,
           '--max-minutes',minutes,'--checkpoint',ck,'--log-dir',logs]
    ok=child(args,logs/'console.log',deadline)
    outcome=logs/'outcome.json'
    return ok and outcome.exists() and json.loads(outcome.read_text())['steps']==steps,ck


def freeze(path, obj):
    obj=json.loads(json.dumps(obj))
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        if json.loads(path.read_text())!=obj:raise ValueError(f'Changed manifest/configuration: {path}; use a new tag')
    else:path.write_text(json.dumps(obj,indent=2))


def tune(tag, deadline, pilot_steps=1048576):
    from evaluate_r5 import evaluate,save_eval
    out=Path('results/r5')/tag;out.mkdir(parents=True,exist_ok=True)
    selected_path=out/'selection.json'
    freeze(out/'tuning_manifest.json',dict(hashes=hashes(),val_seeds=VAL_SEEDS,pilot_seed=619,
                                         pilot_steps=pilot_steps,n_envs=128,settings='windows 1,4 x entropy .001,.01'))
    if selected_path.exists():return True
    from diagnose_r5_t2 import oracle_check
    freeze(out/'oracle.json',dict(trials=oracle_check(),budget=128,max_expansions=100000))
    records=[]
    for bonus in (0.,6.,12.):
        for frontier in (0.,2.):
            cfg=dict(mechanism_bonus=bonus,frontier_bonus=frontier);scores=[]
            for world in ('plain','tv'):
                path=out/f'frontier_m{bonus:g}_f{frontier:g}_{world}.json'
                if path.exists():r=json.loads(path.read_text())
                else:
                    # Selection uses ONLY discovery. Replaying the old task planner
                    # here avoids spending tuning time on the new, unused task scores.
                    r,_=evaluate('frontier',VAL_SEEDS,world=='tv',frontier_cfg=cfg,deadline=deadline)
                    save_eval(path,r)
                scores.append(exploration_score(r))
            records.append(dict(method='frontier',cfg=cfg,score=sum(scores)/2))
    for window in (1,4):
        for entropy in (.001,.01):
            cfg=dict(lpm_window=window,ent_coef=entropy);scores=[]
            pilot=f'{tag}_w{window}_e{entropy:g}'
            for world in ('plain','tv'):
                ok,ck=train_cell(pilot,'lpm',world,619,pilot_steps,cfg,deadline)
                if not ok:return False
                path=out/f'lpm_w{window}_e{entropy:g}_{world}.json'
                if path.exists():r=json.loads(path.read_text())
                else:
                    r,_=evaluate('lpm',VAL_SEEDS,world=='tv',checkpoint=ck,deadline=deadline,policy_seed=83619)
                    save_eval(path,r)
                scores.append(exploration_score(r))
            records.append(dict(method='lpm',cfg=cfg,score=sum(scores)/2))
    best=lambda method:max((r for r in records if r['method']==method),key=lambda r:r['score'])['cfg']
    selected=dict(frontier=best('frontier'),**best('lpm'),records=records,hashes=hashes(),
                  validation_seeds=list(VAL_SEEDS),selection_metric='rooms/9 + mechanisms/8 - .25*TV_fraction')
    selected_path.write_text(json.dumps(selected,indent=2))
    print('FROZEN_SELECTION',selected_path,flush=True);return True


def cell(tag,method,world,seed,steps,selected,deadline):
    from evaluate_r5 import evaluate,save_eval
    ck,logs,path=paths(tag,method,world,seed)
    if not path.exists() or not path.with_suffix('.npz').exists():
        ok,ck=train_cell(tag,method,world,seed,steps,selected,deadline)
        if not ok:return False
        r,data=evaluate(method,TEST_SEEDS,world=='tv',ck if method not in ('frontier','random') else None,
                        selected['frontier'],deadline=deadline,record=True,planner_type='compositional',
                        max_expansions=100000,policy_seed=83519)
        r.update(training_seed=seed,charged_pretrain_steps=steps,pretrain_checkpoint=str(ck),split='fresh',turn=2)
        save_eval(path,r,data)
    secondary=path.with_name(path.stem+'_old.json')
    if not secondary.exists():
        r,_=evaluate(method,SECONDARY,world=='tv',ck if method not in ('frontier','random') else None,
                     selected['frontier'],deadline=deadline,planner_type='compositional',max_expansions=100000,policy_seed=83519)
        r.update(training_seed=seed,charged_pretrain_steps=steps,pretrain_checkpoint=str(ck),split='turn1_secondary',turn=2)
        save_eval(secondary,r)
    print('COMPLETE_CELL',method,world,seed,flush=True);return True


def batch(a,deadline):
    selected=json.loads(Path(a.selection).read_text());out=Path('results/r5')/a.tag
    manifest=dict(hashes=hashes(),selected=selected,steps=a.steps,seeds=list(a.seeds),test_seeds=list(TEST_SEEDS),
                  planner='compositional',task_steps=128,max_expansions=100000,n_envs=128,workers=a.workers)
    freeze(out/'manifest.json',manifest);freeze(out/'cell_selection.json',selected)
    jobs=[]
    for world in ('plain','tv'):
        for method in METHODS:
            for seed in a.seeds:
                _,_,p=paths(a.tag,method,world,seed)
                if not(p.exists() and p.with_suffix('.npz').exists() and p.with_name(p.stem+'_old.json').exists()):
                    jobs.append((method,world,seed))
    if not jobs:
        print('ALL_COMPLETE',flush=True);return True
    wave=out/f'wave{a.wave}.json'
    if wave.exists():raise ValueError('Wave already attempted; advance to the next wave; preserve its receipt')
    wave_record=dict(started=time.time(),jobs=jobs,status='running',max_minutes=a.max_minutes)
    freeze(wave,wave_record)
    # Child imports see these BEFORE NumPy/Numba/MLX initialize. The scheduler itself
    # runs no neural network and holds no model; two workers together use <=4 threads.
    for var in ('NUMBA_NUM_THREADS','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
        os.environ[var]=str(4//a.workers)
    stop=threading.Event()
    def launch(job):
        method,world,seed=job
        remaining=deadline-time.monotonic()
        if remaining<10 or stop.is_set():return False
        _,logs,_=paths(a.tag,method,world,seed)
        try:
            return child(['run_r5_t2.py','cell','--tag',a.tag,'--selection',a.selection,'--method',method,'--world',world,
                          '--seed',seed,'--steps',a.steps,'--max-minutes',min(44.,(remaining-4)/60)],logs/'cell.log',deadline,group=True)
        except Exception:
            stop.set();raise
    started=time.monotonic();completed=0;failed=[]
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        futures={pool.submit(launch,j):j for j in jobs}
        for future in as_completed(futures):
            try:ok=future.result()
            except Exception as e:
                failed.append(dict(job=futures[future],error=str(e)));ok=False
            if ok:completed+=1
    complete=completed==len(jobs)
    wave_record.update(status='complete' if complete else 'partial',completed=completed,queued=len(jobs),
                       failed=failed,elapsed_seconds=time.monotonic()-started)
    wave.write_text(json.dumps(wave_record,indent=2))
    print('ALL_COMPLETE' if complete else 'PARTIAL_RESUMABLE',flush=True)
    if failed:raise RuntimeError(f'Cell errors; inspect {wave}; no failed cell is dropped')
    return complete


def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=('tune','batch','cell'));p.add_argument('--tag',required=True)
    p.add_argument('--selection');p.add_argument('--max-minutes',type=float,required=True)
    p.add_argument('--steps',type=int,default=STEPS);p.add_argument('--wave',type=int,choices=(1,2,3),default=1)
    p.add_argument('--seeds',type=int,nargs='+',choices=SEEDS,default=list(SEEDS))
    p.add_argument('--pilot-steps',type=int,default=1048576)
    p.add_argument('--workers',type=int,choices=(1,2),default=2);p.add_argument('--method',choices=METHODS)
    p.add_argument('--world',choices=('plain','tv'));p.add_argument('--seed',type=int,choices=SEEDS)
    a=p.parse_args()
    signal.signal(signal.SIGTERM,interrupt);signal.signal(signal.SIGINT,interrupt)
    if not a.tag.replace('_','').replace('-','').isalnum():p.error('tag: letters/digits/underscore/hyphen only')
    if not 0<a.max_minutes<=45 or a.steps<=0 or a.steps%16384 or a.pilot_steps<=0 or a.pilot_steps%16384:
        p.error('guard in (0,45]; positive steps/pilot-steps a multiple of 16384')
    if len(set(a.seeds))!=len(a.seeds):p.error('duplicate seeds')
    if a.stage!='tune' and not a.selection:p.error('--selection required')
    if a.stage=='cell' and any(x is None for x in (a.method,a.world,a.seed)):p.error('cell identity required')
    deadline=time.monotonic()+60*a.max_minutes
    try:
        if a.stage=='tune':ok=tune(a.tag,deadline,a.pilot_steps)
        elif a.stage=='batch':ok=batch(a,deadline)
        else:ok=cell(a.tag,a.method,a.world,a.seed,a.steps,json.loads(Path(a.selection).read_text()),deadline)
    except TimeoutError as e:
        print('PARTIAL',e,flush=True);ok=False
    print('COMPLETE' if ok else 'PARTIAL_RESUMABLE',flush=True)
    return 0 if ok else 2


if __name__=='__main__':sys.exit(main())
