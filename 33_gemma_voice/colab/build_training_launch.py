"""Bundle the Colab worker without installing anything or rerunning data preparation."""
import ast
import hashlib
import json
from pathlib import Path
HERE = Path(__file__).resolve().parent
nb = json.loads((HERE/'gemma_voice_colab.ipynb').read_text())
body = '\n\n'.join(''.join(nb['cells'][i]['source']) for i in (3,4,5,7,8,9,10,11))
body += '\n\n'+(HERE/'training_runtime.py').read_text()
digest = hashlib.sha256(body.encode()).hexdigest()
header = '''import os, sys, math, json, time, glob, shutil, subprocess, threading, queue, random, io
import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.signal import resample_poly
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
torch.set_num_threads(4)
with open(sys.argv[1]) as f:
    launch_config = json.load(f)
CFG, WORK, DATA_REPO, CKPT_REPO = (launch_config[k] for k in ('CFG','WORK','DATA_REPO','CKPT_REPO'))
DEV, AMP, LOCAL, SMOKE, IN_COLAB = 'cuda', torch.bfloat16, False, False, False
sys.path.insert(0, os.path.join(WORK,'ming_pkg'))
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
'''
footer = '''
if __name__ == '__main__':
    action = sys.argv[2]
    if action == 'restore':
        print('CHECKPOINT_RESTORE', bool(find_ckpt()), flush=True)
    elif action == 'reference':
        cold_resume_reference()
    elif action == 'replay':
        cold_resume_replay()
    elif action == 'calibrate':
        ctx = train_verified(max_steps_this_session=100, session_hours=0.25)
    elif action == 'train':
        ctx = train_verified(max_steps_this_session=CFG.get('SESSION_STEPS'), session_hours=CFG.get('SESSION_HOURS'))
    elif action == 'evaluate':
        with open(os.path.join(WORK,'training_evaluation.py')) as f:
            exec(compile(f.read(), 'training_evaluation.py', 'exec'))
        evaluate_training_checkpoint()
    else:
        raise ValueError(action)
'''
worker = header+'\nTRAINING_SOURCE_HASH = '+repr(digest)+'\n'+body+footer
ast.parse(worker)
(HERE/'training_worker.py').write_text(worker)
launch = '''# Full-corpus training: exact resume test and short timing calibration.
# All data are already prepared; this never reruns encoding.
assert not SMOKE and 'emilia_yodas' in CFG['SETS']
import signal, gc
STATE.clear()
model = None
_CODECS.clear()
_JUDGE.clear()
for _name in ('preflight_data','preflight_items','preflight_lens','preflight_spk','preflight_tok'):
    globals().pop(_name, None)
gc.collect(); torch.cuda.empty_cache()
CFG.update(RUN='g270m_2500_v1', TRAIN_EPOCHS=3, SEED=20261104,
           LOG_EVERY=25, VAL_EVERY=2000, SAVE_MIN=10, SESSION_HOURS=0.75, UNITS_PER_HOUR=6.73)
training_worker_path = os.path.join(WORK,'training_worker.py')
training_config_path = os.path.join(WORK,'training_launch.json')
'''
launch += 'with open(training_worker_path,"w") as f: f.write('+repr(worker)+')\n'
launch += '''with open(training_config_path,'w') as f:
    json.dump(dict(CFG=CFG,WORK=WORK,DATA_REPO=DATA_REPO,CKPT_REPO=CKPT_REPO),f)
def run_training_worker(action):
    from huggingface_hub import get_token
    worker_env = os.environ.copy()
    worker_env['HF_TOKEN'] = get_token() or ''
    proc = subprocess.Popen([sys.executable,'-u',training_worker_path,training_config_path,action],
                            stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,env=worker_env)
    try:
        for line in proc.stdout:
            print(line,end='',flush=True)
        return proc.wait()
    except KeyboardInterrupt:
        proc.send_signal(signal.SIGINT)
        for line in proc.stdout:
            print(line,end='',flush=True)
        proc.wait()
        raise
# First process saves step 3, computes the uninterrupted step 4, then terminates.
# A completely fresh process reloads step 3 and must reproduce every state byte.
resume_report = os.path.join(WORK,'ckpt',CFG['RUN'],'resume_check.json')
assert run_training_worker('restore') == 0, 'Checkpoint restoration failed'
if not os.path.exists(resume_report):
    expected_report = os.path.join(WORK,'ckpt',CFG['RUN'],'resume_expected.json')
    if not os.path.exists(expected_report):
        rc = run_training_worker('reference')
        assert rc == -signal.SIGTERM, f'Reference worker failed: {rc}'
    assert run_training_worker('replay') == 0, 'Cold resume replay failed'
if not os.path.exists(os.path.join(WORK,'ckpt',CFG['RUN'],'calibration.json')):
    assert run_training_worker('calibrate') == 0, 'Calibration failed'
print('CALIBRATION_READY: choose session duration from measured throughput before the sustained run.',flush=True)
'''
ast.parse(launch)
(HERE/'runs'/'2026-11-04'/'training_preflight.py').write_text(launch)
print('Worker and launch cell parse; source SHA256',digest)
# A review point is one complete epoch. Short child processes resume automatically;
# the notebook cell keeps going and does not ask the user to restart every 40 minutes.
continuous = launch[:launch.index('# First process saves step 3')]
continuous = continuous.replace('# Full-corpus training: exact resume test and short timing calibration.',
                                '# Train the first full epoch, with automatic checkpoint/resume and speech checks.')
continuous += 'with open(os.path.join(WORK,"training_evaluation.py"),"w") as f: f.write('+repr((HERE/'training_evaluation.py').read_text())+')\n'
continuous += '''
resume_report = os.path.join(WORK,'ckpt',CFG['RUN'],'resume_check.json')
assert os.path.exists(resume_report), 'Complete the cold-resume check first.'
with open(os.path.join(WORK,'ckpt',CFG['RUN'],'calibration.json')) as f:
    calibration = json.load(f)
first_epoch_steps = calibration['epoch_counts'][0]
review_steps = sorted(set([5000] + [round(first_epoch_steps*f) for f in (0.25,0.5,0.75,1.0)]))
# Keep the precise training code alongside checkpoints for recovery in a fresh Colab.
from huggingface_hub import HfApi, CommitOperationAdd
HfApi().create_commit(repo_id=CKPT_REPO,commit_message='Preserve verified full-corpus training code',
    operations=[CommitOperationAdd(path_in_repo=f'{CFG["RUN"]}/code/'+os.path.basename(p),path_or_fileobj=p)
                for p in (training_worker_path,os.path.join(WORK,'training_evaluation.py'))])
print('FULL_EPOCH_STARTED',json.dumps(dict(run=CFG['RUN'],target_step=first_epoch_steps,
      planned_epochs=CFG['TRAIN_EPOCHS'],speech_review_steps=review_steps)),flush=True)
while True:
    with open(os.path.join(WORK,'ckpt',CFG['RUN'],'meta.json')) as f:
        before = json.load(f)['step']
    if before >= first_epoch_steps:
        break
    target = next(s for s in review_steps if s > before)
    CFG.update(SESSION_STEPS=target-before,SESSION_HOURS=40/60)
    with open(training_config_path,'w') as f:
        json.dump(dict(CFG=CFG,WORK=WORK,DATA_REPO=DATA_REPO,CKPT_REPO=CKPT_REPO),f)
    assert run_training_worker('train') == 0, 'Training stopped; last completed checkpoint preserved.'
    with open(os.path.join(WORK,'ckpt',CFG['RUN'],'meta.json')) as f:
        after=json.load(f)['step']
    assert after > before, 'Training made no progress.'
    if after in review_steps:
        assert run_training_worker('evaluate') == 0, 'Speech evaluation failed; training checkpoint is safe.'
    if after < target:
        print('AUTOMATIC_CONTINUATION: checkpoint saved; loading the next segment.',flush=True)
# Also handles reconnecting after the final update but before final speech evaluation.
final_report = os.path.join(WORK,'ckpt',CFG['RUN'],f'evaluation_{first_epoch_steps}.json')
if not os.path.exists(final_report):
    assert run_training_worker('evaluate') == 0
print('FIRST_EPOCH_READY: all training examples seen once; review generated speech before epochs 2–3.',flush=True)
'''
ast.parse(continuous)
(HERE/'runs'/'2026-11-04'/'train_first_epoch.py').write_text(continuous)
print('Continuous first-epoch launch parses')
