"""Full-data Colab trainer. Requires the notebook's dataset/model/eval definitions.

No local execution: this module targets the authorized Colab A100 only.
"""
import hashlib
import copy
import gc

TRAINING_VERSION = 2
TRAINING_KEYS = ('CODEC', 'BACKBONE', 'SETS', 'LJ_REPEAT', 'HEAD_W', 'HEAD_DEPTH',
    'HEAD_HEADS', 'BUDGET', 'K', 'CFG_DROP', 'STOP_W', 'POS_W', 'T_SAMPLING',
    'LATENT', 'IN_NOISE', 'LR', 'LR_NEW', 'WARMUP', 'TRAIN_EPOCHS', 'SEED')


def stable_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def capture_rng():
    return dict(python=random.getstate(), numpy=np.random.get_state(),
                torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state_all())


def restore_rng(value):
    random.setstate(value['python']); np.random.set_state(value['numpy'])
    torch.set_rng_state(value['torch'].cpu())
    torch.cuda.set_rng_state_all([s.cpu() for s in value['cuda']])


def find_ckpt():
    p = os.path.join(ckpt_dir(), 'last.pt')
    if os.path.exists(p):
        return p
    if CFG['PUSH']:
        from huggingface_hub import HfApi, hf_hub_download
        files = set(HfApi().list_repo_files(CKPT_REPO))
        prefix = CFG['RUN']+'/'
        if prefix+'last.pt' in files:
            for name in ('last.pt','meta.json','dataset.json','log.jsonl','resume_check.json','calibration.json'):
                if prefix+name in files:
                    hf_hub_download(CKPT_REPO, prefix+name, local_dir=os.path.join(WORK,'ckpt'))
            return p
    return None


def frozen_data(tok):
    """Reject changed shards, speaker IDs, normalization, tokenizer or training policy on resume."""
    items, val, ev, store, speakers, mu, sd = load_data(tok)
    root = os.path.join(WORK, 'data', CFG['CODEC'])
    manifest = []
    for source in CFG['SETS']:
        for jp in sorted(glob.glob(os.path.join(root, source, '*.json'))):
            with open(jp, 'rb') as f:
                digest = hashlib.sha256(f.read()).hexdigest()
            manifest.append([os.path.relpath(jp, root), digest, os.path.getsize(jp[:-5]+'.f16')])
    # Include actual tokenizer vocabulary/config, not only its mutable repository name.
    identity = dict(version=TRAINING_VERSION, manifest=manifest, speakers=speakers,
        mu=mu.tolist(), sd=sd.tolist(), tokenizer_vocab=stable_hash(tok.get_vocab()),
        bos=tok.bos_token_id, config={k: CFG[k] for k in TRAINING_KEYS},
        runtime_source=globals().get('TRAINING_SOURCE_HASH'))
    path = os.path.join(ckpt_dir(), 'dataset.json')
    digest = stable_hash(identity)
    if os.path.exists(path):
        with open(path) as f:
            assert stable_hash(json.load(f)) == digest, 'Frozen dataset/training identity changed; use a new RUN.'
    else:
        with open(path+'.tmp', 'w') as f:
            json.dump(identity, f)
        os.replace(path+'.tmp', path)
    lens = np.asarray([len(it[0])+it[2]+1 for it in items])
    counts = [len(batches_for_epoch(lens, CFG['BUDGET'], CFG['SEED'], ep))
              for ep in range(CFG['TRAIN_EPOCHS'])]
    CFG['STEPS'] = sum(counts)
    print('TRAINING_PLAN', json.dumps(dict(epochs=CFG['TRAIN_EPOCHS'], batches_per_epoch=counts,
        total_steps=CFG['STEPS'], train_clips=len(items), speakers=len(speakers),
        padded_token_budget=CFG['BUDGET'], data_identity=digest)), flush=True)
    return dict(items=items, val=val, ev=ev, store=store, speakers=speakers,
                mu=mu, sd=sd, lens=lens, epoch_counts=counts, identity=digest)


def build_training_context():
    from transformers import AutoTokenizer
    random.seed(CFG['SEED']); np.random.seed(CFG['SEED']); torch.manual_seed(CFG['SEED'])
    p = find_ckpt()
    data = frozen_data(AutoTokenizer.from_pretrained(CFG['BACKBONE']))
    D = {'ming':64, 'mimi':512}[CFG['CODEC']]
    net = GemmaVoice(CFG['BACKBONE'], D, len(data['speakers'])).to(DEV)
    opt = torch.optim.AdamW([
        {'params':net.bb_params(), 'lr':CFG['LR'], 'weight_decay':0.01},
        {'params':net.new_params(), 'lr':CFG['LR_NEW'], 'weight_decay':0.0}],
        betas=(0.9, 0.95), fused=True)
    ctx = dict(data=data, model=net, opt=opt, step=0, epoch=0, batch=0, audio_frames=0,
               order=None, order_epoch=None)
    if p:
        ck = torch.load(p, map_location='cpu', weights_only=False)
        assert ck.get('version') == TRAINING_VERSION, 'Legacy checkpoint cannot resume exactly; use a new RUN.'
        assert ck['identity'] == data['identity'], 'Checkpoint dataset/config mismatch'
        assert ck['backbone_revision'] == net.bb.m.config._commit_hash, 'Backbone revision changed'
        missing, unexpected = net.load_state_dict(ck['model'], strict=False)
        assert not unexpected and all(k.startswith('bb.m.embed_tokens') for k in missing)
        opt.load_state_dict(ck['opt'])
        for k in ('step', 'epoch', 'batch', 'audio_frames'):
            ctx[k] = ck[k]
        assert ctx['step'] == sum(data['epoch_counts'][:ctx['epoch']])+ctx['batch']
        restore_rng(ck['rng'])
        del ck
        print('EXACT_RESUME', json.dumps({k:ctx[k] for k in ('step','epoch','batch')}), flush=True)
    STATE.update(model=net, eval=data['ev'], mu=data['mu'], sd=data['sd'],
                 step=ctx['step'], spk=data['speakers'])
    return ctx


def next_training_batch(ctx):
    ep, k = ctx['epoch'], ctx['batch']
    d = ctx['data']
    assert ep < CFG['TRAIN_EPOCHS']
    if ctx['order_epoch'] != ep:
        ctx['order'] = batches_for_epoch(d['lens'], CFG['BUDGET'], CFG['SEED'], ep)
        ctx['order_epoch'] = ep
    bi = ctx['order'][k]
    # Posterior sampling is keyed by the exact update, independent of prefetch/session length.
    rng = np.random.default_rng(np.random.SeedSequence([CFG['SEED'], ctx['step'], 7]))
    b = make_batch([d['items'][i] for i in bi], d['store'], ctx['model'].D, d['mu'], d['sd'], rng)
    return b


def optimizer_update(ctx):
    net, opt = ctx['model'], ctx['opt']
    net.train()
    b = next_training_batch(ctx)
    for group, base in zip(opt.param_groups, (CFG['LR'], CFG['LR_NEW'])):
        group['lr'] = base * lr_lambda(ctx['step'])
    in_noise = CFG['IN_NOISE'] if CFG['IN_NOISE'] is not None else (0.0 if CFG['CODEC']=='ming' else 0.2)
    loss, flow, stop = loss_fn(net, b, in_noise)
    assert torch.isfinite(loss), 'Non-finite training loss; checkpoint preserved'
    opt.zero_grad(set_to_none=True)
    loss.backward()
    gn = torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0, error_if_nonfinite=True)
    opt.step()
    ctx['step'] += 1; ctx['batch'] += 1
    ctx['audio_frames'] += int(b['flen'].sum())
    if ctx['batch'] == ctx['data']['epoch_counts'][ctx['epoch']]:
        ctx['epoch'] += 1; ctx['batch'] = 0
    STATE['step'] = ctx['step']
    return dict(step=ctx['step'], loss=float(loss), flow=float(flow), stop=float(stop),
                gnorm=float(gn), lr=opt.param_groups[0]['lr'], clips=len(b['flen']),
                audio_seconds=float(b['flen'].sum())/12.5)


def save_training(ctx, push=True):
    """Atomic local checkpoint; one Hub commit publishes weights + identity + logs."""
    net, opt, d = ctx['model'], ctx['opt'], ctx['data']
    ck = dict(version=TRAINING_VERSION,
        model={k:v.detach().cpu() for k,v in net.state_dict().items() if not k.startswith('bb.m.embed_tokens')},
        opt=opt.state_dict(), identity=d['identity'], cfg=copy.deepcopy(CFG), rng=capture_rng(),
        backbone_revision=net.bb.m.config._commit_hash,
        **{k:ctx[k] for k in ('step','epoch','batch','audio_frames')})
    p = os.path.join(ckpt_dir(), 'last.pt')
    torch.save(ck, p+'.tmp'); os.replace(p+'.tmp', p)
    meta = dict(step=ctx['step'], epoch=ctx['epoch'], batch=ctx['batch'],
        audio_hours_seen=ctx['audio_frames']/45000, identity=d['identity'], cfg=copy.deepcopy(CFG))
    with open(os.path.join(ckpt_dir(), 'meta.json'), 'w') as f:
        json.dump(meta, f)
    if push and CFG['PUSH']:
        from huggingface_hub import HfApi
        api = HfApi()
        api.create_repo(CKPT_REPO, private=True, exist_ok=True)
        api.upload_folder(folder_path=ckpt_dir(), path_in_repo=CFG['RUN'], repo_id=CKPT_REPO,
            allow_patterns=['last.pt','meta.json','dataset.json','log.jsonl','resume_check.json','calibration.json'])
    print('CHECKPOINT_SAVED', json.dumps(dict(step=ctx['step'], pushed=bool(push and CFG['PUSH']))), flush=True)


def state_digest(ctx):
    """Compare every trained weight, optimizer tensor, and RNG state after the replayed update."""
    h = hashlib.sha256()
    def add(x):
        if isinstance(x, torch.Tensor):
            h.update(x.detach().cpu().contiguous().numpy().tobytes())
        elif isinstance(x, dict):
            for k in sorted(x, key=str):
                h.update(str(k).encode()); add(x[k])
        elif isinstance(x, (tuple,list)):
            for v in x: add(v)
        else:
            h.update(repr(x).encode())
    add({k:v for k,v in ctx['model'].state_dict().items() if not k.startswith('bb.m.embed_tokens')})
    add(ctx['opt'].state_dict()); add(capture_rng())
    add([ctx['step'],ctx['epoch'],ctx['batch'],ctx['audio_frames']])
    return h.hexdigest()


def cold_resume_reference():
    """Save after 3 updates, record uninterrupted update 4, then deliberately terminate this worker."""
    import signal
    ctx = build_training_context()
    assert ctx['step'] == 0, 'Resume test expects a fresh isolated RUN'
    for _ in range(3): optimizer_update(ctx)
    save_training(ctx, push=False)
    rec = optimizer_update(ctx)
    rec['state_sha256'] = state_digest(ctx)
    with open(os.path.join(ckpt_dir(),'resume_expected.json'), 'w') as f:
        json.dump(rec,f)
    print('REFERENCE_READY: saved step 3; recorded uninterrupted step 4; terminating test worker',flush=True)
    os.kill(os.getpid(), signal.SIGTERM)


def cold_resume_replay():
    ctx = build_training_context()
    assert ctx['step'] == 3
    rec = optimizer_update(ctx)
    rec['state_sha256'] = state_digest(ctx)
    with open(os.path.join(ckpt_dir(),'resume_expected.json')) as f:
        expected = json.load(f)
    assert rec == expected, {'expected':expected, 'actual':rec}
    result = dict(passed=True, exact_state_match=True, resumed_step=3, compared_step=4,
                  loss=rec['loss'], state_sha256=rec['state_sha256'], identity=ctx['data']['identity'])
    with open(os.path.join(ckpt_dir(),'resume_check.json'), 'w') as f:
        json.dump(result,f)
    print('COLD_RESUME_PASSED', json.dumps(result), flush=True)


def train_verified(max_steps_this_session=None, session_hours=None):
    ctx = build_training_context()
    check_path = os.path.join(ckpt_dir(),'resume_check.json')
    assert os.path.exists(check_path), 'Run cold checkpoint test before sustained training'
    with open(check_path) as f:
        check = json.load(f)
    assert check['passed'] and check['identity'] == ctx['data']['identity']
    start_step = ctx['step']; start = time.monotonic(); last_save = start
    stop_at = CFG['STEPS'] if max_steps_this_session is None else min(CFG['STEPS'], start_step+max_steps_this_session)
    elapsed_steps, audio_seconds = [], []
    log_path = os.path.join(ckpt_dir(), 'log.jsonl')
    try:
        with open(log_path, 'a') as log:
            while ctx['step'] < stop_at:
                t0 = time.monotonic()
                rec = optimizer_update(ctx)
                dt = time.monotonic()-t0
                elapsed_steps.append(dt); audio_seconds.append(rec['audio_seconds'])
                if ctx['step'] % CFG['LOG_EVERY'] == 0:
                    recent = elapsed_steps[-CFG['LOG_EVERY']:]
                    speed = sum(recent)/len(recent)
                    rec.update(epoch=ctx['epoch']+ctx['batch']/ctx['data']['epoch_counts'][min(ctx['epoch'],CFG['TRAIN_EPOCHS']-1)],
                        seconds_per_step=speed, eta_hours=(CFG['STEPS']-ctx['step'])*speed/3600,
                        audio_hours_seen=ctx['audio_frames']/45000,
                        gpu_peak_gb=torch.cuda.max_memory_allocated()/2**30)
                    print('TRAIN',json.dumps(rec),flush=True); log.write(json.dumps(rec)+'\n'); log.flush()
                if ctx['step'] % CFG['VAL_EVERY'] == 0:
                    d = ctx['data']
                    vl = val_loss(ctx['model'], d['val'], d['store'], ctx['model'].D, d['mu'], d['sd'],0.0)
                    rec = dict(step=ctx['step'], val=list(vl))
                    print('VALIDATION',json.dumps(rec),flush=True); log.write(json.dumps(rec)+'\n'); log.flush()
                if time.monotonic()-last_save >= CFG['SAVE_MIN']*60:
                    save_training(ctx); last_save=time.monotonic()
                if session_hours is not None and time.monotonic()-start >= session_hours*3600:
                    break
    except KeyboardInterrupt:
        print('Interrupted: saving completed updates.',flush=True)
    save_training(ctx)
    timing = dict(start_step=start_step, end_step=ctx['step'], elapsed_minutes=(time.monotonic()-start)/60,
        median_seconds_per_step=float(np.median(elapsed_steps[5:] or elapsed_steps)) if elapsed_steps else None,
        mean_seconds_per_step=float(np.mean(elapsed_steps[5:] or elapsed_steps)) if elapsed_steps else None,
        epoch_counts=ctx['data']['epoch_counts'], audio_hours_seen=ctx['audio_frames']/45000,
        gpu_peak_gb=torch.cuda.max_memory_allocated()/2**30)
    with open(os.path.join(ckpt_dir(),'calibration.json'), 'w') as f: json.dump(timing,f)
    print('TRAINING_SESSION_SAVED',json.dumps(timing),flush=True)
    return ctx
