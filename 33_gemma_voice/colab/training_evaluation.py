"""Fixed validation speech panels and checkpoint provenance, run between training segments."""
def evaluate_training_checkpoint():
    from huggingface_hub import HfApi
    ctx = build_training_context()
    step = ctx['step']
    original_ids = list(VAL100)
    original_eval = STATE['eval']
    n = 100 if ctx['epoch'] > 0 and ctx['batch'] == 0 else 24
    results = {'lj': evaluate(n=n, tag=f'{CFG["RUN"]}_s{step}_lj{n}')}
    # Unseen Emilia transcripts spoken with the learned LJ voice. This measures
    # text generalization, not zero-shot speaker similarity (no trained ID exists for many heldouts).
    rows = []
    for jp in sorted(glob.glob(os.path.join(WORK,'data','ming','emilia_yodas','*.json'))):
        with open(jp) as f: m=json.load(f)
        for i, cid in enumerate(m['ids']):
            if m['splits'][i] == 'val' and 20 <= len(m['texts'][i]) <= 200:
                rows.append((hashlib.sha256(cid.encode()).hexdigest(), cid, m['texts'][i], m['offs'][i+1]-m['offs'][i]))
    rows.sort()
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(CFG['BACKBONE'])
    panel={cid:((tuple(text_ids(tok,text)),0,frames,0),text) for _,cid,text,frames in rows[:16]}
    try:
        VAL100[:] = list(panel)
        STATE['eval'] = panel
        results['emilia_text_lj_voice'] = evaluate(n=16,tag=f'{CFG["RUN"]}_s{step}_emilia16_ljvoice')
    finally:
        VAL100[:] = original_ids
        STATE['eval'] = original_eval
    api=HfApi()
    # Evaluation uploads are also commits, but last.pt remains the just-saved checkpoint.
    revision=api.model_info(CKPT_REPO).sha if CFG['PUSH'] else None
    report=dict(step=step,epoch=ctx['epoch'],batch=ctx['batch'],checkpoint_revision=revision,
                results={k:{kk:vv for kk,vv in v.items() if kk!='clips'} for k,v in results.items()})
    target=os.path.join(ckpt_dir(),f'evaluation_{step}.json')
    with open(target,'w') as f: json.dump(report,f,indent=2)
    if CFG['PUSH']:
        api.upload_file(path_or_fileobj=target,path_in_repo=f'{CFG["RUN"]}/evaluation_{step}.json',repo_id=CKPT_REPO)
    print('SPEECH_EVALUATION_SAVED',json.dumps(report),flush=True)
    return report
