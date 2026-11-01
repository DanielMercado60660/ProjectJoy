"""Check gemma.py against mlx-lm's gemma3_text (run in ../.venv-moshi), then check the KV-cache path.

  MLX_ENABLE_TF32=0 ../.venv/bin/python parity.py     # -> results/parity.json (true fp32; with TF32 the diffs are ~0.08)

1. Logits on ~300 tokens of real text, both in fp32: max abs diff and top-1 agreement vs mlx-lm.
   (Kept under the 512-token window: mlx-lm 0.26.4 applies no window without a cache.)
2. Cached one-token decoding vs one full forward on 700 tokens (crosses the sliding window): max diff.
3. A short greedy continuation, to eyeball that the text model works.
"""
import json, os, subprocess, sys
import numpy as np
import mlx.core as mx

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from gemma import load_gemma, load_tokenizer, GEMMA_DIR  # noqa: E402

TEXT = ("The Warren Commission was established by President Lyndon B. Johnson on November 29, 1963, to "
        "investigate the assassination of United States President John F. Kennedy that had taken place on "
        "November 22, 1963. Its 888-page final report was presented to President Johnson on September 24, 1964, "
        "and made public three days later. It concluded that President Kennedy was assassinated by Lee Harvey "
        "Oswald and that Oswald acted entirely alone. ")
REF = r'''
import sys, numpy as np, mlx.core as mx
from mlx_lm import load
m, _ = load(sys.argv[1]); m.set_dtype(mx.float32)
t = np.load(sys.argv[2])["tokens"]
out = m(mx.array(t)[None]).astype(mx.float32)
mx.eval(out); np.save(sys.argv[3], np.array(out[0]))
'''


def main():
    tok = load_tokenizer()
    ids = [2] + tok.encode(TEXT * 2, add_special_tokens=False).ids
    ids = ids[:300]
    m, c = load_gemma(dtype=mx.float32)
    ours = np.array(m.logits(mx.array(ids)[None])[0])
    os.makedirs(os.path.join(HERE, "data"), exist_ok=True)
    tpath, rpath = os.path.join(HERE, "data", "parity_tokens.npz"), os.path.join(HERE, "data", "parity_ref.npy")
    np.savez(tpath, tokens=np.array(ids))
    subprocess.run([os.path.join(HERE, "..", ".venv-moshi", "bin", "python"), "-c", REF, GEMMA_DIR[0], tpath, rpath], check=True)
    ref = np.load(rpath)
    res = {"n_tokens": len(ids), "max_abs_diff": float(np.abs(ours - ref).max()),
           "logit_scale": float(np.abs(ref).max()),
           "top1_agree": float((ours.argmax(-1) == ref.argmax(-1)).mean())}
    # next-token loss on the text (sanity: a trained LM should be well under log(262144) = 12.5)
    lp = ours - np.log(np.exp(ours - ours.max(-1, keepdims=True)).sum(-1, keepdims=True)) - ours.max(-1, keepdims=True)
    res["nll_nats_per_token"] = float(-lp[np.arange(len(ids) - 1), ids[1:]].mean())

    # 2. cache consistency across the sliding window
    long_ids = ([2] + tok.encode(TEXT * 8, add_special_tokens=False).ids)[:700]
    full = m.logits(mx.array(long_ids)[None])[0]
    caches = [dict() for _ in m.layers]
    h = m.hidden(m.embed(mx.array(long_ids[:200])[None]), caches, 0)
    step = [m.embed_tokens.as_linear(h)[0]]
    for p in range(200, 700):
        h = m.hidden(m.embed(mx.array([[long_ids[p]]])), caches, p)
        step.append(m.embed_tokens.as_linear(h)[0])
    step = mx.concatenate(step, 0)
    res["cache_vs_full_max_abs_diff_700tok"] = float(mx.abs(step - full).max())

    # 3. greedy continuation
    prompt = [2] + tok.encode("The capital of France is", add_special_tokens=False).ids
    caches = [dict() for _ in m.layers]
    h = m.hidden(m.embed(mx.array(prompt)[None]), caches, 0)
    out, pos = [], len(prompt)
    nxt = int(mx.argmax(m.embed_tokens.as_linear(h[:, -1]), -1).item())
    for _ in range(12):
        out.append(nxt)
        h = m.hidden(m.embed(mx.array([[nxt]])), caches, pos); pos += 1
        nxt = int(mx.argmax(m.embed_tokens.as_linear(h[:, -1]), -1).item())
    res["greedy"] = "The capital of France is" + tok.decode(out)
    print(json.dumps(res, indent=1))
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    json.dump(res, open(os.path.join(HERE, "results", "parity.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
