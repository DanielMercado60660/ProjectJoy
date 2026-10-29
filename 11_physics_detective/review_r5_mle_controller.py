# Exploratory REVIEWER check (reuses turn-2 TEST worlds; not a headline): does restricting gain-MLE's final push to already-probed forces close its gap to gain-Bayes?
# Run: OMP_NUM_THREADS=2 ../.venv/bin/python review_r5_mle_controller.py 40 gain_shift > results/r5/review_mle_controller_gain_shift.txt  (and "40 matched"; ~110 s each)
import json, sys, time, numpy as np
from pathlib import Path; HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
from r5_gain import GainBelief, GainConfig, history_arrays
from r5_core import TerminalTask, FINAL_ACTIONS, true_stop
NW = int(sys.argv[1]) if len(sys.argv) > 1 else 32
rows = json.load(open(HERE / 'checkpoints/r5_turn2_main.json'))['rows']
idx = {(r['world'], r['family'], r['method'], r['budget']): r for r in rows}
cfg = GainConfig(2401, 641)
res = {}
t0 = time.time()
for fam in sys.argv[2].split(','):
    for w in range(NW):
        final = idx[(w, fam, 'eig', 4)]
        post = GainBelief(cfg, 0.15)
        task = TerminalTask(final['target'])
        for b in range(1, 5):
            post.update(*history_arrays(final['history'][b - 1]))
            if b not in (2, 4):
                continue
            mu, g = post.mle()
            a_std = task.point_action(mu, gain=g)
            forces = sorted({float(np.clip(round(h['action'][0]), 1, 15)) for h in final['history'][:b]})
            menu = FINAL_ACTIONS[np.isin(FINAL_ACTIONS[:, 0], forces)]
            d = true_stop(menu, mu, g)
            a_pf = menu[int(np.argmin(np.abs(d - final['target'])))]
            for name, a in (('mle_std', a_std), ('mle_probeforce', a_pf)):
                stop = float(true_stop(np.asarray(a), final['mu_true'], final['gain_true']))
                res.setdefault((fam, b, name), []).append(abs(stop - final['target']) <= 0.025)
            res.setdefault((fam, b, 'recorded_gain_mle'), []).append(idx[(w, fam, 'gain_mle', b)]['success'])
            res.setdefault((fam, b, 'recorded_gain_bayes'), []).append(idx[(w, fam, 'gain_bayes', b)]['success'])
for k in sorted(res):
    print(k, f"{np.mean(res[k])*100:.2f}% n={len(res[k])}")
print('elapsed', round(time.time() - t0, 1))
