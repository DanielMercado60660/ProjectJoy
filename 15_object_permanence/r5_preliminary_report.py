"""Render preliminary turn-1 numbers from retained artifacts; no MLX execution."""
import argparse
import json
from pathlib import Path
import pickle
import numpy as np
from r5_metrics import paired_ci

HEADER = '## Round 5: Remember several possibilities, then look'


def read(name):
    return json.loads(Path('results/r5/'+name+'.json').read_text())


def ci_text(ci):
    return f'{ci["mean"]:+.3f} [{ci["low"]:+.3f}, {ci["high"]:+.3f}]'


def verify(d):
    assert d['complete'] and d['requested_n'] == d['completed_n']
    for method, policies in d['per_sequence'].items():
        for policy, rows in policies.items():
            assert len(rows) == d['completed_n']
            for group in ('all', 'visible', 'hidden', 'reappear'):
                count = sum(row[group]['n'] for row in rows)
                assert count == d['summary'][method][policy][group]['n']
                for key in ('error_px', 'nll', 'coverage1', 'coverage2'):
                    value = sum(row[group][key]*row[group]['n'] for row in rows if row[group]['n'])/count
                    assert np.isclose(value, d['summary'][method][policy][group][key], atol=1e-10)
        delta = [a['all']['nll']-b['all']['nll'] for a, b in zip(policies['uncertainty'], policies['random'])]
        expected = paired_ci(delta)
        actual = d['contrasts'][method+'/uncertainty_minus_random/all_nll']
        assert all(np.isclose(actual[k], expected[k]) for k in actual)


def report():
    fresh, legacy, tune = read('fresh_ac'), read('legacy_ac'), read('tuning_ac')
    audit = read('turn2_cue_audit')
    verify(fresh)
    verify(legacy)
    with open('checkpoints/r5_ac/last.pkl', 'rb') as f:
        ckpt = pickle.load(f)
    val = {row['step']:row['val_nll_px'] for row in ckpt['history'] if 'val_nll_px' in row}
    text = [HEADER, '',
      '**Turn 1 is preliminary; turn 2 is pending runner.** These 128 fresh sequences were examined to design turn 2, so they are no longer an untouched headline set. Values below were checked against retained JSON summaries and their per-sequence arithmetic; Astra did not run MLX.', '',
      'The mixture improves never-peek all-frame NLL (3.222 vs 3.518), but the planned hidden-NLL contrast remains uncertain: '+ci_text(fresh['contrasts']['mixture_minus_gaussian/never/hidden_nll'])+' nats/sequence-mean (95% paired bootstrap CI, 127 sequences with hidden frames). The particle filter with uncertainty peeks has the fewest confusion-entry events (0.109/sequence) and lowest visible error (1.582 px); the mixture has the lowest final-swap fraction (7.03%). Identity, precision and density quality are different outcomes. Source: [fresh_ac.json](results/r5/fresh_ac.json), `summary` and `contrasts`.', '',
      '### Turn-1 setup and actual compute', '',
      f'- Common warm source: `checkpoints/hard/final`, SHA-256 `{ckpt["config"]["warm_sha256"]}`. Both heads received the same batch-8 inputs, noisy 0.25 px slot cues, original wall geometry, and random peeks in 2/3 of training clips. Extra pretraining is shared between the heads and is not matched classical compute.',
      f'- **6000 paired updates took {ckpt["active_seconds"]:.2f} s ({audit["paired_steps_per_second"]:.2f} paired steps/s), not 105 minutes.** The fixed step ceiling ended segment 1; segments 2/3 were skipped. Validation pixel NLL still improved from step 3100 to 6000: Gaussian {val[3100]["gaussian"]:.3f} → {val[6000]["gaussian"]:.3f}; mixture {val[3100]["mixture"]:.3f} → {val[6000]["mixture"]:.3f}. Source: [train_ac_1.log](logs/r5/train_ac_1.log), `VAL` and `PAIR_DONE`; checkpoint history. The original runtime estimate was badly wrong.',
      f'- Classical search completed {len(tune["rows"])} candidates in {tune["elapsed_seconds"]:.2f} s on `hard_val[0:16]`, confirming finalists on `[64:112]`. PF: q=0.5, r=1, miss=0.35, P=512, KDE bandwidth=4 px. **q and miss are upper grid boundaries.** Kalman q=0.5, r=14 is interior. Source: [tuning_ac.json](results/r5/tuning_ac.json), `selected`, `boundary_parameters`, `rows`.',
      '- Thresholds are each tracker’s validation 75th percentile, calibrated on `hard_val[128:144]`: [thresholds_ac.json](results/r5/thresholds_ac.json). Both look policies spend one view per sequence, with 100% request coverage; never spends zero. Decisions use t−1 information; physical motion does not change.',
      f'- Complete evaluations: fresh 128/128 in {fresh["elapsed_seconds"]:.2f} s (seeds 51034000+i); legacy `hard_test[0:64]` 64/64 in {legacy["elapsed_seconds"]:.2f} s. Source: the two evaluation JSONs and [fresh_ac.log](logs/r5/fresh_ac.log)/[legacy_ac.log](logs/r5/legacy_ac.log). Claude’s [smoke log](logs/r5/claude_smoke1.log) records a 7-second pass, including actual SIGKILL/resume equivalence.', '',
      '### Preliminary fresh split: all trackers and policies', '',
      'Source for every cell: [fresh_ac.json](results/r5/fresh_ac.json), `summary.<tracker>.<policy>`. Pixel errors are posterior-mean errors; NLL is the full reported distribution’s unclipped pixel-density NLL. Coverage uses the x-then-y Rosenblatt transform (nominal 0.393/0.865), not a mixture moment ellipse. Errors/NLL/coverage are weighted by object-frames; identity fractions are per sequence. Natural opaque-view masks stay fixed across peek policies.', '',
      '| Tracker / policy | Visible px | Hidden px | Reappear px | All NLL | Hidden NLL | All C1 / C2 | Switches/seq | Final swaps |',
      '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for method, policies in fresh['summary'].items():
        for policy, r in policies.items():
            text.append(f'| {method} / {policy} | {r["visible"]["error_px"]:.3f} | {r["hidden"]["error_px"]:.3f} | {r["reappear"]["error_px"]:.3f} | {r["all"]["nll"]:.3f} | {r["hidden"]["nll"]:.3f} | {r["all"]["coverage1"]:.3f} / {r["all"]["coverage2"]:.3f} | {r["identity"]["switches_per_seq"]:.3f} | {100*r["identity"]["swapped_at_end"]:.2f}% |')
    text += ['', 'Hidden calibration remains weaker than all-frame calibration: never-peek Gaussian C1/C2 is 0.316/0.772 and mixture 0.299/0.764. The much broader classical densities trade NLL for coverage; high coverage alone is not success. The identity metric counts entries into a permutation-confused state with a 2 px margin; it can miss midpoint hedging.', '',
      '### Did a look help?', '',
      'Source: [fresh_ac.json](results/r5/fresh_ac.json), `summary.<tracker>.uncertainty.peek`, and both evaluation files’ `contrasts.<tracker>/uncertainty_minus_random/all_nll`. Positive spread/later-error/later-NLL gains mean improvement versus never; negative uncertainty-minus-random contrasts favor uncertainty. Contrast means/CIs give each sequence equal weight, so they need not equal differences of the object-frame-weighted NLL table. CIs are conditional on one training seed.', '',
      '| Tracker | Spread removed px² | Later error gain px | Later NLL gain | Fresh: uncertainty − random NLL [95% CI] | Legacy: same contrast [95% CI] |',
      '|---|---:|---:|---:|---|---|']
    for method in fresh['summary']:
        r = fresh['summary'][method]['uncertainty']['peek']
        key = method+'/uncertainty_minus_random/all_nll'
        text.append(f'| {method} | {r["spread_reduction_px2"]:.3f} | {r["later_error_gain_px"]:.3f} | {r["later_nll_gain"]:.3f} | {ci_text(fresh["contrasts"][key])} | {ci_text(legacy["contrasts"][key])} |')
    text += ['',
      'On fresh data, the Gaussian, mixture and particle uncertainty policies beat random in the stored paired NLL intervals. **Kalman does not:** its −0.002 point contrast has an interval spanning zero. Its 73.918 px² spread reduction yields only 0.003 later NLL gain. “Removing uncertainty” without better predictions is not useful sensing. The GRU advantage does not replicate on the 64 legacy sequences: both uncertainty-versus-random intervals cross zero. The mixture’s tiny weighted legacy NLL decrease and near-zero equal-sequence contrast should not be described as a confirmed gain.', '',
      '### Binding failure and turn-2 diagnosis', '',
      'Source: [fresh_ac.json](results/r5/fresh_ac.json), `summary.<tracker>.never.visible_by_initial_xgap`. These are never-peek comparisons under matched information; only six sequences have an initial x-gap below 1 px.', '',
      '| Tracker | Visible error, x-gap <1 px (6 seqs) | Visible error, x-gap ≥8 px (79 seqs) |',
      '|---|---:|---:|']
    for method in fresh['summary']:
        r = fresh['summary'][method]['never']['visible_by_initial_xgap']
        text.append(f'| {method} | {r["<1"]["error_px"]:.3f} px | {r[">=8"]["error_px"]:.3f} px |')
    text += ['',
      'The cue is already added at **every step** in turn 1. Both checkpoint position-cue projection matrices have 3072 nonzero weights (norms 0.733/0.738); see [turn2_cue_audit.json](results/r5/turn2_cue_audit.json), produced by `r5_turn2_utils.py`. Thus literal zero cue use is **not established**. Source inspection shows the stronger problem: a single global cue embedding feeds a shared hidden state and three separately indexed heads; labels are always sorted left-to-right. Ignoring or weakly using the cue still offers a shortcut. Under-training is plausible from the falling validation curve, but no intervention has isolated its contribution yet. The runner-only cue-permutation/mean-cue probe will measure sensitivity on validation.', '',
      'What worked: mixture density prediction, finite calibrated outputs, reproducible checkpoint/resume, and useful adaptive looks on one fresh sample. What did not: reliable near-tie binding, consistent peek benefit across samples, and fully bracketed PF tuning. The surprise was throughput: the supposed long training schedule was only 3.4 minutes. A winner is not declared for multimodal hidden-state tracking.', '',
      'Preserved demos: [fixed sequence 0](media/r5_ac_fixed_sequence0.png) and [least-helpful mixture look](media/r5_ac_least_helpful_look.png). The second is selected by the worst subsequent NLL gain and deliberately retains a failure; it is not representative performance. The original source and scripts are archived in `results/r5/turn1_source_snapshot/` with hashes.', '',
      '### Turn 2 — pending runner', '',
      'Turn 2 uses shared per-query GRU/readout weights, cue input at every step plus initial state and head, cue-relative displacements, and permuted query/target slots. Both heads reuse the same original hard-world trunk and share the new initialization; the indexed position head is replaced. Training is planned for 120,000 paired batch-8 steps over at most three 35-minute segments, with one shared best checkpoint selected on validation only. This changes architecture and training exposure together, so improvement cannot isolate which change caused it. PF q/miss ranges are expanded. New headline seeds are **52024000+i** (`turn2_fresh`); turn-1 `fresh` keeps its old meaning.', '',
      '| Turn-2 item | Status |', '|---|---|',
      '| MLX smoke, cue equivariance/gradients, kill/resume | pending runner |',
      '| Validation-selected pair and expanded PF tuning | pending runner |',
      '| New headline, 512 sequences, all metrics/policies | pending runner |',
      '| Legacy secondary and new demos | pending runner |',
      '', 'Commands, measured-throughput budgeting and validation gates: [RUNME_R5.md](RUNME_R5.md), **Turn 2**. No turn-2 learned results have been measured.']
    return '\n'.join(text)+'\n'


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--append', action='store_true')
    args = p.parse_args()
    text = report()
    if args.append:
        path = Path('RESULTS.md')
        if HEADER in path.read_text():
            raise RuntimeError('round-5 section already exists; refusing duplicate append')
        with path.open('a') as f:
            f.write('\n'+text)
        print('PRELIMINARY_REPORT_OK: artifact arithmetic verified; appended RESULTS.md')
    else:
        print(text)
