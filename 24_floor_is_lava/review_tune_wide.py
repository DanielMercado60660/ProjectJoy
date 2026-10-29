"""Reviewer baseline check (REVIEW.md): re-tune the CPG + foothold planner with a WIDER gait range.

Same tuner as tune_cpg.py (random search + CEM + re-evaluation, training layouts only); the only
change is the search box for gait frequency and stride, widened so a faster / longer-stride gait
class (closer to the residual policy's 1.65 m/s) is reachable:
    freq   1.2-3.5 Hz  ->  1.2-5.0 Hz
    stride 0.0-0.30 m  ->  0.0-0.45 m
All other arguments pass straight through to tune_cpg.main().

Usage: ../bin/gpu_run ../.venv/bin/python review_tune_wide.py --planner --warm checkpoints/planner4_params.json \
           --seed 5 --n_random 120 --n_layouts 40 --cem_gens 10 --cem_pop 48 --final_layouts 192 \
           --out checkpoints/review_planner5_wide_params.json
       ../bin/gpu_run ../.venv/bin/python review_tune_wide.py --fast_corner --planner \
           --warm checkpoints/review_fast_corner_params.json ...   (planner restricted to fast gaits)
"""
import os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import task as T
import tune_cpg

WIDE = dict(freq=(1.2, 5.0), stride=(0.0, 0.45))
# --fast_corner: search ONLY fast gaits (nominal freq x stride >= 0.7 m/s), as in review_fast_corner.py
FAST = dict(freq=(3.2, 5.0), stride=(0.22, 0.45))

if __name__ == "__main__":
    os.chdir(HERE)
    fast = "--fast_corner" in sys.argv
    if fast:
        sys.argv.remove("--fast_corner")
    T.CPG_RANGES.update(FAST if fast else WIDE)   # tune_cpg.main() reads T.CPG_RANGES for lo/hi and the CEM clip
    tune_cpg.main()
