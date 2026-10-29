"""Print efficiency/bench_runs.jsonl as a table (optionally filtered by tag substring)."""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
flt = sys.argv[1] if len(sys.argv) > 1 else ""
print(f"{'time':8} {'tag':22} {'variant':26} {'d':>3} {'BxA':>6} {'tok/s med':>10} {'min':>9} {'max':>9} {'spr%':>5} {'rel':>6} {'MFU':>5} {'peakTF':>11} {'mem':>5} {'load':>11}")
for l in open(os.path.join(HERE, "bench_runs.jsonl")):
    r = json.loads(l)
    if flt not in r["tag"]:
        continue
    env = "".join(f"[{k[-2:]}={v}]" for k, v in r["env"].items())
    print(f"{r['time'][11:19]:8} {r['tag'][:22]:22} {(r['variant'] + env)[:26]:26} {r['depth']:>3} {str(r['B'])+'x'+str(r['accum']):>6} "
          f"{r['tok_s_median']:>10,.0f} {r['tok_s_min']:>9,.0f} {r['tok_s_max']:>9,.0f} {r['spread_pct']:>5.1f} {r['rel_to_first']:>6.3f} "
          f"{r['mfu']*100:>4.1f}% {str(r['matmul_peak_tflops']):>11} {r['peak_mem_gb']:>5.1f} {r['loadavg_start'][0]:>5.1f}/{r['loadavg_end'][0]:<5.1f}")
