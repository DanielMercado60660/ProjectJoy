"""Invoke the sole permitted environment interface once and record its decision."""
from pathlib import Path
import argparse
import json
import subprocess
import sys
import shutil
import datetime
HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[2]
p = argparse.ArgumentParser()
p.add_argument('--seed', type=int, required=True)
p.add_argument('--candidate', required=True)
p.add_argument('--reason', required=True)
p.add_argument('--previous', type=float)
a = p.parse_args()
candidate = HERE / a.candidate
with (HERE / 'NOTES.md').open('a') as f:
    f.write(f'\n**Planned call for seed {a.seed}, `{a.candidate}`:** {a.reason}\n')
command = [sys.executable, str(PROJECT / 'wsp/tools/eval_budget.py'), '--attempt', HERE.name,
           '--seed', str(a.seed), '--cand', str(candidate)]
result = subprocess.run(command, capture_output=True, text=True, check=True)
print(result.stdout)
start = result.stdout.index('{')
end = result.stdout.rindex('}') + 1
record = json.loads(result.stdout[start:end])
(HERE / f'call_{record["call"]:02d}.json').write_text(json.dumps(record, indent=2) + '\n')
keep = a.previous is None or record['mean_retained'] > a.previous
if keep:
    shutil.copyfile(candidate, HERE / f'merged_s{a.seed}.npz')
decision = 'established baseline' if a.previous is None else ('kept' if keep else 'rejected; previous incumbent retained')
with (HERE / 'NOTES.md').open('a') as f:
    f.write(f'Call **{record["call"]}/10**: fwd {record["fwd"]["mean_return"]}, '
            f'bwd {record["bwd"]["mean_return"]}; mean retained **{record["mean_retained"]:.4f}**; '
            f'previous {a.previous}; **{decision}**.\n')
with (HERE / 'STATUS.md').open('a') as f:
    f.write(f'{datetime.datetime.now().astimezone().isoformat(timespec="seconds")} — '
            f'Call {record["call"]}/10 seed {a.seed}: {record["mean_retained"]:.4f}, {decision}.\n')
