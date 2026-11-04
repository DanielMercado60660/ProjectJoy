#!/bin/zsh
# usage: launch.sh <tag> <variant: os|b50>
tag=$1; v=$2; N=${v#b}
common="Read wsp/data/TASK.md first and follow its rules exactly. Your tag is $tag and your folder is wsp/attempts/$tag/ (write only there). Measured on this machine: one numpy forward pass of the actor on 100000 inputs takes 0.04 s; one evaluation call takes about 1 s. Plan to finish in well under an hour of work. Your final merged networks will be scored later on held-out reset seeds you never see."
if [[ $v == os ]]; then
  p="You are one attempt of a weight-space programming experiment, variant ONE-SHOT. You have NO evaluation budget: no rollouts or environment use of any kind; you will get no feedback, this is your only turn. Using only the weights in wsp/data (and numpy analysis/forward passes on synthetic inputs you generate), write one merged network per seed with the same architecture that runs forward under the fwd label and backward under the bwd label. $common Deliver merged_s0..3.npz, build.py and NOTES.md as TASK.md specifies."
else
  p="You are one attempt of a weight-space programming experiment, variant BUDGET-$N. You may make at most $N calls to wsp/tools/eval_budget.py in total (all 4 seeds together); every other form of environment use is forbidden. Using the weights in wsp/data (and numpy analysis/forward passes on synthetic inputs you generate), write one merged network per seed with the same architecture that runs forward under the fwd label and backward under the bwd label. Every evaluated candidate must be a reasoned edit you can explain (not a random perturbation); keep an edit only if it scores better on the training resets, and log in NOTES.md what each call tested and what you kept. $common Deliver merged_s0..3.npz, build.py and NOTES.md as TASK.md specifies."
fi
"$(cd "$(dirname "$0")/../../.." && pwd)"/bin/astra_turn 25_skill_transplantation $tag "$p" </dev/null
