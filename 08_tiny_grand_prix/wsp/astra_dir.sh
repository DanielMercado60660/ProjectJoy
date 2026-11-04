#!/bin/zsh
# Usage: astra_dir.sh <workdir_abs> <tag> <sandbox: read-only|workspace-write> <prompt> [resume_thread_id]
# Same as ../../bin/astra_turn (gpt-6-astra, xhigh, Codex sandbox) but runs Codex with -C <workdir>
# (a subfolder of wsp/, so Astra starts in a folder holding only what it is meant to see) and lets
# the Read experiment use a read-only sandbox. Logs: ProjectJoy/round5_logs/08_tiny_grand_prix_<tag>.*
root="$(cd "$(dirname "$0")/../.." && pwd)"
dir=$1; tag=$2; sb=$3; prompt=$4; thread=$5
log=$root/round5_logs/08_tiny_grand_prix_${tag}
if [[ -n $thread ]]; then
  codex exec -m gpt-6-astra -c model_reasoning_effort=xhigh -s $sb -C $dir --skip-git-repo-check --json -o $log.last.md resume $thread "$prompt" < /dev/null > $log.jsonl 2> $log.err
else
  codex exec -m gpt-6-astra -c model_reasoning_effort=xhigh -s $sb -C $dir --skip-git-repo-check --json -o $log.last.md "$prompt" < /dev/null > $log.jsonl 2> $log.err
fi
rc=$?
echo "exit=$rc thread=$(grep -o '"thread_id":"[^"]*"' $log.jsonl | head -1 | cut -d'"' -f4)"
cat $log.last.md
