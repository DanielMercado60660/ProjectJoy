#!/bin/zsh
# Usage: run_agent.sh <name> <prompt>  -> research/<name>.md (report), logs/<name>.{jsonl,err,last.md}
cd /Users/danielmercado/ProjectJoy/33_gemma_voice/research
codex --search exec -s workspace-write -C /Users/danielmercado/ProjectJoy/33_gemma_voice/research --skip-git-repo-check --json -o logs/$1.last.md "$2" < /dev/null > logs/$1.jsonl 2> logs/$1.err
echo "exit=$? $1"
