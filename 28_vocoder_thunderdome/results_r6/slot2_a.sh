cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=4
./r6_run_eval.sh val100 "recon_phase_s24000=checkpoints/r6_runs/recon_phase/snaps/G_24000.safetensors" 2>&1 | grep -E '^val100|"n"'
../.venv/bin/python r6_oracle.py --ckpt checkpoints/followup_val3k/best.safetensors --tag old --gl 32
../.venv/bin/python r6_oracle.py --ckpt checkpoints/r6_runs/recon/snaps/G_24000.safetensors --tag recon24k --gl 32
echo '[{"wavdir":"results_r6/wavs/val100/old_Gmag_GL32","out":"results_r6/mos/val100/old_Gmag_GL32.json"},{"wavdir":"results_r6/wavs/val100/recon24k_Gmag_GL32","out":"results_r6/mos/val100/recon24k_Gmag_GL32.json"}]' > results_r6/jobs_a.json
../.venv-sac/bin/python r6_utmos.py --jobs results_r6/jobs_a.json 2>&1 | grep '"n"'
