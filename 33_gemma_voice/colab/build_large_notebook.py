"""Build the standalone staged-data Colab variant from the base notebook and launch cells.

Run from project 33: ../.venv/bin/python colab/build_large_notebook.py
The original smoke notebook stays available unchanged.
"""
import ast
import json
import runpy
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUN = HERE / "runs" / "2026-11-03"
runpy.run_path(str(HERE / 'build_training_launch.py'))
nb = json.loads((HERE / "gemma_voice_colab.ipynb").read_text())


def set_source(index, source):
    nb["cells"][index]["source"] = source.splitlines(keepends=True)


set_source(0, """# Gemma Voice: 2,500-hour first training milestone

This variant extends the A100 notebook with Emilia-YODAS English. It prepares
at least 2,500 hours of filtered, unique training clips; validation, test and
LJ repeats are excluded from the target. Source revision and filtering policy
are pinned. Original LJ / LibriTTS-R / Hi-Fi data are retained.

The original smoke and full codec gate were completed on 2026-11-03. Full-model
training on this larger corpus has **not** been completed. Run setup/definitions,
then the preparation cell. It runs continuously until the data target and audit
are complete, saving chunks to your private HF repo. Colab can still end the
runtime; rerun preparation to resume. The user authorized removing the 40-minute
limit. A later expansion to 10,000+ hours reuses the completed encoding.

Training uses a verified fresh-process checkpoint recovery path. At the current
batch budget one epoch is 45,143 updates; the learning-rate horizon is three
epochs (135,431 updates). The training cell restores or verifies/calibrates the
run, then continues through the first full epoch with saved speech evaluations.
Review its generated speech before extending to epochs two and three.
""")
config = (RUN / "full_config.py").read_text()
config = config.replace("UNITS_PER_HOUR=8.88", "UNITS_PER_HOUR=6.73")
config = config.replace("# The first training session is 15 minutes; resume continues toward STEPS=30000.",
                        "# STEPS is provisional; first full-data training is limited to 100 steps / 15 minutes.")
set_source(1, config)
set_source(12, """## Run

The full codec gate already passed its technical checks (results in the run
folder). On reconnect run configuration/setup/definitions, then rerun preparation.
Completed chunks are restored from HF. Do not start model training on a partial
dataset. The next cell checks the target and all original corpus shards first.
""")
set_source(13, (RUN / "full_gate.py").read_text())
set_source(14, (HERE / "emilia_expansion.py").read_text() +
           "\n# User-authorized continuous preparation to 2,500 usable training hours.\n"+
           "# Saves progress throughout; rerun after a Colab runtime interruption.\n"+
           "preparation = prepare_large_data(minutes=None)\n")
preflight = (HERE / "runs" / "2026-11-04" / "training_preflight.py").read_text()
continuous = (HERE / "runs" / "2026-11-04" / "train_first_epoch.py").read_text()
eval_start = continuous.index('with open(os.path.join(WORK,"training_evaluation.py")')
set_source(15, preflight + "\n# Continue automatically through the first full epoch.\n" + continuous[eval_start:])
set_source(16, "# Re-evaluate the latest saved full-data checkpoint, without training.\n"
               "assert run_training_worker('evaluate') == 0\n")
source = []
for cell in nb["cells"]:
    cell["metadata"] = {}
    text = "".join(cell["source"])
    if cell["cell_type"] == "code":
        ast.parse(text)
        cell.update(execution_count=None, outputs=[])
        source.append("# %%\n" + text.rstrip())
    else:
        source.append("# %% [markdown]\n" + "\n".join("# " + line for line in text.splitlines()))
out = HERE / "gemma_voice_large.ipynb"
out.write_text(json.dumps(nb, indent=1) + "\n")
py = HERE / "gemma_voice_large.py"
py.write_text("\n\n".join(source) + "\n")
ast.parse(py.read_text())
print(out)
print(py)
print(len(nb["cells"]), "cells; all code cells parse")
