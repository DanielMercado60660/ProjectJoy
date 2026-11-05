"""Turn gemma_voice_colab.py ('# %%' cells, '# %% [markdown]' cells) into gemma_voice_colab.ipynb for Colab.

  python3 build_notebook.py
The .py stays the source of truth (it also runs on the laptop: GV_LOCAL=1 GV_WORK=<dir> python gemma_voice_colab.py).
"""
import json, os

HERE = os.path.dirname(os.path.abspath(__file__))
src = open(os.path.join(HERE, "gemma_voice_colab.py")).read().split("\n")
cells, cur, kind = [], [], None


def flush():
    if kind is None:
        return
    lines = cur[:]
    while lines and not lines[-1].strip():
        lines.pop()
    if kind == "markdown":
        lines = [l[2:] if l.startswith("# ") else l.lstrip("#") for l in lines]
    text = [l + "\n" for l in lines]
    if text:
        text[-1] = text[-1].rstrip("\n")
    c = {"cell_type": kind, "metadata": {}, "source": text}
    if kind == "code":
        c.update(execution_count=None, outputs=[])
    cells.append(c)


for line in src:
    if line.startswith("# %%"):
        flush()
        kind, cur = ("markdown" if "[markdown]" in line else "code"), []
    else:
        cur.append(line)
flush()
nb = {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"provenance": [], "gpuType": "A100"},
                                   "kernelspec": {"name": "python3", "display_name": "Python 3"},
                                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 0}
out = os.path.join(HERE, "gemma_voice_colab.ipynb")
json.dump(nb, open(out, "w"), indent=1)
print(out, len(cells), "cells")
