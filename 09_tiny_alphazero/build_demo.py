"""Build media/demo.html: a self-contained page (JS engine + net + MCTS, weights inlined
as base64-float32 JSON) where you can play against a checkpoint.

  python build_demo.py checkpoints/main/gen_080 media/demo.html "optional note"
"""
import json
import os
import sys

from export_web import export

HERE = os.path.dirname(os.path.abspath(__file__))


def build(prefix, out, note=""):
    tmp = out + ".weights.json"
    spec = export(prefix, tmp)
    os.remove(tmp)
    spec["meta"]["note"] = note
    spec.pop("tests", None)
    with open(os.path.join(HERE, "web", "demo_template.html")) as f:
        html = f.read()
    with open(os.path.join(HERE, "web", "az.js")) as f:
        js = f.read()
    html = html.replace("/*AZ_JS*/", js).replace("/*WEIGHTS*/", json.dumps(spec))
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w") as f:
        f.write(html)
    print(f"wrote {out} ({os.path.getsize(out)/1e6:.2f} MB)")


if __name__ == "__main__":
    build(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else "")
