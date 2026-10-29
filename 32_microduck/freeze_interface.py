"""Write results/m2/interface_frozen.json: the command interface spec plus the sha256 of interface.py.
Run once at the end of M2. tests/test_m2.py fails if interface.py or its spec changes afterwards:
changing the interface after this point needs a new milestone decision, not a quiet edit."""
import datetime as dt
import hashlib
import json
from pathlib import Path

from interface import frozen_spec

HERE = Path(__file__).resolve().parent
OUT = HERE / "results" / "m2" / "interface_frozen.json"

if __name__ == "__main__":
    doc = {"frozen": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
           "interface_py_sha256": hashlib.sha256((HERE / "interface.py").read_bytes()).hexdigest(),
           "spec": frozen_spec()}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(doc, indent=1))
    print(json.dumps(doc, indent=1))
