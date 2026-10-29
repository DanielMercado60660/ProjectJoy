"""Re-verify splits/ from the files alone. Writes results/m1/split_check.json; exits non-zero on failure.

Checks
 1. Files match MANIFEST.json sha256; every config is pinned to MuJoCo 3.16.0 and its stored hash matches.
 2. Ids and hashes are unique across all files; no dev/test config hash appears in train.json.
 3. The family rules are pairwise disjoint (cells per object, partition kind, interior-pad counts,
    appearance hue bands and floor textures), and every config obeys its own family (validate()).
 4. One factor at a time: instances = train families + fresh seeds; arrangements = reserved layout family
    + train appearance; appearances = a training config's exact layout/instance with a reserved
    appearance family; distractor = a training config's exact layout/appearance + a screen.
 5. Unseen seeds: no layout or instance seed of instances/arrangements dev/test occurs in train.
 6. (Added in review M1; 2's hash check includes the config id, so it can't fail.) By content: no dev/test
    config equals a training config once ids, seeds and bookkeeping are stripped; no instances/arrangements
    layout or start pose occurs in train; dev and test share no layout, content or source config within a
    factor; no distractor in train; floor, wall and light parameters lie in their appearance family's bands.
"""
import hashlib
import json
import sys
from itertools import combinations
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from configs import APPEARANCE_FAMILIES, ARRANGEMENT_FAMILIES, rgb_to_hsv_deg, validate  # noqa: E402
from room import MUJOCO_VERSION, config_hash  # noqa: E402

SPLITS = HERE / "splits"


def load():
    data = {"train": json.loads((SPLITS / "train.json").read_text())["configs"]}
    for f in ("instances", "arrangements", "appearances", "distractor"):
        d = json.loads((SPLITS / f"{f}.json").read_text())
        data[f] = {"dev": d["dev"], "test": d["test"]}
    return data


def main():
    res, fails = {}, []
    man = json.loads((SPLITS / "MANIFEST.json").read_text())
    sha = {n: hashlib.sha256((SPLITS / n).read_bytes()).hexdigest() for n in man["files_sha256"]}
    res["sha256_match"] = sha == man["files_sha256"]
    data = load()
    train = data["train"]
    allc = [("train", "train", c) for c in train] + [(f, s, c) for f in data if f != "train"
                                                      for s in ("dev", "test") for c in data[f][s]]
    res["n_configs"] = len(allc)
    res["pinned_and_hash_ok"] = all(c["mujoco_version"] == MUJOCO_VERSION and c["hash"] == config_hash(c)
                                    for _, _, c in allc)
    ids = [c["id"] for _, _, c in allc]
    hashes = [c["hash"] for _, _, c in allc]
    res["ids_unique"] = len(set(ids)) == len(ids)
    res["hashes_unique"] = len(set(hashes)) == len(hashes)
    train_hashes = {c["hash"] for c in train}
    res["held_out_in_train"] = sum(c["hash"] in train_hashes for f, s, c in allc if f != "train")

    # 3. families disjoint
    disj = {}
    for key in ("home", "button", "occluder"):
        disj[key] = all(not set(ARRANGEMENT_FAMILIES[a][key]) & set(ARRANGEMENT_FAMILIES[b][key])
                        for a, b in combinations(ARRANGEMENT_FAMILIES, 2))
    disj["partition"] = len({v["partition"] for v in ARRANGEMENT_FAMILIES.values()}) == 3
    ivals = [ARRANGEMENT_FAMILIES[k]["pads_interior"] for k in ARRANGEMENT_FAMILIES]
    disj["pads_interior"] = all(a[1] < b[0] or b[1] < a[0] for a, b in combinations(ivals, 2))
    hb = [APPEARANCE_FAMILIES[k]["hue_offset"] for k in APPEARANCE_FAMILIES]
    disj["shade_hue_bands"] = all(a[1] < b[0] or b[1] < a[0] for a, b in combinations(hb, 2))
    disj["floor_textures"] = len({(v["floor_builtin"], v["floor_mark"]) for v in APPEARANCE_FAMILIES.values()}) == 3
    for k in ("floor_hue", "wall_v", "diffuse"):
        bands = [APPEARANCE_FAMILIES[f][k] for f in APPEARANCE_FAMILIES]
        disj[k] = all(a[1] < b[0] or b[1] < a[0] for a, b in combinations(bands, 2))
    res["families_disjoint"] = disj
    invalid = [(c["id"], p) for _, _, c in allc for p in [validate(c)] if p]
    res["invalid_configs"] = invalid[:10]
    res["n_invalid"] = len(invalid)

    # 4. one factor at a time
    by_id = {c["id"]: c for c in train}
    one = {}
    for f in ("instances", "arrangements", "appearances", "distractor"):
        ok = True
        for s in ("dev", "test"):
            for c in data[f][s]:
                lf, af, dist = c["layout"]["family"], c["appearance"]["family"], c["distractor"] is not None
                if f == "instances":
                    ok &= (lf, af, dist) == ("train", "train", False)
                elif f == "arrangements":
                    ok &= (lf, af, dist) == (s, "train", False)
                elif f == "appearances":
                    src = by_id[c["source_train_id"]]
                    ok &= (lf, af, dist) == ("train", s, False) and c["layout"] == src["layout"]
                else:
                    src = by_id[c["source_train_id"]]
                    ok &= (lf, af, dist) == ("train", "train", True) and c["layout"] == src["layout"] \
                        and c["appearance"] == src["appearance"]
        one[f] = bool(ok)
    res["one_factor_at_a_time"] = one

    # 5. unseen seeds
    tl = {c["seeds"]["layout"] for c in train}
    ti = {c["seeds"]["instance"] for c in train}
    res["seeds_unseen"] = {f: all(c["seeds"]["layout"] not in tl and c["seeds"]["instance"] not in ti
                                  for s in ("dev", "test") for c in data[f][s])
                           for f in ("instances", "arrangements")}

    # 6. by content (review M1)
    idlike = ("id", "hash", "seeds", "seed", "source_train_id", "layout_attempts")
    content = lambda c: json.dumps({k: v for k, v in c.items() if k not in idlike}, sort_keys=True)  # noqa: E731
    layout = lambda c: json.dumps(c["layout"], sort_keys=True)  # noqa: E731
    start = lambda c: json.dumps(c["layout"]["start"], sort_keys=True)  # noqa: E731
    tc, tl_, ts = {content(c) for c in train}, {layout(c) for c in train}, {start(c) for c in train}
    held = [(f, s, c) for f, s, c in allc if f != "train"]
    res["held_out_content_in_train"] = sum(content(c) in tc for _, _, c in held)
    fresh = [c for f, s, c in held if f in ("instances", "arrangements")]
    res["fresh_layouts_in_train"] = sum(layout(c) in tl_ for c in fresh)
    res["fresh_start_poses_in_train"] = sum(start(c) in ts for c in fresh)
    res["dev_test_shared"] = {f: len({layout(c) for c in data[f]["dev"]} & {layout(c) for c in data[f]["test"]})
                              + len({content(c) for c in data[f]["dev"]} & {content(c) for c in data[f]["test"]})
                              + len({c.get("source_train_id") for c in data[f]["dev"]}
                                    & {c.get("source_train_id") for c in data[f]["test"]} - {None})
                              for f in data if f != "train"}
    res["distractor_in_train"] = sum(c["distractor"] is not None for c in train)
    band_bad = []
    for _, _, c in allc:
        a = c["appearance"]
        fam = APPEARANCE_FAMILIES[a["family"]]
        fh, fs, fv = rgb_to_hsv_deg(a["floor"]["rgb1"])
        vals = {"floor_hue": (fh, 1.0), "floor_v": (fv, 0.01), "wall_v": (rgb_to_hsv_deg(a["wall_rgb"])[2], 0.01),
                "diffuse": (a["light"]["diffuse"][0], 1e-3), "ambient": (a["light"]["ambient"][0], 1e-3),
                "texrepeat": (a["floor"]["texrepeat"], 0)}
        for k, (x, tol) in vals.items():
            lo, hi = fam[k]
            if not lo - tol <= x <= hi + tol:
                band_bad.append((c["id"], k))
    res["appearance_band_violations"] = band_bad[:10]
    content_ok = (res["held_out_content_in_train"] == 0 and res["fresh_layouts_in_train"] == 0
                  and res["fresh_start_poses_in_train"] == 0 and not any(res["dev_test_shared"].values())
                  and res["distractor_in_train"] == 0 and not band_bad)

    # summary
    checks = [content_ok, res["sha256_match"], res["pinned_and_hash_ok"], res["ids_unique"], res["hashes_unique"],
              res["held_out_in_train"] == 0, all(disj.values()), res["n_invalid"] == 0,
              all(one.values()), all(res["seeds_unseen"].values())]
    res["pass"] = bool(all(checks))
    out = HERE / "results" / "m1" / "split_check.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))
    if not res["pass"]:
        fails.append("split check failed")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
