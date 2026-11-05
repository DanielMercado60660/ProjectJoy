"""Paste/run after the base notebook definitions. Encoding only; never starts training.

Emilia-YODAS EN -> bounded Ming posterior chunks, private HF backups, measured
unique training hours. Runs to the agreed target; rerun after a Colab disconnect.
"""
import collections
import hashlib
import pathlib
import tarfile
import gc
from huggingface_hub import HfApi, CommitOperationAdd, hf_hub_download

EMILIA_SET = "emilia_yodas"
EMILIA_REPO = "amphion/Emilia-Dataset"
EMILIA_REV = "d7f2f7340a6385696f3766c8049fa920a4707c07"
BASE_SETS = ["lj", "libritts_clean100", "libritts_clean360", "libritts_other500", "hifi_clean", "hifi_other"]
EMILIA_POLICY = dict(version=1, repo=EMILIA_REPO, revision=EMILIA_REV,
    prefix="Emilia-YODAS/EN/", language="en", min_s=0.8, max_s=20.0,
    min_dnsmos=3.3, min_chars=3, max_chars=1000, chunk_clips=512,
    seed=20261103, split="sha256(source video): 1% test, 1% val, 98% train",
    codec="ming", codec_code=MING_COMMIT)
POLICY_ID = hashlib.sha256(json.dumps(EMILIA_POLICY, sort_keys=True).encode()).hexdigest()
# User chose an intermediate listening/training milestone on 2026-11-03.
# Increasing this later reuses all completed encoded chunks.
TARGET_TRAIN_HOURS = 2500.0
if "_source_shards_original" not in globals():
    _source_shards_original = source_shards


def atomic_json(path, value):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path + ".partial", "w") as f:
        json.dump(value, f, ensure_ascii=False)
    os.replace(path + ".partial", path)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def pair_ok(jp, check_hash=False):
    try:
        with open(jp) as f:
            m = json.load(f)
        fp = jp[:-5] + ".f16"
        assert os.path.getsize(fp) == m["frames"] * 2 * m["D"] * 2
        n = len(m["ids"])
        assert all(len(m[k]) == n for k in ("texts", "speakers", "splits"))
        assert len(m["offs"]) == n + 1 and m["offs"][0] == 0 and m["offs"][-1] == m["frames"]
        assert all(a <= b for a, b in zip(m["offs"], m["offs"][1:]))
        if check_hash and "sha256" in m:
            assert sha256_file(fp) == m["sha256"]
        return m
    except (OSError, ValueError, KeyError, AssertionError):
        return None


def emilia_split(cid):
    # Source video IDs can contain underscores, so split from the RIGHT.
    group = cid.rsplit("_W", 1)[0]
    bucket = int(hashlib.sha256(group.encode()).hexdigest()[:8], 16) % 10000
    return "test" if bucket < 100 else "val" if bucket < 200 else "train"


def metadata_rejection(m):
    try:
        dur, score = float(m["duration"]), float(m["dnsmos"])
        text = str(m["text"]).strip()
        if m["language"].lower() != "en":
            return "language"
        if not math.isfinite(dur) or not 0.8 <= dur <= 20:
            return "duration"
        if not math.isfinite(score) or score < 3.3:
            return "dnsmos"
        if not 3 <= len(text) <= 1000:
            return "text"
        if not m.get("speaker"):
            return "speaker"
    except (KeyError, TypeError, ValueError):
        return "metadata"
    return None


def encoded_inventory():
    summary = {}
    for s in CFG["SETS"]:
        by_split = collections.defaultdict(float)
        n = clips = 0
        for jp in sorted(glob.glob(os.path.join(WORK, "data", "ming", s, "*.json"))):
            m = pair_ok(jp)
            assert m is not None, f"Incomplete encoded pair: {jp}"
            splits = set(m["splits"])
            assert len(splits) <= 1, f"Mixed split stats are unsafe: {jp}"
            if s == EMILIA_SET:
                assert m.get("policy_id") == POLICY_ID, f"Encoding policy changed: {jp}"
            if splits:
                by_split[next(iter(splits))] += m["hours"]
            n += 1
            clips += len(m["ids"])
        summary[s] = dict(shards=n, clips=clips, hours=dict(by_split))
    return summary


def training_hours(summary):
    return sum(s["hours"].get("train", 0.0) for s in summary.values())


def commit_paths(api, rels, remote, message):
    root = os.path.join(WORK, "data", "ming")
    missing = [r for r in rels if r not in remote]
    if missing:
        api.create_commit(repo_id=DATA_REPO, repo_type="dataset", commit_message=message,
            operations=[CommitOperationAdd(path_in_repo=r, path_or_fileobj=os.path.join(root, r)) for r in missing])
        remote.update(missing)


def restore_and_reconcile(api, remote):
    """Remote JSON+audio pairs commit together; repair old base uploads interrupted between files."""
    root = os.path.join(WORK, "data", "ming")
    restored = 0
    for rel in sorted(remote):
        if not rel.endswith(".json") or rel.split("/")[0] not in CFG["SETS"]:
            continue
        fp = rel[:-5] + ".f16"
        assert fp in remote, f"Remote metadata has no audio: {rel}"
        jp = os.path.join(root, rel)
        if not pair_ok(jp):
            for r in (fp, rel):
                hf_hub_download(DATA_REPO, r, repo_type="dataset", local_dir=root)
            assert pair_ok(jp, check_hash=True), f"Restored data failed verification: {rel}"
            restored += 1
    pending = []
    for s in CFG["SETS"]:
        for jp in sorted(glob.glob(os.path.join(root, s, "*.json"))):
            if pair_ok(jp):
                rel = os.path.relpath(jp, root)
                pending.extend([rel[:-5] + ".f16", rel])
    for i in range(0, len(pending), 100):
        commit_paths(api, pending[i:i+100], remote, "Resume: persist complete encoded pairs")
    print("Restore verified:", restored, "pairs downloaded; complete local pairs backed up", flush=True)


def emilia_id_owners():
    owners = {}
    for jp in sorted(glob.glob(os.path.join(WORK,"data","ming",EMILIA_SET,"*.json"))):
        m = pair_ok(jp)
        assert m and m.get("policy_id") == POLICY_ID
        for cid in m["ids"]:
            assert cid not in owners, f"Duplicate encoded clip: {cid}"
            owners[cid] = m["source_archive"]
    return owners


def encode_emilia_chunk(tar, rows, shard, archive, split):
    fp, jp = shard_paths(EMILIA_SET, shard)
    prior = pair_ok(jp)
    if prior:
        assert prior.get("policy_id") == POLICY_ID
        return False
    cod = get_codec()
    clips, rejected = [], collections.Counter()
    for cid, meta, audio_member in rows:
        try:
            x, sr = sf.read(io.BytesIO(tar.extractfile(audio_member).read()), dtype="float32")
        except (RuntimeError, ValueError, OSError):
            rejected["decode"] += 1
            continue
        if x.ndim == 2:
            x = x.mean(axis=1)
        dur = len(x) / sr
        if not 0.8 <= dur <= 20.0 or abs(dur - float(meta["duration"])) > 0.25:
            rejected["decoded_duration"] += 1
            continue
        if not np.isfinite(x).all() or np.max(np.abs(x), initial=0) < 1e-5:
            rejected["invalid_audio"] += 1
            continue
        clips.append((cid, resample(x, sr, cod.sr), str(meta["text"]).strip(),
                      "emilia:" + str(meta["speaker"]), dur))
    clips.sort(key=lambda c: (len(c[1]), c[0]))
    lats = []
    batched = batching_ok(cod, [c[1] for c in clips[:2] + clips[-1:]]) if clips else False
    i = 0
    with torch.no_grad():
        while i < len(clips):
            j = i
            while batched and j < len(clips) and (j-i+1)*len(clips[j][1]) <= CFG["ENC_BATCH_S"]*cod.sr:
                j += 1
            j = max(j, i+1)
            lats.extend(cod.encode([c[1] for c in clips[i:j]]))
            i = j
    z = np.concatenate(lats).astype(np.float32) if lats else np.zeros((0, 2*cod.D), np.float32)
    assert np.isfinite(z).all() and (z[:, cod.D:] >= 0).all()
    z16 = z.astype(np.float16)
    assert np.isfinite(z16).all(), "Float16 posterior overflow"
    z16.tofile(fp + ".partial")
    os.replace(fp + ".partial", fp)
    mean = z[:, :cod.D].astype(np.float64)
    meta = dict(codec=cod.name, D=cod.D, frames=len(z), ids=[c[0] for c in clips],
        texts=[c[2] for c in clips], speakers=[c[3] for c in clips], splits=[split]*len(clips),
        offs=np.concatenate([[0], np.cumsum([len(x) for x in lats])]).tolist(),
        sum=mean.sum(0).tolist(), sumsq=(mean**2).sum(0).tolist(),
        sum_var=(z[:, cod.D:].astype(np.float64)**2).sum(0).tolist(),
        hours=sum(c[4] for c in clips)/3600, duration_s=[c[4] for c in clips],
        policy_id=POLICY_ID, source_repo=EMILIA_REPO, source_revision=EMILIA_REV,
        source_archive=archive, rejected=dict(rejected), sha256=sha256_file(fp))
    atomic_json(jp, meta)  # JSON is the completion marker, written last.
    print(shard, len(clips), "clips", round(meta["hours"],3), "hours", flush=True)
    return True


def preparation_deadline(minutes, start):
    """None means run to the data target, subject to Colab's own runtime limits."""
    if minutes is None:
        return math.inf
    if not math.isfinite(minutes) or minutes <= 0:
        raise ValueError("minutes must be None or a positive finite number")
    return start + minutes*60


def prepare_large_data(minutes=None, max_new_chunks=None):
    """Prepare the agreed dataset. Target counts unique, filtered TRAIN audio only."""
    global DATA_READY, source_shards
    assert not SMOKE and CFG["CODEC"] == "ming" and "LIMIT" not in CFG
    assert CFG["PUSH"], "The large run requires durable private backups"
    start = time.monotonic()
    deadline = preparation_deadline(minutes, start)
    DATA_READY = False
    CFG["SETS"] = BASE_SETS + [EMILIA_SET]
    CFG["TARGET_TRAIN_HOURS"] = TARGET_TRAIN_HOURS
    _JUDGE.clear(); _CODECS.pop("mimi", None); STATE.clear()
    gc.collect(); torch.set_num_threads(4)
    if DEV == "cuda":
        torch.cuda.empty_cache()
    api = HfApi()
    assert api.repo_info(DATA_REPO, repo_type="dataset").private, "Expected the private experiment repo"
    remote = set(api.list_repo_files(DATA_REPO, repo_type="dataset"))
    root = os.path.join(WORK, "data", "ming")
    policy_rel = "preparation/emilia-policy.json"
    if policy_rel in remote:
        p = hf_hub_download(DATA_REPO, policy_rel, repo_type="dataset", local_dir=root)
        assert json.load(open(p)) == EMILIA_POLICY, "Use a new dataset repo for a changed preparation policy"
    else:
        atomic_json(os.path.join(root, policy_rel), EMILIA_POLICY)
        commit_paths(api, [policy_rel], remote, "Pin Emilia-YODAS preparation policy")
    restore_and_reconcile(api, remote)
    owners = emilia_id_owners()
    new_chunks = 0
    print("PREPARATION CONFIG", json.dumps(dict(target_train_hours=TARGET_TRAIN_HOURS,
        timed_stop_minutes=minutes, backups="private HF after each base shard / <=16 Emilia chunks")), flush=True)
    print("Starting accepted training hours:", round(training_hours(encoded_inventory()),3), flush=True)
    # Include the original corpora in the target BEFORE adding Emilia. This gives
    # an actual 2,500-hour mixed-data milestone, not 2,500 plus another ~700 hours.
    base_complete = True
    for s in BASE_SETS:
        for shard, loader in _source_shards_original(s):
            if time.monotonic() >= deadline:
                base_complete = False
                break
            fp, jp = shard_paths(s, shard)
            if not pair_ok(jp):
                if os.path.exists(jp):
                    # Preserve invalid metadata for inspection; allow the base
                    # encoder to regenerate this disposable incomplete pair.
                    os.replace(jp, jp + ".incomplete")
                print("Encoding base corpus", s, shard, flush=True)
                old_push = CFG["PUSH"]
                try:
                    source_shards = lambda _: [(shard, loader)]
                    CFG["PUSH"] = False
                    encode_set(s)
                finally:
                    source_shards = _source_shards_original
                    CFG["PUSH"] = old_push
            assert pair_ok(jp), f"Incomplete base shard: {s}/{shard}"
            commit_paths(api, [f"{s}/{shard}.f16",f"{s}/{shard}.json"], remote, f"Encode {s}/{shard}")
        if not base_complete:
            break
        print("BASE SOURCE READY",s,"| total accepted train hours:",
              round(training_hours(encoded_inventory()),3),flush=True)
    files = sorted(f for f in api.list_repo_files(EMILIA_REPO, repo_type="dataset", revision=EMILIA_REV)
                   if f.startswith("Emilia-YODAS/EN/") and f.endswith(".tar"))
    assert files, "No English Emilia-YODAS archives found"
    random.Random(EMILIA_POLICY["seed"]).shuffle(files)
    # Start with the already inspected archive, then use the seeded permutation.
    probe = "Emilia-YODAS/EN/EN-B000000.tar"
    files = [probe] + [f for f in files if f != probe]
    print("English archives:", len(files), "| target:", TARGET_TRAIN_HOURS, "unique training hours", flush=True)
    for archive in files:
        summary = encoded_inventory()
        if training_hours(summary) >= TARGET_TRAIN_HOURS or time.monotonic() >= deadline:
            break
        name = pathlib.PurePosixPath(archive).stem
        receipt = f"preparation/emilia/{name}.json"
        if receipt in remote:
            continue
        if shutil.disk_usage(WORK).free < 8*2**30:
            raise RuntimeError("Less than 8 GiB free; stop safely before the next archive")
        path = hf_hub_download(EMILIA_REPO, archive, repo_type="dataset", revision=EMILIA_REV,
                               token=True, local_dir=os.path.join(WORK,"raw","emilia"))
        pending, complete = [], True
        with tarfile.open(path) as tar:
            members = {m.name:m for m in tar.getmembers() if m.isfile()}
            groups = {s:[] for s in ("train", "val", "test")}
            rejected, seen = collections.Counter(), set()
            for key in sorted(members):
                if not key.endswith(".json"):
                    continue
                meta = json.load(tar.extractfile(members[key]))
                cid = pathlib.PurePosixPath(key).stem
                assert meta.get("_id", meta.get("id", cid)) == cid, "Archive identity mismatch"
                assert cid not in seen, "Duplicate clip ID in archive"
                seen.add(cid)
                if cid in owners and owners[cid] != archive:
                    rejected["duplicate_id"] += 1
                    continue
                reason = metadata_rejection(meta)
                if reason:
                    rejected[reason] += 1
                    continue
                audio = key[:-5] + ".mp3"
                if audio not in members:
                    rejected["missing_audio"] += 1
                    continue
                groups[emilia_split(cid)].append((cid, meta, members[audio]))
            print(name, "metadata retained", {s:len(v) for s,v in groups.items()},
                  "rejected", dict(rejected), flush=True)
            stop = False
            for split, rows in groups.items():
                for offset in range(0, len(rows), EMILIA_POLICY["chunk_clips"]):
                    if time.monotonic() >= deadline or (max_new_chunks is not None and new_chunks >= max_new_chunks):
                        complete = False; stop = True; break
                    shard = f"{name}-{split}-{offset//EMILIA_POLICY['chunk_clips']:05d}"
                    new = encode_emilia_chunk(tar, rows[offset:offset+EMILIA_POLICY["chunk_clips"]], shard, archive, split)
                    new_chunks += int(new)
                    chunk_meta = pair_ok(shard_paths(EMILIA_SET,shard)[1])
                    for cid in chunk_meta["ids"]:
                        owners[cid] = archive
                    pending.extend([f"{EMILIA_SET}/{shard}.f16", f"{EMILIA_SET}/{shard}.json"])
                    if len(pending) >= 32:
                        commit_paths(api, pending, remote, f"Encode {name}: progress")
                        pending.clear()
                if stop:
                    break
            if complete:
                atomic_json(os.path.join(root,receipt), dict(archive=archive, revision=EMILIA_REV,
                    policy_id=POLICY_ID, metadata_clips=len(seen), rejected=dict(rejected)))
                pending.append(receipt)
        commit_paths(api, pending, remote, f"Encode {name}: {'complete' if complete else 'paused'}")
        if complete:
            os.remove(path)  # Disposable download; encoded pairs are verified and backed up.
        summary = encoded_inventory()
        print("PREPARATION", json.dumps(summary), "\nUnique training hours:", round(training_hours(summary),3), flush=True)
        if not complete:
            break
    summary = encoded_inventory()
    if base_complete and training_hours(summary) >= TARGET_TRAIN_HOURS:
        audit_large_data()
        DATA_READY = True
    report = dict(policy_id=POLICY_ID, target_train_hours=TARGET_TRAIN_HOURS,
                  unique_train_hours=training_hours(summary), sources=summary,
                  new_chunks=new_chunks, session_minutes=(time.monotonic()-start)/60,
                  data_ready=DATA_READY, timed_stop_minutes=minutes)
    atomic_json(os.path.join(WORK,"preparation_progress.json"),report)
    print("SESSION SAVED", json.dumps(report), flush=True)
    if DATA_READY:
        print("DATA READY: target reached, original corpora complete, backup/identity audit passed. Main training has not started.", flush=True)
    else:
        print("Preparation incomplete. Rerun prepare_large_data() to continue. Main training has not started.", flush=True)
    return report


def audit_large_data():
    """Fail closed until the requested dataset and backups are complete. No training here."""
    summary = encoded_inventory()
    assert training_hours(summary) >= TARGET_TRAIN_HOURS, f"Continue preparation: fewer than {TARGET_TRAIN_HOURS:,.0f} unique training hours"
    for s in BASE_SETS:
        for shard, _ in _source_shards_original(s):
            assert pair_ok(shard_paths(s,shard)[1]), f"Original corpus incomplete: {s}/{shard}"
    emilia_id_owners()  # Enforce unique clip IDs across archives/chunks.
    remote = set(HfApi().list_repo_files(DATA_REPO, repo_type="dataset"))
    root = os.path.join(WORK,"data","ming")
    manifest = []
    for s in CFG["SETS"]:
        for jp in sorted(glob.glob(os.path.join(root,s,"*.json"))):
            rel = os.path.relpath(jp,root)
            assert rel in remote and rel[:-5]+".f16" in remote, f"Backup incomplete: {rel}"
            manifest.append(dict(path=rel,metadata_sha256=sha256_file(jp)))
    result = dict(policy_id=POLICY_ID, unique_train_hours=training_hours(summary),
                  sources=summary, shards=manifest)
    atomic_json(os.path.join(WORK,"large_data_audit.json"),result)
    print("Data audit passed. Set training steps from measured epoch length and validate cold checkpoint resume before the long run.", flush=True)
    return result
