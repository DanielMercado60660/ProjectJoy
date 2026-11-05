"""CPU-only checks of the production data helpers; no torch import or network."""
import ast
import collections
import glob
import hashlib
import json
import math
import os
from pathlib import Path
import numpy as np
import pytest


@pytest.fixture
def helpers(tmp_path):
    tree = ast.parse(Path(__file__).with_name("emilia_expansion.py").read_text())
    names = {"atomic_json", "sha256_file", "pair_ok", "emilia_split", "metadata_rejection",
             "encoded_inventory", "training_hours", "emilia_id_owners", "commit_paths", "preparation_deadline"}
    code = ast.Module(body=[n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names], type_ignores=[])
    env = dict(os=os, json=json, hashlib=hashlib, glob=glob, math=math, collections=collections,
               WORK=str(tmp_path), CFG={"SETS":["lj","emilia_yodas"],"LJ_REPEAT":2},
               EMILIA_SET="emilia_yodas", POLICY_ID="pinned", DATA_REPO="private/test")
    exec(compile(code,"emilia_expansion.py","exec"),env)
    return env


def write_pair(h, source="emilia_yodas", name="chunk", split="train", cid="EN_vid_W000001"):
    jp = Path(h["WORK"])/"data"/"ming"/source/(name+".json")
    jp.parent.mkdir(parents=True,exist_ok=True)
    fp=jp.with_suffix(".f16")
    np.zeros((3,128),np.float16).tofile(fp)
    m=dict(frames=3,D=64,ids=[cid],texts=["hello"],speakers=["emilia:a"],splits=[split],
           offs=[0,3],hours=1.5,policy_id="pinned",source_archive="a.tar",sha256=h["sha256_file"](str(fp)))
    h["atomic_json"](str(jp),m)
    return str(jp),m


def test_chunk_requires_complete_consistent_pair(helpers):
    jp,m=write_pair(helpers)
    assert helpers["pair_ok"](jp,True)==m
    Path(jp[:-5]+".f16").write_bytes(b"truncated")
    assert helpers["pair_ok"](jp) is None
    assert not Path(jp+".partial").exists()


def test_chunk_hash_rejects_same_length_corruption(helpers):
    jp,_=write_pair(helpers)
    Path(jp[:-5]+".f16").write_bytes(b"x"*(3*128*2))
    assert helpers["pair_ok"](jp,True) is None


def test_source_group_stays_together(helpers):
    split=helpers["emilia_split"]
    assert len({split("EN_vid_with_underscores_W%06d"%i) for i in range(100)})==1
    counts=collections.Counter(split(f"EN_{i}_W000001") for i in range(10000))
    assert 60 < counts["test"] < 140 and 60 < counts["val"] < 140
    assert counts["train"] > 9700


def test_filter_rejects_bad_metadata(helpers):
    f=helpers["metadata_rejection"]
    good=dict(language="en",duration=8,dnsmos=3.5,text="Hello there.",speaker="s")
    assert f(good) is None
    for patch in ({"dnsmos":float("nan")},{"duration":float("inf")},{"language":"de"},
                  {"duration":21},{"dnsmos":3.2},{"text":""},{"speaker":""}):
        assert f(good|patch) is not None


def test_hours_exclude_holdouts_and_repeats(helpers):
    write_pair(helpers,source="lj",name="train")
    write_pair(helpers,name="train",cid="a")
    write_pair(helpers,name="val",split="val",cid="b")
    write_pair(helpers,name="test",split="test",cid="c")
    inv=helpers["encoded_inventory"]()
    assert helpers["training_hours"](inv)==3.0
    assert inv["emilia_yodas"]["hours"]["val"]==1.5


def test_duplicate_encoded_ids_fail_audit(helpers):
    write_pair(helpers,name="one")
    write_pair(helpers,name="two")
    with pytest.raises(AssertionError,match="Duplicate encoded"):
        helpers["emilia_id_owners"]()


def test_failed_upload_is_retryable(helpers):
    calls=[]
    class API:
        def create_commit(self,**kw):
            calls.append(kw)
            if len(calls)==1:
                raise ConnectionError("interrupted")
    helpers["CommitOperationAdd"] = lambda **kw:kw
    remote=set()
    args=(API(),["emilia_yodas/a.f16","emilia_yodas/a.json"],remote,"save")
    with pytest.raises(ConnectionError):
        helpers["commit_paths"](*args)
    assert not remote
    helpers["commit_paths"](*args)
    assert len(remote)==2 and len(calls[1]["operations"])==2
    helpers["commit_paths"](*args)
    assert len(calls)==2


def test_continuous_preparation_has_no_timed_stop(helpers):
    deadline=helpers["preparation_deadline"]
    assert math.isinf(deadline(None,100))
    assert deadline(40,100)==2500
    assert deadline(600,100)==36100
    for bad in (0,-1,float("nan"),float("inf")):
        with pytest.raises(ValueError):
            deadline(bad,100)
