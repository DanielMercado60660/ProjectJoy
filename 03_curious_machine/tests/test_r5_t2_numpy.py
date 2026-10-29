"""Turn-2 tests use NumPy/Numba only; no MLX import, no training."""
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pytest
from env import VecRooms, OBS_DIM
from r5_core import TASKS, Frontier
from evaluate_r5 import task_success,evaluate
from planner_r5_t2 import Knowledge,CompositionalPlanner
from progress_r5_t2 import ErrorQueue,RewardScale


@pytest.mark.parametrize('tv',[False,True])
def test_full_knowledge_reaches_every_task_within_128(tv):
    env=VecRooms(1,max_steps=513,noisy_tv=tv,seed=64600)
    frontier=Frontier(64600,mechanism_bonus=6,frontier_bonus=0);frontier.observe(env.obs[0])
    observations=[env.obs[0].copy()];actions=[]
    for _ in range(512):
        action=frontier.act();obs,_,_=env.step([action]);frontier.observe(obs[0],action)
        observations.append(obs[0].copy());actions.append(action)
    knowledge=Knowledge.from_replay(observations,actions)
    assert knowledge.stats()['links']==5
    for source,k in [('replay',knowledge),('oracle',Knowledge.oracle(env))]:
        for task in TASKS:
            planner=CompositionalPlanner(k);plan,expanded=planner.plan(task)
            assert plan is not None and len(plan)<=128 and expanded<=100000,(source,task)
            fresh=VecRooms(1,max_steps=129,noisy_tv=tv,seed=64601)
            for action in plan:fresh.step([action])
            assert task_success(fresh,[task])[0],(source,task,plan)


def test_missing_causal_knowledge_cannot_open_remote_door():
    k=Knowledge.oracle(VecRooms(1));k.links={}
    plan,_=CompositionalPlanner(k).plan(('reach',8))
    assert plan is None


def test_planner_hard_compute_and_action_limits():
    k=Knowledge.oracle(VecRooms(1));p=CompositionalPlanner(k)
    assert p.plan(('reach',8),max_expansions=1)==(None,1)
    plan,expanded=p.plan(('reach',8),budget=8,max_expansions=1000)
    assert plan is None and expanded<=1000


def test_normal_evaluation_cannot_call_oracle(monkeypatch):
    def forbidden(*args,**kwargs):raise AssertionError('oracle leaked into ordinary evaluation')
    monkeypatch.setattr(Knowledge,'oracle',forbidden)
    r,_=evaluate('frontier',[64600],tv=True,explore_steps=32,task_steps=16,
                 max_expansions=2000,planner_type='compositional',policy_seed=83519)
    assert r['knowledge_stats'][0]['source']=='replay'
    assert r['planner_type']=='compositional'
    assert r['policy_rng_seed']==83519


def test_queue_wrap_and_rng_resume_preserve_targets():
    q=ErrorQueue(3,OBS_DIM);obs=np.zeros((5,OBS_DIM),np.float32)
    obs[np.arange(5),np.arange(5)]=1
    q.add(obs,np.arange(5,dtype=np.int32),np.arange(5,dtype=np.float32)*-.5)
    assert q.count==3
    a=ErrorQueue(3,OBS_DIM);a.load(q.state())
    x,actions,labels=a.sample(np.random.default_rng(8),100)
    assert set(actions)=={2,3,4}
    np.testing.assert_array_equal(x.argmax(1),actions)
    np.testing.assert_array_equal(labels,actions*-.5)
    for left,right in zip(q.sample(np.random.default_rng(9),30),a.sample(np.random.default_rng(9),30)):
        np.testing.assert_array_equal(left,right)
    with pytest.raises(AssertionError):q.add(np.full((1,OBS_DIM),.5),np.array([0]),np.array([0.]))


def test_adaptive_scale_recovers_after_large_early_rewards_and_zero_safe():
    norm=RewardScale(decay=.5)
    out,scale=norm.normalize(np.zeros(4),np.ones(4))
    assert scale==.1 and np.isfinite(out).all() and not out.any()
    norm.normalize(np.array([-100.,100.]),np.ones(2))
    for _ in range(40):out,scale=norm.normalize(np.array([-.02,.02,999.]),[1,1,0])
    assert scale==.1
    np.testing.assert_allclose(out,[-.2,.2,0],atol=1e-6)
    state=norm.__dict__.copy();resumed=RewardScale();resumed.__dict__.update(state)
    np.testing.assert_array_equal(norm.normalize([.1,-.1],[1,1])[0],resumed.normalize([.1,-.1],[1,1])[0])


def test_new_splits_are_disjoint_and_manifest_round_trip(tmp_path):
    from run_r5_t2 import SEEDS,VAL_SEEDS,TEST_SEEDS,freeze
    from r5_core import TRAIN_SEEDS as OLD_TRAIN, VAL_SEEDS as OLD_VAL, TEST_SEEDS as OLD_TEST
    sets=[set(x) for x in (SEEDS,VAL_SEEDS,TEST_SEEDS,OLD_TRAIN,OLD_VAL,OLD_TEST)]
    for i,a in enumerate(sets):
        for b in sets[i+1:]:assert a.isdisjoint(b)
    p=tmp_path/'manifest.json';freeze(p,dict(seeds=SEEDS));freeze(p,dict(seeds=SEEDS))
    with pytest.raises(ValueError):freeze(p,dict(seeds=(999,)))


def test_child_treats_guard_stop_as_partial_and_error_as_error(tmp_path):
    import time
    from run_r5_t2 import child
    assert child(['-c','raise SystemExit(2)'],tmp_path/'partial.log',time.monotonic()+10) is False
    with pytest.raises(RuntimeError):child(['-c','raise SystemExit(1)'],tmp_path/'error.log',time.monotonic()+10)


def test_deadline_stops_owned_descendants(tmp_path):
    import time
    from run_r5_t2 import child
    heartbeat=tmp_path/'heartbeat'
    grandchild='import time; from pathlib import Path; p=Path('+repr(str(heartbeat))+');\nwhile True:\n p.write_text(str(time.time())); time.sleep(.05)'
    code='import subprocess,sys,time; subprocess.Popen([sys.executable,"-c",'+repr(grandchild)+']); time.sleep(30)'
    assert child(['-c',code],tmp_path/'timeout.log',time.monotonic()+8.1,group=True) is False
    before=heartbeat.read_text();time.sleep(.2)
    assert heartbeat.read_text()==before
