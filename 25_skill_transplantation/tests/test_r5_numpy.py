"""Pure NumPy protocol tests. Never import common.py, ppo.py or r5_train.py."""
import ast
from pathlib import Path
import sys
import tempfile
import time

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from r5_core import (CPG_HI, CPG_LO, TARGETS, Deadline, Replay, add_adapter, atomic_pickle,
                     bank, budget, deployment_params, forward, gae, gait, init_params, inputs, logstd,
                     normalize_advantages, reset_exploration, scheduled_lr, training_reward,
                     orders, read_pickle, retention, reward, schedule, segment_metrics,
                     size_counts, sustained, train_keys)


def test_seeds_fresh_disjoint():
    sets = [set(bank(split, seed, 10000)) for split in ('train','cpg','val','test','test_t2','smoke') for seed in range(3)]
    for i, a in enumerate(sets):
        assert not a.intersection(bank('legacy', 0, 10))
        for b in sets[i+1:]:
            assert not a.intersection(b)
    with pytest.raises(ValueError):
        bank('test', 100)


def test_input_scaling_and_labels():
    x = inputs(np.ones((3,17)), [0,1,2])
    assert x.shape == (3,20) and x.dtype == np.float32
    np.testing.assert_array_equal(x[:,-3:], np.eye(3))
    assert inputs(np.full((1,17), 1e9), [0])[:,:17].max() == 10


def test_rewards_make_stopping_costly_and_new_skill_distinct():
    a = np.zeros((3,6))
    np.testing.assert_allclose(reward(np.arange(3), TARGETS, a), 1)
    assert np.all(reward(np.arange(3), np.zeros(3), a) < 1)
    assert reward(2,1,np.ones(6)) < reward(2,1,np.zeros(6))
    assert reward(0,-3,np.zeros(6)) < reward(0,3,np.zeros(6))


def test_adapter_identity_and_old_routes_unchanged():
    p = init_params(1, 12)
    q = add_adapter(p, 1, 3, 2)
    x = inputs(np.random.default_rng(4).normal(size=(8,17)), np.arange(8)%3)
    tids = np.arange(8)%3
    assert all(np.array_equal(p[k], q[k]) for k in p)
    np.testing.assert_allclose(forward(p,x,tids), forward(q,x,tids), atol=1e-7, rtol=1e-6)
    for key in train_keys(q,'adapter',1):
        q[key] += .1
    np.testing.assert_allclose(forward(p,x[tids!=1],tids[tids!=1]), forward(q,x[tids!=1],tids[tids!=1]), atol=1e-7, rtol=1e-6)
    assert not np.allclose(forward(p,x[tids==1],tids[tids==1]), forward(q,x[tids==1],tids[tids==1]))
    r = add_adapter(q,2,3,3)
    for key in train_keys(r,'adapter',2):
        r[key] += .1
    np.testing.assert_allclose(forward(q,x[tids!=2],tids[tids!=2]), forward(r,x[tids!=2],tids[tids!=2]), atol=1e-7, rtol=1e-6)
    assert np.isfinite(logstd(r,tids)).all()
    assert set(train_keys(r,'dagger',2)) == {k for k in r if k.startswith('a')}
    assert size_counts(r,3)['actor_parameters'] > size_counts(p,1)['actor_parameters']


def test_gae_resets_and_timeout_bootstrap_contract():
    r = np.array([[1.],[2.]], np.float32)
    v = np.array([[.3],[.4],[.5]], np.float32)
    a, ret = gae(r,v,np.array([[0.],[1.]]),gamma=1,lam=1)
    np.testing.assert_allclose(ret[:,0],[3,2])
    # Caller adds final-observation bootstrap to timeout reward before setting done.
    r[-1] += .8
    _, ret = gae(r,v,np.array([[1.],[1.]]),gamma=1,lam=1)
    np.testing.assert_allclose(ret[:,0],[1,2.8])


def test_replay_retains_old_tasks_and_is_reproducible():
    replay = Replay(3)
    for tid in (0,1,2):
        replay.add(np.ones((8,20))*tid,np.ones((8,6))*tid,np.full(8,tid))
    assert replay.count.tolist() == [3,3,3]
    a = replay.sample(np.random.default_rng(7),300)
    b = replay.sample(np.random.default_rng(7),300)
    for x,y in zip(a,b):
        np.testing.assert_array_equal(x,y)
    for tid in range(3):
        assert 60 < np.sum(a[2]==tid) < 140
        assert (a[0][a[2]==tid] == tid).all()


def test_switches_no_off_by_one_all_pairs_covered():
    assert [schedule(t,(0,2,1),2) for t in range(6)] == [0,0,2,2,1,1]
    pairs = {(o[i],o[i+1]) for o in orders(3) for i in range(2)}
    assert pairs == {(i,j) for i in range(3) for j in range(3) if i!=j}


def test_switch_failure_penalty_and_sustained_windows():
    h, pitch = np.ones(60)*.7, np.zeros(60)
    good = segment_metrics(np.ones(60)*3,h,pitch,3)
    assert good['settle_s'] == .5 and not good['failure']
    stopped = segment_metrics(np.zeros(60),h,pitch,3)
    assert stopped['stall'] and stopped['failure'] and stopped['settle_s'] is None
    assert stopped['capped_settle_s'] == 3.
    brief = np.r_[np.ones(9)*3, np.zeros(51)]
    assert not segment_metrics(brief,h,pitch,3)['reached']
    fall = segment_metrics(np.ones(60)*3,np.zeros(60),pitch,3)
    assert fall['fall_proxy'] and fall['failure']
    wrapped = segment_metrics(np.ones(60)*3,h,np.ones(60)*2*np.pi,3)
    assert not wrapped['fall_proxy']
    assert sustained([1,1,0,1,1,1],3) == 3


def test_exact_teacher_inclusive_budget():
    b = budget(64,128,32)
    assert b['addition'] == 192 and b['final'] == 448
    assert 64 + 2*(128+32+32) == b['final']
    assert 64 + 2*b['addition'] == b['final']


def test_retention_never_calls_failed_acquisition_a_success():
    rows = [dict(method='dagger',stage=1,task='fwd',kind='isolated',tracking_fraction=0),
            dict(method='dagger',stage=3,task='fwd',kind='isolated',tracking_fraction=.1)]
    out = retention(rows)
    assert all(r['retained_ratio'] is None for r in out)
    assert out[-1]['loss'] == -.1


def test_cpg_bounds_and_phase():
    theta = (CPG_LO+CPG_HI)/2
    theta[1:7] = 2
    for t in range(100):
        a = gait(theta,t)
        assert a.shape == (6,) and np.abs(a).max() <= 1
    theta[0] = 2
    np.testing.assert_allclose(gait(theta,0),gait(theta,10),atol=1e-10)


def test_checkpoint_roundtrip_rng_replay():
    rng = np.random.default_rng(3)
    replay = Replay(2)
    replay.add(np.ones((1,20)), np.ones((1,6)), [0])
    root = Path(__file__).resolve().parents[1]/'results/r5'
    root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='numpy_test_', dir=root) as directory:
        path = Path(directory)/'r5.pkl'
        atomic_pickle(path,dict(weights=init_params(0,8), replay=replay, rng=rng.bit_generator.state, step=4))
        a = rng.normal(size=10)
        st = read_pickle(path)
        rng.bit_generator.state = st['rng']
        np.testing.assert_array_equal(a,rng.normal(size=10))
        assert st['step']==4 and st['replay'].count.tolist()==[1,0,0]
        assert not Path(str(path)+'.tmp').exists()


def test_deadline_validation_and_expiry():
    with pytest.raises(ValueError):
        Deadline(46)
    d = Deadline(.01)
    d.end = time.monotonic()-1
    with pytest.raises(TimeoutError):
        d.check()


def test_numpy_suite_has_not_imported_mlx():
    assert not any(k == 'mlx' or k.startswith('mlx.') for k in sys.modules)
    for filename in ('r5_core.py','r5.py','r5_train.py','r5_report.py'):
        ast.parse((Path(__file__).resolve().parents[1]/filename).read_text())


def test_dense_reward_gives_progress_signal_when_reversing():
    action = np.zeros(6)
    assert training_reward(1,2.9,action,.2)-training_reward(1,3.,action,.2) > .019
    assert reward(1,2.9,action)-reward(1,3.,action) < 1e-6
    for tid,target in enumerate(TARGETS):
        assert training_reward(tid,target,action,.2)==reward(tid,target,action)
        assert training_reward(tid,target+.5,action,.2) < training_reward(tid,target,action,.2)
        assert training_reward(tid,target-3,action,0.)==reward(tid,target-3,action)


def test_annealing_and_zero_variance_advantages():
    assert scheduled_lr(3e-4,0,10,.1)==pytest.approx(3e-4)
    assert scheduled_lr(3e-4,9,10,.1)==pytest.approx(3e-5)
    assert scheduled_lr(3e-4,99,10,.1)==pytest.approx(3e-5)
    assert scheduled_lr(3e-4,9,10)==3e-4
    assert np.isfinite(normalize_advantages(np.zeros(32))).all()
    np.testing.assert_array_equal(normalize_advantages(np.ones(32)),np.zeros(32))


def test_exploration_reset_does_not_change_deterministic_policy():
    p = init_params(4,12)
    p['ls'][:] = -2.
    q = reset_exploration(p,-.5)
    assert np.all(q['ls']==-.5)
    assert all(np.array_equal(p[k],q[k]) for k in p if k!='ls')
    a = add_adapter(p,1,2,5)
    b = reset_exploration(a,-.5,adapter_tid=1)
    assert all(np.array_equal(a[k],b[k]) for k in a if k!='d1_ls')
    np.testing.assert_allclose(logstd(b,np.array([0,1])),[[-2.]*6,[-.5]*6])


def test_selected_deployment_keeps_final_resume_state():
    final, chosen = {'a':np.ones(3)}, {'a':np.zeros(3)}
    state = dict(params=final, selection=dict(params=chosen,step=2),step=4)
    assert deployment_params(state) is chosen
    assert state['params'] is final and state['step']==4
    assert deployment_params(dict(params=final)) is final


def test_weak_acquisition_is_unqualified_in_turn2():
    rows = [dict(method='adapter',stage=1,task='fwd',kind='isolated',tracking_fraction=.4),
            dict(method='adapter',stage=3,task='fwd',kind='isolated',tracking_fraction=.4)]
    result = retention(rows,.7)
    assert all(not r['qualified'] and r['retained_ratio'] is None for r in result)
    assert all(r['loss']==0 for r in result)  # preservation alone is not acquisition


def test_turn2_budget_and_unseen_headline_bank():
    from r5 import PROFILES
    z = PROFILES['t2']
    b = budget(z['base'],z['teacher'],z['student'])
    assert b['final']==15_728_640 and b['addition']==6_291_456
    assert z['base']+2*(z['teacher']+z['student']+2*b['addition'])==36_700_160
    assert min(z['base'],z['teacher'],z['student'],b['addition']//3)>=2_000_000
    for profile in PROFILES.values():
        assert all(profile[k]%(profile['envs']*profile['rollout'])==0 for k in ('base','teacher','student'))
    assert z['test_split']=='test_t2'
    assert PROFILES['tiny_t2']['test_split']=='smoke'
    assert PROFILES['battery']['base']==262144  # turn-1 default intact
