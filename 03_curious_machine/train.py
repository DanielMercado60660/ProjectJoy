"""Train a reward-free explorer: PPO (MLX) driven only by an intrinsic bonus.

  --agent random    uniform random actions (baseline, no learning)
  --agent count     PPO + 1/sqrt(N) bonus on a hash of the egocentric observation
  --agent rnd       PPO + Random Network Distillation (Burda et al. 2018)
  --agent disagree  PPO + forward-model ensemble disagreement (Pathak et al. 2019)

There is no extrinsic reward. Following Burda et al. the policy still has two value heads
(intrinsic: non-episodic, gamma 0.99; extrinsic: episodic, gamma 0.999); the extrinsic stream is
identically zero here and kept only so the architecture matches the paper.

Example:
  ../bin/gpu_run ../.venv/bin/python train.py --agent rnd --seed 0 --total_steps 10000000 --out runs/rnd_s0
"""
import argparse
import json
import os
import pickle
import time

import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_unflatten

from env import VecRooms, OBS_DIM, N_ACTIONS, EP_FIELDS
from agents import PolicyNet, RND, Disagreement, HashCounter, RunningMeanStd, NGUAux, EpisodicKNN

DEFAULTS = dict(agent="rnd", seed=0, total_steps=10_000_000, n_envs=64, rollout=128,
                max_steps=512, noisy_tv=False, lr=2.5e-4, epochs=4, minibatches=4, clip=0.1,
                ent_coef=0.001, gamma_int=0.99, gamma_ext=0.999, lam=0.95, int_coef=1.0,
                ext_coef=0.0, aux_frac=0.25, ens_k=5, hidden=256, ckpt_every=50, device="gpu")


def to_np_tree(tree):
    return {k: np.array(v) for k, v in tree_flatten(tree)}


def from_np_tree(d):
    return tree_unflatten([(k, mx.array(v)) for k, v in d.items()])


class Explorer:
    """Everything that is learned or accumulated during a run (what a checkpoint must hold)."""

    def __init__(self, cfg):
        self.cfg = cfg
        mx.random.seed(cfg["seed"])
        self.rng = np.random.default_rng(cfg["seed"])
        self.policy = PolicyNet(OBS_DIM, N_ACTIONS, cfg["hidden"])
        self.opt = optim.Adam(learning_rate=cfg["lr"])
        self.aux, self.aux_opt, self.counter = None, None, None
        if cfg["agent"] in ("rnd", "rnd_ep"):
            self.aux = RND(OBS_DIM)
        elif cfg["agent"] == "ngu":
            self.aux = NGUAux(OBS_DIM, N_ACTIONS)
            self.knn = EpisodicKNN(cfg["n_envs"], cfg["max_steps"] + 1)
            self.err_rms = RunningMeanStd(())
        elif cfg["agent"] == "disagree":
            self.aux = Disagreement(OBS_DIM, N_ACTIONS, k=cfg["ens_k"])
        elif cfg["agent"] == "count":
            self.counter = HashCounter(OBS_DIM, seed=cfg["seed"])
        if cfg["agent"] in ("rnd_ep", "ep"):
            # NGU-lite: per-episode visit counts of observation hashes (the agent's episodic memory)
            self.hasher = HashCounter(OBS_DIM, table_bits=1, seed=cfg["seed"])
            self.ep_counts = [dict() for _ in range(cfg["n_envs"])]
            self.err_rms = RunningMeanStd(())
        if self.aux is not None:
            self.aux_opt = optim.Adam(learning_rate=cfg["lr"])
        mx.eval(self.policy.parameters())
        if self.aux is not None:
            mx.eval(self.aux.parameters())
        self.obs_rms = RunningMeanStd((OBS_DIM,))
        self.ret_rms = RunningMeanStd(())
        self.rewems = np.zeros(cfg["n_envs"])
        self.update, self.steps = 0, 0

    # ---- acting
    def act(self, obs, greedy=False):
        logits, vi, ve = self.policy(mx.array(obs))
        mx.eval(logits, vi, ve)
        lg = np.array(logits, np.float64)
        lg -= lg.max(1, keepdims=True)
        logp_all = lg - np.log(np.exp(lg).sum(1, keepdims=True))
        if greedy:
            a = lg.argmax(1)
        else:
            a = (lg + self.rng.gumbel(size=lg.shape)).argmax(1)
        return a, logp_all[np.arange(len(a)), a], np.array(vi), np.array(ve)

    # ---- checkpoint
    def save(self, path, env, extra=None):
        d = dict(cfg=self.cfg, update=self.update, steps=self.steps,
                 policy=to_np_tree(self.policy.parameters()), opt=to_np_tree(self.opt.state),
                 obs_rms=self.obs_rms.state(), ret_rms=self.ret_rms.state(), rewems=self.rewems,
                 rng=self.rng.bit_generator.state, env=env.get_state() if env is not None else None,
                 extra=extra or {})
        if self.aux is not None:
            d["aux"] = to_np_tree(self.aux.parameters())
            d["aux_opt"] = to_np_tree(self.aux_opt.state)
        if self.counter is not None:
            d["count_table"] = self.counter.table
        if self.cfg["agent"] == "ngu":
            d["knn"], d["err_rms"] = self.knn.state(), self.err_rms.state()
        if self.cfg["agent"] in ("rnd_ep", "ep"):
            d["ep_counts"], d["err_rms"] = self.ep_counts, self.err_rms.state()
        tmp = path + ".tmp"
        with open(tmp, "wb") as f:
            pickle.dump(d, f)
        os.replace(tmp, path)

    @staticmethod
    def load(path, env=None, cfg_override=None):
        with open(path, "rb") as f:
            d = pickle.load(f)
        cfg = dict(d["cfg"])
        if cfg_override:
            cfg.update(cfg_override)
        ex = Explorer(cfg)
        ex.policy.update(from_np_tree(d["policy"]))
        ex.opt.state = from_np_tree(d["opt"])
        if ex.aux is not None:
            ex.aux.update(from_np_tree(d["aux"]))
            ex.aux_opt.state = from_np_tree(d["aux_opt"])
        if ex.counter is not None:
            ex.counter.table[:] = d["count_table"]
        if cfg["agent"] == "ngu":
            ex.knn.load(d["knn"])
            ex.err_rms.load(d["err_rms"])
        if cfg["agent"] in ("rnd_ep", "ep"):
            ex.ep_counts = d["ep_counts"]
            ex.err_rms.load(d["err_rms"])
        ex.obs_rms.load(d["obs_rms"])
        ex.ret_rms.load(d["ret_rms"])
        ex.rewems = d["rewems"]
        ex.rng.bit_generator.state = d["rng"]
        ex.update, ex.steps = d["update"], d["steps"]
        if env is not None and d["env"] is not None:
            env.set_state(d["env"])
        return ex, d


def ppo_loss(model, x, a, oldlogp, adv, ret_i, ret_e, clip, ent_coef):
    logits, vi, ve = model(x)
    logp_all = logits - mx.logsumexp(logits, axis=1, keepdims=True)
    logp = mx.take_along_axis(logp_all, a[:, None], axis=1)[:, 0]
    ratio = mx.exp(logp - oldlogp)
    pg = -mx.mean(mx.minimum(ratio * adv, mx.clip(ratio, 1 - clip, 1 + clip) * adv))
    vloss = 0.5 * mx.mean((vi - ret_i) ** 2) + 0.5 * mx.mean((ve - ret_e) ** 2)
    ent = -mx.mean(mx.sum(mx.exp(logp_all) * logp_all, axis=1))
    return pg + vloss - ent_coef * ent, (pg, vloss, ent)


def gae(rew, val, last_val, dones, gamma, lam):
    """rew/val/dones [T,N]; dones[t]=episode ended after step t (None => non-episodic)."""
    T = rew.shape[0]
    adv = np.zeros_like(rew)
    last = 0.0
    for t in reversed(range(T)):
        nv = last_val if t == T - 1 else val[t + 1]
        nonterm = 1.0 if dones is None else 1.0 - dones[t]
        delta = rew[t] + gamma * nv * nonterm - val[t]
        last = delta + gamma * lam * nonterm * last
        adv[t] = last
    return adv, adv + val


def intrinsic_rewards(ex, XOg, Ag, XNg):
    """Raw intrinsic reward for each transition (flattened [T*N]); inputs are normalised mx arrays."""
    agent = ex.cfg["agent"]
    out = []
    for i in range(0, XNg.shape[0], 8192):
        sl = slice(i, i + 8192)
        if agent in ("rnd", "rnd_ep", "ngu"):
            out.append(ex.aux.errors(XNg[sl]))
        else:
            out.append(ex.aux.errors(XOg[sl], Ag[sl], None))
    r = mx.concatenate(out)
    mx.eval(r)
    return np.array(r)


def norm_fn(rms):
    mean = mx.array(rms.mean.astype(np.float32))
    istd = mx.array((1.0 / np.sqrt(rms.var + 1e-8)).astype(np.float32))
    return lambda x: mx.clip((x - mean) * istd, -5.0, 5.0)


def train(cfg, resume=None, log_every=1, stop_after_updates=None):
    os.makedirs(cfg["out"], exist_ok=True)
    if cfg["device"] == "cpu":
        mx.set_default_device(mx.cpu)
    env = VecRooms(cfg["n_envs"], cfg["max_steps"], cfg["noisy_tv"], seed=cfg["seed"])
    ckpt_path = os.path.join(cfg["out"], "ckpt.pkl")
    log_path = os.path.join(cfg["out"], "log.jsonl")
    ep_path = os.path.join(cfg["out"], "episodes.jsonl")
    if resume:
        ex, _ = Explorer.load(resume, env)
        print(f"resumed from {resume} at update {ex.update}, steps {ex.steps}", flush=True)
    else:
        ex = Explorer(cfg)
        with open(os.path.join(cfg["out"], "config.json"), "w") as f:
            json.dump(cfg, f, indent=1)
        for p in (log_path, ep_path):
            open(p, "w").close()
    obs = env.obs.copy()
    N, T = cfg["n_envs"], cfg["rollout"]
    learn = cfg["agent"] != "random"
    if learn and ex.update == 0 and cfg["agent"] in ("rnd", "rnd_ep", "ngu", "disagree"):
        # initialise observation normalisation with a random-action rollout (Burda et al.)
        tmp = VecRooms(N, cfg["max_steps"], cfg["noisy_tv"], seed=cfg["seed"] + 999)
        for _ in range(T):
            o, _, _ = tmp.step(ex.rng.integers(0, N_ACTIONS, N))
            ex.obs_rms.update(o.astype(np.float64))
    loss_and_grad = nn.value_and_grad(ex.policy, ppo_loss)
    if ex.aux is not None:
        if cfg["agent"] in ("rnd", "rnd_ep"):
            aux_lg = nn.value_and_grad(ex.aux, lambda m, x, a, xn, mk: m.loss(x, mk))
        elif cfg["agent"] == "ngu":
            aux_lg = nn.value_and_grad(ex.aux, lambda m, x, a, xn, mk: m.loss(x, xn, a, mk[0], mk[1]))
        else:
            aux_lg = nn.value_and_grad(ex.aux, lambda m, x, a, xn, mk: m.loss(x, a, xn, mk))
    recent = []
    t0, s0 = time.time(), ex.steps
    n_updates = cfg["total_steps"] // (N * T)
    while ex.update < n_updates:
        ob_b = np.zeros((T, N, OBS_DIM), np.float32)
        nx_b = np.zeros((T, N, OBS_DIM), np.float32)
        a_b = np.zeros((T, N), np.int64)
        lp_b = np.zeros((T, N)); vi_b = np.zeros((T, N)); ve_b = np.zeros((T, N))
        d_b = np.zeros((T, N)); ri_b = np.zeros((T, N)); ep_b = np.zeros((T, N))
        new_eps = []
        nrm_roll = norm_fn(ex.obs_rms) if cfg["agent"] == "ngu" else None
        for t in range(T):
            if learn:
                a, lp, vi, ve = ex.act(obs)
            else:
                a = ex.rng.integers(0, N_ACTIONS, N); lp = vi = ve = 0.0
            nobs, done, eps = env.step(a)
            ob_b[t], nx_b[t], a_b[t], lp_b[t], vi_b[t], ve_b[t], d_b[t] = obs, nobs, a, lp, vi, ve, done
            if ex.counter is not None:
                ri_b[t] = ex.counter.bonus_and_update(nobs)
            if cfg["agent"] == "ngu":
                e = ex.aux.embed(nrm_roll(mx.array(nobs)))
                mx.eval(e)
                ep_b[t] = ex.knn.reward_and_add(np.array(e), done)
            if cfg["agent"] in ("rnd_ep", "ep"):
                hs = ex.hasher.hash64(nobs)
                for i in range(N):
                    if done[i]:
                        ex.ep_counts[i] = {}
                    c = ex.ep_counts[i].get(int(hs[i]), 0) + 1
                    ex.ep_counts[i][int(hs[i])] = c
                    ep_b[t, i] = 1.0 / np.sqrt(c)
            for e in eps:
                e["step"] = ex.steps + (t + 1) * N
            new_eps += eps
            obs = nobs
        ex.steps += N * T
        ex.update += 1
        recent = (recent + new_eps)[-2 * N:]
        if new_eps:
            with open(ep_path, "a") as f:
                for e in new_eps:
                    f.write(json.dumps(e) + "\n")
        stats = {}
        if learn:
            B = N * T
            Xg = mx.array(ob_b.reshape(B, OBS_DIM))
            XNg = mx.array(nx_b.reshape(B, OBS_DIM))
            Ag = mx.array(a_b.reshape(B))
            if ex.aux is not None:
                nrm = norm_fn(ex.obs_rms)
                XOg, XNg_n = nrm(Xg), nrm(XNg)
                ri_b = intrinsic_rewards(ex, XOg, Ag, XNg_n).reshape(T, N)
            if cfg["agent"] in ("rnd_ep", "ngu"):
                # NGU-style: episodic novelty, modulated by lifelong (RND) novelty in [1, 5]
                ex.err_rms.update(ri_b.reshape(-1))
                alpha = 1.0 + (ri_b - ex.err_rms.mean) / np.sqrt(ex.err_rms.var + 1e-12)
                ri_b = ep_b * np.clip(alpha, 1.0, 5.0)
            elif cfg["agent"] == "ep":
                ri_b = ep_b  # ablation: episodic novelty only, no lifelong novelty
            raw_mean = float(ri_b.mean())
            # normalise by running std of the (non-episodic) discounted intrinsic return
            rets = np.zeros((T, N))
            for t in range(T):
                ex.rewems = ex.rewems * cfg["gamma_int"] + ri_b[t]
                rets[t] = ex.rewems
            ex.ret_rms.update(rets.reshape(-1))
            ri_n = ri_b / np.sqrt(ex.ret_rms.var + 1e-8)
            _, _, last_vi, last_ve = ex.act(obs)
            adv_i, ret_i = gae(ri_n, vi_b, last_vi, None, cfg["gamma_int"], cfg["lam"])
            adv_e, ret_e = gae(np.zeros((T, N)), ve_b, last_ve, d_b, cfg["gamma_ext"], cfg["lam"])
            adv = cfg["int_coef"] * adv_i + cfg["ext_coef"] * adv_e
            if ex.aux is not None:
                # obs-normalisation stats: batch moments in float32, merged in float64
                bm = mx.mean(XNg, axis=0); bv = mx.var(XNg, axis=0)
                mx.eval(bm, bv)
                ex.obs_rms.update_moments(np.array(bm), np.array(bv), B)
            LPg = mx.array(lp_b.reshape(B).astype(np.float32))
            ADV = adv.reshape(B).astype(np.float32)
            RIg = mx.array(ret_i.reshape(B).astype(np.float32))
            REg = mx.array(ret_e.reshape(B).astype(np.float32))
            if ex.aux is not None:
                NOTDONE = 1.0 - d_b.reshape(B).astype(np.float32)
            mb = B // cfg["minibatches"]
            pls, vls, ents, als = [], [], [], []
            for _ in range(cfg["epochs"]):
                perm = ex.rng.permutation(B)
                for k in range(cfg["minibatches"]):
                    idx = perm[k * mb:(k + 1) * mb]
                    ig = mx.array(idx)
                    adv_mb = ADV[idx]
                    adv_mb = (adv_mb - adv_mb.mean()) / (adv_mb.std() + 1e-8)
                    (loss, (pg, vl, en)), g = loss_and_grad(
                        ex.policy, Xg[ig], Ag[ig], LPg[ig], mx.array(adv_mb), RIg[ig], REg[ig],
                        cfg["clip"], cfg["ent_coef"])
                    g, _ = optim.clip_grad_norm(g, 0.5)
                    ex.opt.update(ex.policy, g)
                    if ex.aux is not None:
                        if cfg["agent"] in ("rnd", "rnd_ep"):
                            mk = (ex.rng.random(mb) < cfg["aux_frac"]).astype(np.float32)
                            al, ag = aux_lg(ex.aux, XNg_n[ig], None, None, mx.array(mk))
                        elif cfg["agent"] == "ngu":
                            mk = (ex.rng.random(mb) < cfg["aux_frac"]).astype(np.float32)
                            al, ag = aux_lg(ex.aux, XOg[ig], Ag[ig], XNg_n[ig],
                                            (mx.array(mk), mx.array(NOTDONE[idx])))
                        else:
                            mk = (ex.rng.random((cfg["ens_k"], mb)) < 0.5).astype(np.float32) * NOTDONE[idx][None]
                            al, ag = aux_lg(ex.aux, XOg[ig], Ag[ig], XNg_n[ig], mx.array(mk))
                        ex.aux_opt.update(ex.aux, ag)
                        mx.eval(ex.aux.parameters(), ex.aux_opt.state)
                        als.append(al)
                    mx.eval(ex.policy.parameters(), ex.opt.state)
                    pls.append(pg); vls.append(vl); ents.append(en)
            f = lambda z: float(np.mean([float(v) for v in z])) if z else None
            pls, vls, ents, als = f(pls), f(vls), f(ents), f(als)
            stats = dict(adv_int_std=float(adv_i.std()), adv_ext_std=float(adv_e.std()), r_int_raw=raw_mean, r_int_norm=float(ri_n.mean()), pg=pls, vloss=vls,
                         entropy=ents, aux_loss=als)
            if ex.counter is not None:
                stats["distinct_hashes"] = ex.counter.n_distinct()
        if ex.update % log_every == 0:
            rec = dict(update=ex.update, steps=ex.steps, sps=(ex.steps - s0) / (time.time() - t0),
                       **stats)
            if recent:
                for f_ in EP_FIELDS:
                    rec["ep_" + f_] = float(np.mean([e[f_] for e in recent]))
            with open(log_path, "a") as f:
                f.write(json.dumps(rec) + "\n")
            if ex.update % (10 * log_every) == 0 or ex.update == 1:
                print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in rec.items()}),
                      flush=True)
        if learn and (ex.update % cfg["ckpt_every"] == 0 or ex.update == n_updates):
            ex.save(ckpt_path, env)
        if stop_after_updates is not None and ex.update >= stop_after_updates:
            break
    return ex, env


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    for k, v in DEFAULTS.items():
        if isinstance(v, bool):
            p.add_argument("--" + k, action="store_true", default=v)
        else:
            p.add_argument("--" + k, type=type(v), default=v)
    p.add_argument("--out", required=True)
    p.add_argument("--resume", default=None)
    p.add_argument("--stop_after_updates", type=int, default=None, help="for resume tests")
    a = vars(p.parse_args())
    resume, stop = a.pop("resume"), a.pop("stop_after_updates")
    train(a, resume=resume, stop_after_updates=stop)
