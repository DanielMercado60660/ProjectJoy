"""Hand-written colonies. Each is a function obs -> (move probs, deposit probs) over exactly
the same local observation vector and action space that the learned policy gets.

classic  : the textbook two-pheromone forager. Searching ants follow "food" pheromone and lay
           "home" pheromone; carrying ants follow "home" pheromone and lay "food" pheromone.
           Deposits stop after DEPOSIT_WINDOW steps since the last nest visit / pick-up.
compass  : classic + the path-integration compass (which the learned policy also sees):
           homing ants blend the home-pheromone gradient with the nest bearing.
random   : uniform random moves and deposits (sanity floor).
"""
import numpy as np
import sim

DEPOSIT_WINDOW = 150
_ANG = (np.arange(5) - 2) * np.pi / 4
DIR_F, DIR_S = np.cos(_ANG).astype(np.float32), np.sin(_ANG).astype(np.float32)


def _softmax(z):
    z = z - z.max(-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(-1, keepdims=True)


def handwritten(obs, use_compass=True, w_p=6.0, w_fwd=1.0, w_c=2.5, eps=0.05):
    B, N, _ = obs.shape
    S = obs[..., :sim.SENS_DIM].reshape(B, N, 2, 5, 5)  # [b,n,dist,turn,channel]
    carry = obs[..., sim.I_CARRY] > 0.5
    energy = obs[..., sim.I_ENERGY]
    clock = obs[..., sim.I_CLOCK] * 100.0
    cf, cs = obs[..., sim.I_CMP_F], obs[..., sim.I_CMP_S]
    ndist = obs[..., sim.I_NDIST] * 32.0
    at_nest = obs[..., sim.I_ATNEST] > 0.5
    # go home when carrying, or when the remaining energy only just covers the way back
    homing = carry | (energy < ndist * sim.C_MOVE * 1.6 + 0.06)
    if not use_compass:  # classic ants don't know the distance either: fixed energy reserve
        homing = carry | (energy < 0.35)

    blocked = S[:, :, 0, :, sim.CH_OBST] > 0.5
    p_home = S[:, :, 0, :, sim.CH_HOME] + 0.5 * S[:, :, 1, :, sim.CH_HOME]
    p_food = S[:, :, 0, :, sim.CH_FOOD] + 0.5 * S[:, :, 1, :, sim.CH_FOOD]
    hm = homing[..., None]
    P = np.where(hm, p_home, p_food)
    # relative gradient: subtract the mean so only differences between directions matter
    P = P - P.mean(-1, keepdims=True)
    score = w_p * P + w_fwd * DIR_F
    if use_compass:
        align = cf[..., None] * DIR_F + cs[..., None] * DIR_S
        score = score + np.where(hm, w_c * align, 0.0)
    see_food = S[:, :, 0, :, sim.CH_FOODITEM] + 0.5 * S[:, :, 1, :, sim.CH_FOODITEM]
    see_nest = S[:, :, 0, :, sim.CH_NEST] + 0.5 * S[:, :, 1, :, sim.CH_NEST]
    score = score + np.where(hm, 8.0 * see_nest, 8.0 * (~carry[..., None]) * see_food)
    score = np.where(blocked, -1e9, score)
    probs5 = _softmax(score)
    free = (~blocked).astype(np.float32)
    nfree = free.sum(-1, keepdims=True)
    probs5 = (1 - eps) * probs5 + eps * free / np.maximum(nfree, 1)

    pm = np.zeros((B, N, sim.N_MOVE), np.float32)
    pm[..., :5] = probs5
    # U-turn: just picked up food, just delivered, or boxed in
    uturn = (carry & (clock < 0.5)) | (nfree[..., 0] == 0)
    just_home = at_nest & ~carry & (clock < 0.5) & (p_food.max(-1) < 0.05)
    pm[uturn] = 0.0
    pm[uturn, 5] = 1.0
    pm[just_home, :5] *= 0.7
    pm[just_home, 5] += 0.3

    pd = np.zeros((B, N, sim.N_DEP), np.float32)
    lay_food = carry & (clock < DEPOSIT_WINDOW)
    lay_home = ~carry & (clock < DEPOSIT_WINDOW)
    pd[..., 0] = 1.0
    pd[lay_food] = (0, 0, 1)
    pd[lay_home] = (0, 1, 0)
    return pm, pd


def classic(obs):
    return handwritten(obs, use_compass=False)


def compass(obs):
    return handwritten(obs, use_compass=True)


def random_policy(obs):
    B, N, _ = obs.shape
    pm = np.full((B, N, sim.N_MOVE), 1.0 / 6, np.float32)
    pm[..., sim.STAY] = 0.0
    pd = np.full((B, N, sim.N_DEP), 1.0 / 3, np.float32)
    return pm, pd


def compass_tuned(obs):
    """Added by the cross-review: compass colony with 3 weights changed (w_fwd 1->2, w_c 2.5->5,
    eps 0.05->0.02), picked by a ~1-minute 60-config grid on the validation seeds 900-915."""
    return handwritten(obs, use_compass=True, w_p=6.0, w_fwd=2.0, w_c=5.0, eps=0.02)


POLICIES = {"classic": classic, "compass": compass, "compass_tuned": compass_tuned, "random": random_policy}
