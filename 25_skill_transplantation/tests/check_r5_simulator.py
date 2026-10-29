"""Optional CPU/MuJoCo integration check, separate from the NumPy-only test suite."""
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from r5_core import make_env, pack_envs, restore_envs, episode


def main():
    rng = np.random.default_rng(9)
    with make_env(48) as a, make_env(48) as b:
        a.reset(seed=918)
        b.reset(seed=23)
        for _ in range(9):
            a.step(rng.uniform(-1,1,6))
        state = pack_envs([a])
        restore_envs([b],state)
        errors = []
        for _ in range(20):
            action = rng.uniform(-1,1,6)
            x,y = a.step(action), b.step(action)
            errors.append(float(np.max(np.abs(x[0]-y[0]))))
            assert x[2:4] == y[2:4]
        assert max(errors) < 1e-10, errors
        assert a._elapsed_steps == b._elapsed_steps == 29
        np.testing.assert_array_equal(a.reset()[0],b.reset()[0])
    received = []
    def policy(obs,tid,t,age):
        received.append((tid,t,age))
        return np.zeros(6)
    ep = episode(policy,7_999_999,(0,2,1),20)
    assert len(ep['trace']['v'])==60
    assert [r['switch'] for r in ep['segments']]==[False,True,True]
    assert received[19:22]==[(0,19,19),(2,20,0),(2,21,1)]
    assert received[40] == (1,40,0)
    assert not any(k=='mlx' or k.startswith('mlx.') for k in sys.modules)
    print(f'R5_CPU_SIM_OK state roundtrip max observation error={max(errors):.3g}; continuous 60-step switch schedule; no MLX imported')


if __name__=='__main__':
    main()
