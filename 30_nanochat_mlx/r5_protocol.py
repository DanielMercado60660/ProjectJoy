"""Single CPU-only wire format and command grammar for SFT and rollout."""
import numpy as np
from r5_world import ACTIONS, Env, prompt


def prompt_ids(tok, observation):
    return tok.render_for_completion([dict(role="user", content=prompt(observation))])


def action_ids(tok, action):
    if action not in ACTIONS:
        raise ValueError(f"not a grammar command: {action!r}")
    return tok.encode(action) + [tok.encode_special("<|assistant_end|>")]


def unpack_example(tokens, mask):
    """Recover exact prompt/assistant targets from the shifted SFT mask."""
    positions = np.flatnonzero(mask)
    if not len(positions) or not np.array_equal(positions, np.arange(positions[0], positions[-1] + 1)):
        raise ValueError("assistant mask must be nonempty and contiguous")
    return tokens[:positions[0] + 1].tolist(), tokens[positions + 1].tolist()


def verify_episode(row, require_success=False):
    """Re-execute a complete receipt; never trust a success flag alone."""
    env = Env(row["seed"], row["split"], row["mode"])
    invalid = 0
    for step in row["trace"]:
        if env.over() or step["observation"] != env.observe():
            raise ValueError("trace observation does not match replayed environment")
        valid = env.step(step["action"])
        invalid += not valid
        if valid != step["valid"] or env.feedback != step["feedback"]:
            raise ValueError("trace feedback does not match replayed environment")
    if (not env.over() or env.success() != row["success"] or env.steps != row["steps"] or
            invalid != row["invalid"] or env.disruption != row["disruption"] or
            (require_success and not env.success())):
        raise ValueError("terminal receipt does not match replayed environment")
    return True


class ActionGrammar:
    def __init__(self, tok):
        self.children, self.terminal = {}, {}
        for action in ACTIONS:
            ids = tuple(action_ids(tok, action))
            if ids in self.terminal:
                raise ValueError("two commands have identical tokenizations")
            self.terminal[ids] = action
            for k, token in enumerate(ids):
                self.children.setdefault(ids[:k], set()).add(token)
        self.max_tokens = max(map(len, self.terminal))

    def decode(self, scores_fn, rng=None, temperature=0.):
        """scores_fn(prefix, allowed) returns logits on exactly that same prefix."""
        prefix = ()
        for _ in range(self.max_tokens):
            allowed = sorted(self.children[prefix])
            scores = np.asarray(scores_fn(prefix, allowed), dtype=np.float64)
            if scores.shape != (len(allowed),) or not np.isfinite(scores).all():
                raise ValueError("invalid grammar logits")
            if temperature > 0:
                prob = np.exp((scores - scores.max()) / temperature)
                token = int(rng.choice(allowed, p=prob / prob.sum()))
            else:
                token = allowed[int(scores.argmax())]
            prefix += (token,)
            if prefix in self.terminal:
                return self.terminal[prefix]
        raise RuntimeError("incomplete grammar decode")
