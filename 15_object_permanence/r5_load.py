"""Runner-only checkpoint dispatch; preserves original turn-1 trainer/source hashes."""
import pickle


def load_pair(path):
    import mlx.core as mx
    with open(path, 'rb') as f:
        state = pickle.load(f)
    if state['config'].get('cue_mode') == 'bound_shared':
        from r5_turn2_train import make_models
    else:
        from r5_train import make_models
    models = make_models(state['config'])
    for name, model in models.items():
        model.load_weights([(k, mx.array(v)) for k, v in state['models'][name].items()])
    mx.eval([m.parameters() for m in models.values()])
    return models, state
