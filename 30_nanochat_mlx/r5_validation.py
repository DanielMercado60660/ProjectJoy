"""Pure NumPy aggregation: cover every validation row, weight by supervised tokens."""
import numpy as np


def validation_batches(n, batch):
    if n <= 0 or batch <= 0:
        raise ValueError("positive validation and batch sizes required")
    for start in range(0, n, batch):
        yield np.arange(start, min(start + batch, n))


def validation_summary(nll_sums, counts, exact, deliveries):
    sums, counts = np.asarray(nll_sums), np.asarray(counts)
    exact, deliveries = np.asarray(exact), np.asarray(deliveries)
    assert len(sums) == len(counts) == len(exact) == len(deliveries) and len(sums)
    assert np.isfinite(sums).all() and (counts > 0).all()
    def summarize(take):
        return dict(examples=int(take.sum()), supervised_units=int(counts[take].sum()),
                    nll=float(sums[take].sum() / counts[take].sum()),
                    teacher_forced_action_exact=float(exact[take].mean()))
    result = summarize(np.ones(len(sums), dtype=bool))
    result["by_deliveries"] = {str(int(n)): summarize(deliveries == n) for n in np.unique(deliveries)}
    return result
