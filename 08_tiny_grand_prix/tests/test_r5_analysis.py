"""NumPy-only tests for post-run diagnostics, independent of checkpoints."""
import numpy as np
from analyze_r5 import aggregate, event_curve, first_packet_estimates, single_packet_probe
from racing.sim import G, MAX_STEER, V_SCALE, WHEELBASE


def test_failure_cap_is_not_recovery():
    rows = [dict(events=[dict(status='recovered', capped_latency_s=.3),
                         dict(status='unreached', capped_latency_s=5.),
                         dict(status='window_expired', capped_latency_s=5.)])]
    c = event_curve(rows, [.1, .3, 5.])
    np.testing.assert_allclose(c['recovered_fraction'], [0, 1/3, 1/3])
    assert c['planned'] == 3
    assert event_curve([dict(events=[])], [1.])['recovered_fraction'] == []


def test_single_packet_probe_at_rest_and_with_excitation():
    obs = np.zeros((3, 14), np.float32)
    obs[1:, 10] = 20 / V_SCALE
    obs[1:, 11] = .7
    demand = 20 * np.tan(.7 * MAX_STEER) / WHEELBASE
    obs[1, 13] = demand  # unlimited yaw: only a lower bound, not an estimate
    obs[2, 13] = .5 * G / 20
    flagged, mu = single_packet_probe(obs)
    np.testing.assert_array_equal(flagged, [False, False, True])
    assert np.isfinite(mu).all()
    np.testing.assert_allclose(mu[2], .5, atol=1e-7)


def test_dnf_penalty_and_unreached_event_stay_in_denominators():
    rows = [dict(finished=True, crashed=False, lap_penalty_s=30.,
                 events=[dict(status='recovered', exposed=True, capped_latency_s=.2)]),
            dict(finished=False, crashed=True, lap_penalty_s=60.,
                 events=[dict(status='unreached', exposed=False, capped_latency_s=5.)])]
    s = aggregate(rows)
    assert s['mean_finished_lap_s'] == 30 and s['mean_dnf_penalized_lap_s'] == 45
    assert s['event_recovery_coverage'] == .5 and s['adaptation_capped_s'] == 2.6


def test_first_packet_stops_at_next_change_and_counts_unreached():
    samples = np.zeros((10, 1, 5))
    samples[5, 0] = [1, 1, 1, .01, 1]
    rows = [dict(events=[dict(change_step=1), dict(change_step=5), dict(change_step=-1)])]
    s = first_packet_estimates(samples, rows, np.array([True]))
    assert s['planned'] == 3 and s['flagged_within_window'] == 1
    np.testing.assert_allclose(s['flagged_fraction'], [1/3] * 4)
    assert s['first_flag_grip_mae'] == .01
