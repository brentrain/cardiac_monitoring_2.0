"""The simulator should give the same run for the same seed, so results can be repeated."""
import random

from icu_telemetry_simulator import ICUTelemetrySimulator


def run(seed, steps=300):
    random.seed(seed)
    sim = ICUTelemetrySimulator(num_patients=8)
    trace = []
    for _ in range(steps):
        sim.update_step()
        trace.append(tuple(round(p.vitals.heart_rate, 3) for p in sim.patients))
    return trace


def test_same_seed_gives_same_run():
    assert run(42) == run(42)


def test_different_seeds_give_different_runs():
    assert run(42) != run(7)


def test_vitals_stay_in_physical_range():
    random.seed(42)
    sim = ICUTelemetrySimulator(num_patients=8)
    for _ in range(300):
        sim.update_step()
        for p in sim.patients:
            assert 0 <= p.vitals.heart_rate <= 200
            assert 50 <= p.vitals.o2_saturation <= 100
