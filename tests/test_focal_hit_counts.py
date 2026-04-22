from __future__ import annotations

import numpy as np
import pytest
import xarray as xa

from openlifu import Point, Pulse, Sequence, Solution, Transducer
from openlifu.plan.hit_count_optimizer import optimize_hit_counts
from openlifu.plan.param_constraint import ParameterConstraint
from openlifu.xdc.element import Element

_REF_IMPEDANCE = 1000.0 * 1500.0  # density * sound_speed, matching SolutionAnalysisOptions defaults

def _make_intensity_dataset(n_foci: int, isppa_per_focus: list[float]) -> xa.Dataset:
    nx, ny, nz = 3, 2, 3
    intensity_data = np.zeros((n_foci, nx, ny, nz))
    for i, isppa in enumerate(isppa_per_focus):
        intensity_data[i] = isppa
    # intensity = 1e-4 * p² / (2·Z)
    pressure_data = np.sqrt(2 * _REF_IMPEDANCE * intensity_data * 1e4)
    coords = {
        'x': xa.DataArray(dims=["x"], data=np.linspace(0, 1, nx), attrs={'units': "m", 'long_name': "Lateral"}),
        'y': xa.DataArray(dims=["y"], data=np.linspace(0, 1, ny), attrs={'units': "m", 'long_name': "Elevation"}),
        'z': xa.DataArray(dims=["z"], data=np.linspace(0, 1, nz), attrs={'units': "m", 'long_name': "Axial"}),
        'focal_point_index': list(range(n_foci)),
    }
    dims = ["focal_point_index", "x", "y", "z"]
    return xa.Dataset({
        'intensity': xa.DataArray(data=intensity_data, dims=dims, attrs={'units': "W/cm^2", 'long_name': "Intensity"}),
        'p_min': xa.DataArray(data=pressure_data, dims=dims, attrs={'units': "Pa", 'long_name': "PNP"}),
        'p_max': xa.DataArray(data=pressure_data, dims=dims, attrs={'units': "Pa", 'long_name': "PPP"}),
    }, coords=coords)


def _make_solution(
    n_foci: int,
    pulse_count: int,
    isppa_per_focus: list[float],
    focal_hit_counts: list[int] | None = None,
    apodizations: np.ndarray | None = None,
) -> Solution:
    rng = np.random.default_rng(0)
    transducer = Transducer(
        elements=[Element(index=i + 1, position=[i * 5, 0, 0], units="mm") for i in range(4)],
        frequency=400e3,
        units="mm",
    )
    if apodizations is None:
        apodizations = np.ones((n_foci, 4))
    return Solution(
        transducer=transducer,
        foci=[Point(id=f"focus_{i}") for i in range(n_foci)],
        delays=rng.random((n_foci, 4)),
        apodizations=apodizations,
        pulse=Pulse(frequency=500e3, duration=20e-6),
        sequence=Sequence(
            pulse_count=pulse_count,
            pulse_interval=0.1,
            pulse_train_interval=0.1 * pulse_count + 5.0,
        ),
        focal_hit_counts=focal_hit_counts if focal_hit_counts is not None else [],
        simulation_result=_make_intensity_dataset(n_foci, isppa_per_focus),
    )


# ---- Tests for Solution.focal_hit_counts validation ----

def test_hit_counts_wrong_length_raises():
    with pytest.raises(ValueError, match="Focal hit counts length"):
        _make_solution(n_foci=3, pulse_count=6, isppa_per_focus=[10, 8, 6], focal_hit_counts=[2, 4])

def test_hit_counts_wrong_sum_raises():
    with pytest.raises(ValueError, match="Focal hit counts sum"):
        _make_solution(n_foci=3, pulse_count=6, isppa_per_focus=[10, 8, 6], focal_hit_counts=[1, 1, 1])


# ---- Tests for Solution.get_ita ----

def test_ita_is_hit_count_weighted_average():
    isppa = [10.0, 6.0]
    counts = [3, 1]
    pulse_count = 4
    sol = _make_solution(n_foci=2, pulse_count=pulse_count, isppa_per_focus=isppa, focal_hit_counts=counts)
    ita = sol.get_ita(intensity=sol.simulation_result['intensity'], units="W/cm^2")
    expected = (
        sum(isppa[i] * counts[i] / pulse_count for i in range(2))
        * sol.get_pulsetrain_dutycycle()
        * sol.get_sequence_dutycycle()
    )
    np.testing.assert_allclose(float(ita.mean()), expected, rtol=1e-10)


# ---- Tests for optimize_hit_counts ----

def test_optimizer_strictly_improves_min_ispta():
    isppa = [10.0, 5.0]
    pulse_count = 6
    opt_counts = optimize_hit_counts(per_focus_isppa=isppa, per_focus_tic=[0.5, 0.5], pulse_count=pulse_count)

    def min_ispta(counts):
        return min(isppa[i] * counts[i] / pulse_count for i in range(len(isppa)))

    assert min_ispta(opt_counts) > min_ispta([3, 3])

def test_optimizer_output_sums_to_pulse_count():
    counts = optimize_hit_counts(per_focus_isppa=[10.0, 5.0, 8.0], per_focus_tic=[0.5, 0.5, 0.5], pulse_count=12)
    assert sum(counts) == 12

def test_optimizer_respects_tic_constraint():
    isppa = [10.0, 5.0]
    tic = [2.0, 0.5]
    pulse_count = 6
    max_tic = 1.0
    counts = optimize_hit_counts(
        per_focus_isppa=isppa,
        per_focus_tic=tic,
        pulse_count=pulse_count,
        param_constraints={"TIC": ParameterConstraint(operator="<", error_value=max_tic)},
    )
    assert sum(tic[i] * counts[i] for i in range(len(tic))) / pulse_count <= max_tic


# ---- Tests for Solution.analyze ----

def test_analyze_tic_is_hit_count_weighted_average():
    counts = [3, 1]
    sol = _make_solution(
        n_foci=2, pulse_count=4, isppa_per_focus=[10.0, 5.0],
        focal_hit_counts=counts,
        apodizations=np.array([[1.0, 1.0, 1.0, 1.0], [0.2, 0.2, 0.2, 0.2]]),
    )
    analysis = sol.analyze()
    assert np.isclose(analysis.TIC, float(np.average(analysis.per_focus_tic, weights=counts)), rtol=1e-10)


# ---- Tests for Solution JSON serialization ----

def test_json_roundtrip_preserves_hit_counts():
    sol = _make_solution(n_foci=3, pulse_count=6, isppa_per_focus=[10, 8, 6], focal_hit_counts=[1, 2, 3])
    sol2 = Solution.from_json(sol.to_json(include_simulation_data=False, compact=False), simulation_result=sol.simulation_result)
    assert sol2.focal_hit_counts == [1, 2, 3]

# ---- Property tests: optimizer is at least as good as any random valid distribution ----

def _min_ispta(isppa: list[float], counts: list[int], pulse_count: int) -> float:
    return min(isp * c / pulse_count for isp, c in zip(isppa, counts))

def _random_valid_hit_counts(rng, n_foci: int, pulse_count: int, min_hits: int = 1, n_samples: int = 300) -> list[list[int]]:
    """Generate diverse valid hit count distributions via random walk from equal distribution."""
    base = [pulse_count // n_foci] * n_foci
    for i in range(pulse_count % n_foci):
        base[i] += 1
    current = base[:]
    samples = []
    for step in range(n_samples * 20):
        src, dst = rng.integers(0, n_foci, size=2)
        if src != dst and current[src] > min_hits:
            current[src] -= 1
            current[dst] += 1
        if step % 20 == 0:
            samples.append(current[:])
    return samples[:n_samples]


# Each tuple: (isppa, tic, pulse_count, max_tic)
# Covers 2-, 3-, 5-, and 7-focus scenarios (matching SinglePoint, 2-spoke, 4-spoke, 6-spoke Wheel),
# with equal ISPPAs, skewed ISPPAs, and binding / non-binding TIC constraints.
_OPTIMIZER_SCENARIOS = [
    # 2 foci
    ([10.0, 5.0],                       [0.5, 0.5],          6,  None),
    ([5.0,  5.0],                       [0.5, 0.5],          6,  None),   # equal ISPPAs
    ([20.0, 1.0],                       [0.5, 0.5],          10, None),   # strongly skewed
    ([10.0, 5.0],                       [2.0, 0.5],          6,  1.0),    # binding TIC
    # 3 foci
    ([10.0, 5.0, 8.0],                  [0.5, 0.5, 0.5],     12, None),
    ([10.0, 5.0, 8.0],                  [1.5, 0.5, 1.0],     12, 0.8),   # binding TIC
    # 5 foci (Wheel: 4 spokes + center)
    ([20.0, 5.0, 5.0, 5.0, 5.0],       [1.0, 0.3, 0.3, 0.3, 0.3], 10, None),
    ([10.0, 8.0, 6.0, 4.0, 2.0],       [0.5] * 5,           20, None),
    ([5.0]  * 5,                        [0.5] * 5,           10, None),   # all equal
    ([10.0, 5.0, 5.0, 5.0, 5.0],       [1.5, 0.4, 0.4, 0.4, 0.4], 10, 0.7),  # binding TIC
    # 7 foci (Wheel: 6 spokes + center)
    ([15.0] + [5.0] * 6,               [0.8] + [0.3] * 6,   14, None),
    ([10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0], [0.5] * 7,      21, None),
]

@pytest.mark.parametrize(("isppa", "tic", "pulse_count", "max_tic"), _OPTIMIZER_SCENARIOS)
def test_optimizer_beats_random_distributions(isppa, tic, pulse_count, max_tic):
    n_foci = len(isppa)
    rng = np.random.default_rng(42)

    constraints = {"TIC": ParameterConstraint(operator="<", error_value=max_tic)} if max_tic else None
    opt_counts = optimize_hit_counts(
        per_focus_isppa=isppa,
        per_focus_tic=tic,
        pulse_count=pulse_count,
        param_constraints=constraints,
    )

    assert sum(opt_counts) == pulse_count
    assert all(c >= 1 for c in opt_counts)

    if max_tic is not None:
        opt_tic = sum(t * c for t, c in zip(tic, opt_counts)) / pulse_count
        assert opt_tic <= max_tic + 1e-9, f"TIC constraint violated: {opt_tic:.4f} > {max_tic}"

    opt_ispta = _min_ispta(isppa, opt_counts, pulse_count)
    random_distributions = _random_valid_hit_counts(rng, n_foci, pulse_count)

    # When there is a TIC constraint, only compare against feasible random distributions
    if max_tic is not None:
        random_distributions = [
            c for c in random_distributions
            if sum(t * ci for t, ci in zip(tic, c)) / pulse_count <= max_tic
        ]

    failures = [
        (counts, _min_ispta(isppa, counts, pulse_count))
        for counts in random_distributions
        if _min_ispta(isppa, counts, pulse_count) > opt_ispta + 1e-9
    ]
    assert not failures, (
        f"Optimizer (min ISPTA={opt_ispta:.6f}, counts={opt_counts}) was beaten by "
        f"{len(failures)} random distribution(s). Worst: counts={failures[0][0]}, "
        f"min ISPTA={failures[0][1]:.6f}."
    )
