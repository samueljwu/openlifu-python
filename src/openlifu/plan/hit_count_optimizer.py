from __future__ import annotations

from typing import Dict, List

import numpy as np
from scipy.optimize import linprog

from openlifu.plan.param_constraint import ParameterConstraint


def optimize_hit_counts(
    per_focus_isppa: List[float],
    per_focus_tic: List[float],
    pulse_count: int,
    param_constraints: Dict[str, ParameterConstraint] | None = None,
    min_hits: int = 1,
) -> List[int]:
    """Return hit counts that maximize the minimum ISPTA across foci, subject to a TIC hard limit.

    Solves a linear program over the continuous relaxation of hit counts, then rounds to integers.
    Without a TIC constraint the solution is closed-form

    Args:
        per_focus_isppa: Peak spatial-average intensity per pulse per focus (W/cm^2).
        per_focus_tic: TIC computed independently for each focus at full duty cycle.
        pulse_count: Total pulse budget (sequence.pulse_count).
        param_constraints: Protocol.param_constraints. TIC error_value is used as the hard limit.
        min_hits: Minimum pulses per focus. Defaults to 1.

    Returns:
        List of integer hit counts, one per focus, summing to pulse_count.
    """

    n = len(per_focus_isppa)
    if param_constraints is None:
        param_constraints = {}

    budget = pulse_count - min_hits * n
    if budget < 0:
        raise ValueError(
            f"pulse_count ({pulse_count}) is too small to give each of {n} foci "
            f"at least {min_hits} hit(s)."
        )

    # Variables: x = [hits[0], ..., hits[n-1], t]  (hits are the extra hits above min_hits)
    # Minimize -t  (equivalent to maximizing t = minimum ISPTA proxy)
    c = np.array([0.0] * n + [-1.0])

    # ISPTA constraints: isppa[i] * (hits[i] + min_hits) / pulse_count >= t
    # Rearranged: -isppa[i]/pulse_count * hits[i] + t <= isppa[i]*min_hits/pulse_count
    A_ub, b_ub = [], []
    for i in range(n):
        row = [0.0] * (n + 1)
        row[i] = -per_focus_isppa[i] / pulse_count
        row[n] = 1.0
        A_ub.append(row)
        b_ub.append(per_focus_isppa[i] * min_hits / pulse_count)

    # TIC constraint: sum(tic[i] * (hits[i] + min_hits)) / pulse_count <= max_tic
    # Rearranged: sum(tic[i] * hits[i]) <= max_tic * pulse_count - sum(tic[i]) * min_hits
    tic_constraint = param_constraints.get("TIC")
    if tic_constraint is not None and tic_constraint.error_value is not None:
        max_tic = float(tic_constraint.error_value)
        row = [*per_focus_tic, 0.0]
        A_ub.append(row)
        b_ub.append(max_tic * pulse_count - sum(per_focus_tic) * min_hits)

    # Budget: sum(hits[i]) == budget
    A_eq = [[1.0] * n + [0.0]]
    b_eq = [float(budget)]

    # Bounds: hits[i] >= 0, t unbounded
    bounds = [(0.0, None)] * n + [(None, None)]

    result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
    if not result.success:
        raise ValueError(f"Hit count optimization failed: {result.message}")

    # Round continuous solution to integers, then do an objective-aware local improvement
    # pass so the final integer solution better matches the true max-min objective.
    hits_continuous = result.x[:n]
    hits = np.floor(hits_continuous).astype(int)
    remainder = int(budget - hits.sum())
    if remainder > 0:
        fractions = hits_continuous - np.floor(hits_continuous)
        adjust = np.argsort(fractions)[::-1][:remainder]
        hits[adjust] += 1

    isppa_arr = np.array(per_focus_isppa, dtype=float)
    tic_arr = np.array(per_focus_tic, dtype=float)

    def objective(candidate: np.ndarray) -> float:
        total_hits = candidate + min_hits
        return float(np.min(isppa_arr * total_hits / pulse_count))

    def tic_feasible(candidate: np.ndarray) -> bool:
        if tic_constraint is None or tic_constraint.error_value is None:
            return True
        total_hits = candidate + min_hits
        return float(np.dot(tic_arr, total_hits) / pulse_count) <= float(tic_constraint.error_value) + 1e-12

    improved = True
    while improved:
        improved = False
        current_obj = objective(hits)
        best_obj = current_obj
        best_move = None

        for src in range(n):
            if hits[src] <= 0:
                continue
            for dst in range(n):
                if src == dst:
                    continue

                candidate = hits.copy()
                candidate[src] -= 1
                candidate[dst] += 1

                if not tic_feasible(candidate):
                    continue

                cand_obj = objective(candidate)
                if cand_obj > best_obj + 1e-12:
                    best_obj = cand_obj
                    best_move = (src, dst)

        if best_move is not None:
            src, dst = best_move
            hits[src] -= 1
            hits[dst] += 1
            improved = True

    if tic_constraint is not None and tic_constraint.error_value is not None:
        tic_arr = np.array(per_focus_tic)
        while np.dot(tic_arr, hits + min_hits) / pulse_count > tic_constraint.error_value:
            src = int(np.argmax(np.where(hits > 0, tic_arr, -np.inf)))
            dst = int(np.argmin(np.where(np.arange(n) != src, tic_arr, np.inf)))
            if hits[src] == 0:
                raise ValueError("Cannot satisfy TIC constraint with given pulse_count and min_hits.")
            hits[src] -= 1
            hits[dst] += 1

    return (hits + min_hits).tolist()
