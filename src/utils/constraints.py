"""Monotonicity constraints for the additive integer score.

Rationale
---------
Every domain of the original SOFA score is monotone by construction: points
only increase as PaO2/FiO2 falls, as GCS falls, as platelets fall, as
creatinine and bilirubin rise, and as vasoactive support escalates. A
non-monotone interval weight is therefore a violation of the very structure
this framework claims to preserve.

Mean arterial pressure is deliberately EXEMPT. In the eICU-CRD the variable is
taken from `apacheApsVar`, which stores the value generating the highest
APACHE points during the first 24 ICU hours. The APACHE point function for
mean arterial pressure is bidirectional (it scores both hypotensive and
hypertensive deviations), so the stored value is the most deviant extreme in
either direction rather than the nadir. Its encoding is bidirectional by
construction and a monotone constraint would be inappropriate.

Implementation
--------------
Constraints are imposed by reparametrisation rather than by a constrained
solver, so L-BFGS-B (which supports box bounds only) can still be used:

    direction = "increasing"  (risk rises with the variable)
        w[0] = 0,  w[b] = sum(delta[1..b]),      delta >= 0
    direction = "decreasing"  (risk falls with the variable)
        w[k-1] = 0,  w[b] = sum(delta[b..k-2]),  delta >= 0
    direction = "free"
        w[b] unconstrained in [-W, W]

Fixing one bin per feature to zero loses no generality: the objective is
exactly invariant to per-feature level shifts, because the score enters the
logistic calibration standardised, and an additive constant changes neither
the standardised score nor its standard deviation. The shift is absorbed by
the calibration intercept.

Integer projection accumulates FIRST and rounds SECOND. `np.rint` is a
non-decreasing function, so applying it to a monotone sequence returns a
monotone sequence. Rounding the increments instead would let a run of small
positive increments collapse to zero.
"""

import numpy as np
from sklearn.isotonic import isotonic_regression

INCREASING = "increasing"
DECREASING = "decreasing"
FREE = "free"
_VALID = (INCREASING, DECREASING, FREE)


class FeatureSpec:
    """Layout of a single feature within the flat parameter vectors."""

    def __init__(self, name, k, direction, bin_start, theta_start):
        if direction not in _VALID:
            raise ValueError(
                f"Unknown direction '{direction}' for feature '{name}'. "
                f"Valid values: {_VALID}"
            )
        if k < 2:
            raise ValueError(f"Feature '{name}' needs k >= 2 bins, got {k}.")

        self.name = name
        self.k = int(k)
        self.direction = direction
        self.bin_slice = slice(bin_start, bin_start + self.k)
        n_theta = self.k if direction == FREE else self.k - 1
        self.theta_slice = slice(theta_start, theta_start + n_theta)
        self.n_theta = n_theta

    @property
    def constrained(self):
        return self.direction != FREE

    def __repr__(self):
        return (f"FeatureSpec({self.name}, k={self.k}, "
                f"direction={self.direction})")


class ParamSpec:
    """Full parameter layout across all features."""

    def __init__(self, feature_names, k_bins_list, directions):
        if len(feature_names) != len(k_bins_list):
            raise ValueError(
                "feature_names and k_bins_list must have the same length "
                f"({len(feature_names)} vs {len(k_bins_list)})."
            )

        self.features = []
        bin_ptr = 0
        theta_ptr = 0

        for name, k in zip(feature_names, k_bins_list):
            direction = directions.get(name, FREE)
            spec = FeatureSpec(name, k, direction, bin_ptr, theta_ptr)
            self.features.append(spec)
            bin_ptr += spec.k
            theta_ptr += spec.n_theta

        self.n_bins = bin_ptr
        self.n_theta = theta_ptr

    def summary(self):
        rows = [
            f"  {f.name:<16} k={f.k:<3} {f.direction}"
            for f in self.features
        ]
        n_con = sum(f.constrained for f in self.features)
        return (f"ParamSpec: {len(self.features)} features, "
                f"{n_con} constrained, {self.n_bins} bins, "
                f"{self.n_theta} free parameters\n" + "\n".join(rows))


def build_param_spec(feature_names, k_bins_list, directions):
    """Build the parameter layout.

    Args:
        feature_names (list[str]): Feature names, in column order.
        k_bins_list (list[int]): Number of bins per feature.
        directions (dict): Maps feature name -> "increasing" | "decreasing" |
            "free". Missing names default to "free".

    Returns:
        ParamSpec
    """
    return ParamSpec(feature_names, k_bins_list, directions)


def theta_to_weights(theta, spec):
    """Map free parameters to the flat weight vector.

    Monotonicity holds by construction for every constrained feature.
    """
    theta = np.asarray(theta, dtype=float)
    w = np.zeros(spec.n_bins, dtype=float)

    for f in spec.features:
        th = theta[f.theta_slice]

        if f.direction == FREE:
            w[f.bin_slice] = th
            continue

        wj = np.zeros(f.k, dtype=float)
        if f.direction == INCREASING:
            # w[0] = 0; increments accumulate upward
            wj[1:] = np.cumsum(th)
        else:
            # w[k-1] = 0; increments accumulate downward
            wj[:-1] = np.cumsum(th[::-1])[::-1]
        w[f.bin_slice] = wj

    return w


def project_weights_to_feasible(w_flat, spec):
    """Project an arbitrary weight vector onto the monotone feasible set.

    Uses isotonic regression (pool adjacent violators) per feature. The
    result is the closest feasible vector in the least-squares sense, and is
    used to obtain a feasible starting point for the optimiser.
    """
    w_flat = np.asarray(w_flat, dtype=float)
    out = w_flat.copy()

    for f in spec.features:
        wj = w_flat[f.bin_slice]

        if f.direction == FREE:
            continue
        if f.direction == INCREASING:
            out[f.bin_slice] = isotonic_regression(wj, increasing=True)
        else:
            out[f.bin_slice] = isotonic_regression(wj, increasing=False)

    return out


def weights_to_theta(w_flat, spec, W_bound=None):
    """Extract free parameters from a (possibly infeasible) weight vector.

    The vector is first projected onto the feasible set, then the per-feature
    increments are read off. Level shifts are discarded, which is harmless:
    they are absorbed by the calibration intercept.
    """
    w_proj = project_weights_to_feasible(w_flat, spec)
    theta = np.zeros(spec.n_theta, dtype=float)

    for f in spec.features:
        wj = w_proj[f.bin_slice]

        if f.direction == FREE:
            theta[f.theta_slice] = wj
            continue

        if f.direction == INCREASING:
            d = np.diff(wj)
        else:
            d = -np.diff(wj)

        # Numerical noise from the isotonic solver can leave tiny negatives
        d = np.maximum(d, 0.0)
        if W_bound is not None:
            d = np.minimum(d, float(W_bound))
        theta[f.theta_slice] = d

    return theta


def theta_bounds(spec, W_bound):
    """Box bounds for the free parameters, in L-BFGS-B format."""
    W = float(W_bound)
    bounds = []
    for f in spec.features:
        if f.direction == FREE:
            bounds.extend([(-W, W)] * f.n_theta)
        else:
            bounds.extend([(0.0, W)] * f.n_theta)
    return bounds


def round_weights_monotone(w_flat, W_bound):
    """Round to integers and clip, preserving monotonicity.

    Both `rint` and `clip` are non-decreasing maps, so a monotone input gives
    a monotone output. No per-feature handling is needed.
    """
    W = int(W_bound)
    w_int = np.rint(np.asarray(w_flat, dtype=float))
    return np.clip(w_int, -W, W).astype(float)


def is_feasible(w_flat, spec, tol=1e-9):
    """Check every constrained feature for monotonicity violations."""
    w_flat = np.asarray(w_flat, dtype=float)
    for f in spec.features:
        wj = w_flat[f.bin_slice]
        if f.direction == INCREASING and np.any(np.diff(wj) < -tol):
            return False
        if f.direction == DECREASING and np.any(np.diff(wj) > tol):
            return False
    return True


def monotonicity_report(w_flat, spec, tol=1e-9):
    """Human-readable per-feature feasibility report."""
    w_flat = np.asarray(w_flat, dtype=float)
    lines = []
    ok_all = True

    for f in spec.features:
        wj = w_flat[f.bin_slice]
        if f.direction == FREE:
            status = "exempt"
        else:
            d = np.diff(wj)
            bad = (np.any(d < -tol) if f.direction == INCREASING
                   else np.any(d > tol))
            status = "VIOLATION" if bad else "ok"
            ok_all = ok_all and not bad
        vals = " ".join(f"{v:+.0f}" for v in wj)
        lines.append(f"  {f.name:<16} {f.direction:<11} {status:<10} [{vals}]")

    header = "Monotonicity check: " + ("PASSED" if ok_all else "FAILED")
    return header + "\n" + "\n".join(lines), ok_all


def recenter_weights(w_flat, spec):
    """Shift each feature so its most favourable bin is zero.

    This is a pure reparametrisation: adding a constant to all weights of a
    feature changes neither the standardised score, its standard deviation,
    nor any prediction. The shift is absorbed by the calibration intercept.

    The purpose is presentational. It makes the weight table read like a SOFA
    table, where every organ contributes 0 in its normal range and positive
    points as it deteriorates, instead of carrying uninterpretable negative
    levels such as "norepinephrine absent -> -4".
    """
    w_flat = np.asarray(w_flat, dtype=float)
    out = w_flat.copy()
    for f in spec.features:
        wj = out[f.bin_slice]
        out[f.bin_slice] = wj - np.min(wj)
    return out


def bin_counts_per_feature(X, thresholds_list, k_bins_list):
    """Observation count in each bin. Depends only on the data and the bins,
    so it can be computed once outside the objective function."""
    counts = []
    for j in range(X.shape[1]):
        idx = np.digitize(
            X[:, j], thresholds_list[j], right=False
        ).astype(int)
        idx = np.clip(idx, 0, k_bins_list[j] - 1)
        counts.append(np.bincount(idx, minlength=k_bins_list[j]).astype(float))
    return np.concatenate(counts)


def sparse_bin_penalty(w_flat, counts_flat, alpha, eps, mode="shrinkage",
                       spec=None):
    """Penalty discouraging large weights in sparsely populated bins.

    mode="shrinkage" (corrected):
        alpha * sum_j sum_b  (w[j,b] - mean_j)^2 / (n[j,b] + eps)

        Shrinks contributions towards the feature's own mean in proportion to
        how little empirical support each bin has, which is what the Methods
        describe the penalty as doing.

        Weights are CENTRED PER FEATURE before squaring. This is required for
        correctness, not cosmetic. The model is invariant to per-feature level
        shifts (the calibration intercept absorbs them), so a penalty on the
        raw weights would penalise an arbitrary and meaningless quantity.
        Worse, it would do so unequally: a monotone-constrained feature is
        parametrised with its reference bin pinned at zero, so reaching a
        given spread costs it more squared norm than an unconstrained feature
        free to straddle zero. Without centring, the penalty silently pushes
        all constrained features towards zero and lets the exempt one absorb
        the score variance demanded by the scale anchor.

    mode="legacy" (as originally published):
        alpha * sum_j sum_b  1 / (n[j,b] + eps)

        This term does not involve the weights at all. Once the bins are
        fixed it is an additive constant with zero gradient, so it cannot
        influence the location of the optimum. It can still perturb results
        indirectly, because L-BFGS-B uses a RELATIVE convergence criterion:
        a large constant inflates the objective and triggers earlier
        stopping. Retained only to reproduce the original results.
    """
    if mode == "legacy":
        return float(alpha) * float(np.sum(1.0 / (counts_flat + eps)))

    if mode == "shrinkage":
        w = np.asarray(w_flat, dtype=float)
        if spec is None:
            w_c = w - np.mean(w)
        else:
            w_c = w.copy()
            for f in spec.features:
                wj = w[f.bin_slice]
                w_c[f.bin_slice] = wj - np.mean(wj)
        return float(alpha) * float(np.sum((w_c ** 2) / (counts_flat + eps)))

    raise ValueError(f"Unknown bin penalty mode: {mode}")