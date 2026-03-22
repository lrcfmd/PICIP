"""
synthetic_calibration.py — Synthetic phase field experiments for predicted_error calibration.

Workflow
--------
1. PhaseFieldSimulation
   Populates a Phase_Field with random synthetic phases at edge, face, and
   internal locations, then Delaunay-triangulates them.

2. MockSampler
   Treats one internal phase as the 'unknown'.  For each mock experiment it
   finds the Delaunay simplex containing the unknown, uses the simplex vertices
   as the known co-existing phases, perturbs the true barycentric weights with
   Gaussian noise (simulating measurement uncertainty), and returns a PICIP
   Sample object.

3. calibrate_predicted_error
   Grid-searches predicted_error values by running multiple mock PICIP
   experiments and measuring how close the probability-weighted mean is to
   the true unknown composition.

Typical usage
-------------
    from setup_phase_field import Phase_Field
    from synthetic_calibration import PhaseFieldSimulation, calibrate_predicted_error

    pf = Phase_Field()
    pf.setup_uncharged(["Fe", "Mn", "Ti"])

    sim = PhaseFieldSimulation(pf, n_edge=2, n_internal=6, seed=42)
    best_pe, results = calibrate_predicted_error(
        sim, weight_noise_sigma=0.05, n_samples=3, n_trials=20, verbose=True
    )
"""

import numpy as np
from itertools import combinations
from scipy.spatial import Delaunay

from run_PICIP import PICIP, Sample


# ── helpers ────────────────────────────────────────────────────────────────────

def _uniform_simplex_point(vertices, rng):
    """Uniform random interior point of a simplex (vertices shape (k, d)).
    Returns (point, barycentric_weights)."""
    weights = rng.dirichlet(np.ones(len(vertices)))
    return weights @ vertices, weights


def _uniform_edge_point(v1, v2, rng, margin=0.1):
    """Uniform random point strictly interior to an edge (avoids corners)."""
    t = rng.uniform(margin, 1 - margin)
    return (1 - t) * v1 + t * v2


def _barycentric(tri, simplex_idx, point):
    """Barycentric coordinates of `point` in the given Delaunay simplex."""
    T = tri.transform[simplex_idx]
    dim = point.shape[0]
    b = T[:dim, :dim] @ (point - T[dim])
    bary = np.append(b, 1 - b.sum())
    return np.clip(bary, 0, 1)


# ── PhaseFieldSimulation ───────────────────────────────────────────────────────

class PhaseFieldSimulation:
    """
    Populates a Phase_Field with synthetic phases and Delaunay-triangulates them.

    Phase locations
    ---------------
    'corner'   — the simplex corners (always included).
    'edge'     — random points on each edge of the simplex.
    'face'     — random points on each triangular face (3-D only).
    'internal' — random points strictly inside the simplex.

    Parameters
    ----------
    phase_field : Phase_Field
    n_edge : int
        Phases per edge.
    n_face : int
        Phases per triangular face (3-D only; ignored in 2-D).
    n_internal : int
        Internal phases; one will be chosen as the 'unknown'.
    seed : int or None
    """

    def __init__(self, phase_field, n_edge=2, n_face=2, n_internal=5, seed=None):
        self.pf = phase_field
        self.dim = phase_field.constrained_dim
        self.rng = np.random.default_rng(seed)

        corners_c = phase_field.convert_to_constrained_basis(phase_field.corners)
        self.corners_c = corners_c

        self._generate(corners_c, n_edge, n_face, n_internal)
        self._triangulate()

    # ── generation ─────────────────────────────────────────────────────────────

    def _generate(self, corners_c, n_edge, n_face, n_internal):
        points, labels = [], []

        # Corners
        for c in corners_c:
            points.append(c)
            labels.append('corner')

        # Edges
        for i, j in self.pf.edges:
            for _ in range(n_edge):
                pt = _uniform_edge_point(corners_c[i], corners_c[j], self.rng)
                points.append(pt)
                labels.append('edge')

        # Faces (3-D: each set of 3 corners forms a face)
        if self.dim == 3 and n_face > 0:
            for face in combinations(range(len(corners_c)), 3):
                verts = corners_c[list(face)]
                for _ in range(n_face):
                    pt, _ = _uniform_simplex_point(verts, self.rng)
                    points.append(pt)
                    labels.append('face')

        # Internal
        for _ in range(n_internal):
            pt, _ = _uniform_simplex_point(corners_c, self.rng)
            points.append(pt)
            labels.append('internal')

        self.points_c = np.array(points)
        self.labels = np.array(labels)
        self.internal_indices = np.where(self.labels == 'internal')[0].tolist()

    # ── triangulation ───────────────────────────────────────────────────────────

    def _triangulate(self):
        self.tri = Delaunay(self.points_c)

    # ── queries ─────────────────────────────────────────────────────────────────

    def find_containing_simplex(self, point_c):
        """
        Find the Delaunay simplex containing `point_c`.

        Returns
        -------
        vertex_indices : list[int]   Indices into self.points_c.
        vertex_coords  : ndarray (dim+1, dim)  Constrained-basis coords.
        bary           : ndarray (dim+1,)       Barycentric weights.

        Raises ValueError if the point lies outside the triangulation.
        """
        s_idx = self.tri.find_simplex(point_c)
        if s_idx == -1:
            raise ValueError(
                f"Point {point_c} is outside the triangulated domain.  "
                "Try increasing n_edge / n_face to fill the boundary better."
            )
        vertex_indices = self.tri.simplices[s_idx].tolist()
        vertex_coords  = self.points_c[vertex_indices]
        bary           = _barycentric(self.tri, s_idx, point_c)
        return vertex_indices, vertex_coords, bary

    def summary(self):
        counts = {k: int(np.sum(self.labels == k))
                  for k in ('corner', 'edge', 'face', 'internal')}
        print(f"PhaseFieldSimulation  dim={self.dim}  total phases={len(self.points_c)}")
        for k, v in counts.items():
            print(f"  {k:8s}: {v}")
        print(f"  Delaunay simplices: {len(self.tri.simplices)}")


# ── MockSampler ────────────────────────────────────────────────────────────────

class MockSampler:
    """
    Generates mock Sample objects for a chosen internal 'unknown' phase.

    The sample composition is always the true unknown.  The known phases are
    the vertices of the Delaunay simplex containing the unknown.  Molar weights
    are the true barycentric coordinates perturbed by Gaussian noise and
    renormalised.

    Parameters
    ----------
    simulation : PhaseFieldSimulation
    unknown_index : int
        Index into simulation.points_c.  Must be an internal phase.
    weight_noise_sigma : float
        Standard deviation of additive Gaussian noise on the barycentric
        weights (before renormalisation).  0 = noiseless.
    """

    def __init__(self, simulation, unknown_index, weight_noise_sigma=0.05):
        if simulation.labels[unknown_index] != 'internal':
            raise ValueError(
                f"Index {unknown_index} has label '{simulation.labels[unknown_index]}'. "
                "Choose an 'internal' phase as the unknown."
            )
        self.sim   = simulation
        self.pf    = simulation.pf
        self.sigma = weight_noise_sigma

        # True unknown in both bases
        self.unknown_c = simulation.points_c[unknown_index]
        self.unknown_s = self.pf.convert_to_standard_basis(
            self.unknown_c[np.newaxis])[0]
        self.unknown_comp = self.pf.convert_standard_to_pymatgen(self.unknown_s)

        # Simplex containing the unknown
        vert_idx, vert_c, true_bary = simulation.find_containing_simplex(self.unknown_c)
        self.vert_indices = vert_idx
        self.vert_c       = vert_c          # (dim+1, dim) constrained
        self.true_bary    = true_bary       # (dim+1,) true barycentric weights

        # Pymatgen Composition for each vertex
        vert_s = self.pf.convert_to_standard_basis(vert_c)
        self.known_comps = [self.pf.convert_standard_to_pymatgen(v) for v in vert_s]

    # ── single sample ───────────────────────────────────────────────────────────

    def make_sample(self, name, predicted_error, rng=None):
        """
        One mock Sample with noisy known-phase weights.

        Parameters
        ----------
        name : str
        predicted_error : float   Value to assign to sample.predicted_error.
        rng : numpy Generator or None
        """
        if rng is None:
            rng = np.random.default_rng()

        noisy = self.true_bary + rng.normal(0, self.sigma, size=len(self.true_bary))
        noisy = np.clip(noisy, 1e-3, None)
        noisy /= noisy.sum()

        s = Sample(name, self.unknown_comp)
        s.add_knowns([c.formula for c in self.known_comps])
        s.add_molar_weights(noisy.tolist())
        s.set_predicted_error(predicted_error)
        return s

    def make_samples(self, n, predicted_error, seed=None):
        """Make `n` independent noisy mock samples."""
        rng = np.random.default_rng(seed)
        return [self.make_sample(f's{i+1}', predicted_error, rng=rng)
                for i in range(n)]

    def true_composition_constrained(self):
        return self.unknown_c.copy()


# ── single trial ───────────────────────────────────────────────────────────────

def _run_trial(sampler, predicted_error, n_samples, n_l, n_p, rng):
    """
    Run one PICIP trial; return Euclidean distance (constrained) from
    probability-weighted mean to true unknown.  Returns np.inf on failure.
    """
    samples = sampler.make_samples(
        n_samples, predicted_error, seed=int(rng.integers(1_000_000))
    )
    pf = sampler.pf
    picip = PICIP(pf)
    for s in samples:
        picip.add_sample(s)

    try:
        pred = picip.run(n_l=n_l, n_p=n_p)
    except Exception:
        return np.inf

    p = pred.prob_density
    if p.sum() == 0:
        return np.inf

    mean_c = (p / p.sum()) @ pf.omega
    return float(np.linalg.norm(mean_c - sampler.unknown_c))


# ── calibration ────────────────────────────────────────────────────────────────

def calibrate_predicted_error(
    simulation,
    unknown_index=None,
    weight_noise_sigma=0.05,
    candidates=None,
    n_samples=3,
    n_trials=20,
    n_l=30,
    n_p=20,
    seed=None,
    verbose=True,
):
    """
    Grid-search predicted_error to minimise mean prediction error.

    For each candidate value, `n_trials` independent PICIP experiments are run
    (each with `n_samples` noisy mock samples).  The mean Euclidean distance
    (constrained basis) from the prediction's probability-weighted mean to the
    true unknown is computed, and the candidate with the lowest score is
    returned.

    Parameters
    ----------
    simulation : PhaseFieldSimulation
    unknown_index : int or None
        Index into simulation.points_c.  None → first internal phase.
    weight_noise_sigma : float
        Noise on known-phase weights in mock samples.
    candidates : list[float] or None
        predicted_error values to evaluate.
        Default: [0.02, 0.05, 0.1, 0.2, 0.3, 0.5]
    n_samples : int
        Mock samples per PICIP run.
    n_trials : int
        Repeated runs per candidate (reduces Monte-Carlo noise).
    n_l, n_p : int
        PICIP resolution parameters.
    seed : int or None
    verbose : bool

    Returns
    -------
    best_predicted_error : float
    results : dict  {predicted_error: mean_distance}
    """
    if unknown_index is None:
        unknown_index = simulation.internal_indices[0]

    if candidates is None:
        candidates = [0.02, 0.05, 0.1, 0.2, 0.3, 0.5]

    sampler = MockSampler(simulation, unknown_index,
                          weight_noise_sigma=weight_noise_sigma)

    if verbose:
        true_c = sampler.unknown_c
        print(f"Unknown: {sampler.unknown_comp.reduced_formula}  "
              f"(constrained {np.round(true_c, 3)})")
        known_names = [c.reduced_formula for c in sampler.known_comps]
        print(f"Known simplex: {' + '.join(known_names)}")
        print(f"True weights:  {np.round(sampler.true_bary, 3)}")
        print(f"Weight noise σ={weight_noise_sigma}  "
              f"n_samples={n_samples}  n_trials={n_trials}")
        print()

    rng = np.random.default_rng(seed)
    results = {}

    for pe in candidates:
        distances = [
            _run_trial(sampler, pe, n_samples, n_l, n_p, rng)
            for _ in range(n_trials)
        ]
        finite = [d for d in distances if np.isfinite(d)]
        mean_d = float(np.mean(finite)) if finite else np.inf
        results[pe] = mean_d
        if verbose:
            print(f"  predicted_error={pe:.3f}  mean_distance={mean_d:.4f}"
                  f"  ({len(finite)}/{n_trials} valid trials)")

    best = min(results, key=results.get)
    if verbose:
        print(f"\n→ best predicted_error = {best}  "
              f"(mean_distance={results[best]:.4f})")

    return best, results


# ── multi-unknown sweep ────────────────────────────────────────────────────────

def sweep_unknowns(
    simulation,
    weight_noise_sigma=0.05,
    candidates=None,
    n_samples=3,
    n_trials=20,
    n_l=30,
    n_p=20,
    seed=None,
    verbose=True,
):
    """
    Run calibrate_predicted_error for every internal phase and return
    a summary of recommended predicted_error values.

    Returns
    -------
    dict  {unknown_index: (best_pe, results_dict)}
    """
    all_results = {}
    for idx in simulation.internal_indices:
        if verbose:
            print(f"\n{'═'*60}")
            print(f"Unknown index {idx} / {len(simulation.points_c)-1}")
        best, res = calibrate_predicted_error(
            simulation, unknown_index=idx,
            weight_noise_sigma=weight_noise_sigma,
            candidates=candidates,
            n_samples=n_samples, n_trials=n_trials,
            n_l=n_l, n_p=n_p, seed=seed, verbose=verbose,
        )
        all_results[idx] = (best, res)

    if verbose:
        print(f"\n{'═'*60}")
        print("Summary:")
        best_values = [v[0] for v in all_results.values()]
        print(f"  Recommended predicted_error values: {best_values}")
        print(f"  Median: {np.median(best_values):.3f}")
        print(f"  Mean:   {np.mean(best_values):.3f}")

    return all_results
