"""
picip.evaluation — build synthetic phase fields and score PICIP against them.

Everything here exists to answer one question: *if we hide a phase we already
know the answer for, how close does PICIP get to it?*  Because the truth is
known by construction, every stage of the inference can be checked and scored.

The workflow has four steps, and one function each.

1. **Generate** a phase field, or a set of them::

       from picip.evaluation import generate_test_phase_field
       field = generate_test_phase_field(dim=3, structure='4c1a', seed=0)

2. **Run** PICIP against it, hiding each internal phase in turn.  One shot
   (a single batch of suggestions) or iterative (each batch is observed and
   fed back, so PICIP chooses where to look next)::

       from picip.evaluation import evaluate
       one_shot  = evaluate(field, batch_size=5)
       iterative = evaluate(field, batch_size=3, n_batches=4)

3. **Inspect** a single trial stage by stage — the Dirichlet support on the
   known simplex, the projected rays, the interpolated density, and the
   direction implied by the measured weights::

       from picip.evaluation import diagnostic_figure, save_figure
       save_figure(diagnostic_figure(iterative[0]), 'trial.html')

4. **Score** how the closest suggestion improves as batches accumulate::

       from picip.evaluation import distance_vs_batch_figure
       save_figure(distance_vs_batch_figure(iterative), 'convergence.html')

Figure functions return plotly ``Figure`` objects and never write to disk on
their own; :func:`save_figure` writes one self-contained HTML file with the
plotly library inlined, so there is no ``plotly.min.js`` sidecar to keep
alongside it.

Field generation
----------------
:func:`generate_test_phase_field` picks a small set of common single-valence
ions, builds a charge-neutral ``Phase_Field`` of the requested dimensionality,
and populates it with random charge-balanced phases placed in the interior,
along the edges, and (in 3-D) on the faces of the constrained polytope.

Dimensionality follows directly from charge neutrality: ``n`` elements with the
sum-to-one and one charge-neutrality constraint give ``constrained_dim = n - 2``.

    ┌────────┬──────────────┬───────────────┬─────────┬──────────────────┐
    │  dim   │  structure   │ cations/anions│ elements│ polytope         │
    ├────────┼──────────────┼───────────────┼─────────┼──────────────────┤
    │   2    │  '2c2a'      │      2 / 2    │    4    │ quadrilateral    │
    │   2    │  '3c1a'      │      3 / 1    │    4    │ triangle         │
    │   3    │  '4c1a'      │      4 / 1    │    5    │ tetrahedron      │
    │   3    │  '3c2a'      │      3 / 2    │    5    │ prism (6 corners)│
    └────────┴──────────────┴───────────────┴─────────┴──────────────────┘

The corners of a charged field are always the neutral binaries (every
cation×anion pairing), so the generated phases are genuine multi-component
mixtures of those end-members and are neutral by construction.
"""

import itertools
import math
import numbers
import os
from dataclasses import dataclass, field as field_

# Cap BLAS/OpenMP threads to 1 before numpy loads.  This workload (Qhull /
# griddata / small linear algebra) does not benefit from BLAS threading, and
# multi-threaded BLAS oversubscribes cores badly once trials run in parallel
# (n_jobs > 1) — and is even slightly slower serially.  ``setdefault`` lets a
# user override by exporting these variables themselves.
for _var in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
             'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_var, '1')

import numpy as np
from scipy.spatial import ConvexHull, Delaunay

# Import submodules directly rather than the package root, so that
# ``picip/__init__.py`` is free to import this module without a cycle.
from picip.phase_field import Phase_Field
from picip.picip import PICIP, Prediction, Sample
from picip.visualise_cube import make_plotter
from picip.exceptions import ZeroPException


# ── ion pools ────────────────────────────────────────────────────────────────
# Common ions restricted to a single, well-defined formal charge (so a plain
# ``setup_charged`` is unambiguous).  Cations cover +1/+2/+3, anions -1/-2.

CATIONS = {
    1: ['Li', 'Na', 'K', 'Rb', 'Cs', 'Ag'],
    2: ['Mg', 'Ca', 'Sr', 'Ba', 'Zn', 'Cd'],
    3: ['Al', 'Sc', 'Y', 'Ga', 'In', 'La'],
}
ANIONS = {
    -1: ['F', 'Cl', 'Br', 'I'],
    -2: ['O', 'S', 'Se', 'Te'],
}

# structure name -> (n_cations, n_anions); grouped by resulting constrained_dim.
STRUCTURES = {
    2: {'2c2a': (2, 2), '3c1a': (3, 1)},
    3: {'4c1a': (4, 1), '3c2a': (3, 2)},
}


# ── result container ─────────────────────────────────────────────────────────

@dataclass
class TestPhaseField:
    """A generated charged phase field plus its random test phases."""
    phase_field: Phase_Field
    species: dict                       # {element: formal charge}
    structure: str                      # e.g. '2c2a'
    dim: int                            # constrained_dim (2 or 3)
    corners_constrained: np.ndarray     # (C, dim)
    corners_standard: np.ndarray        # (C, nelements)
    corner_formulas: list               # neutral-binary end-members, len C
    phases_constrained: np.ndarray      # (P, dim)
    phases_standard: np.ndarray         # (P, nelements)
    phase_compositions: list            # pymatgen Composition, len P
    phase_formulas: list                # len P
    phase_labels: np.ndarray            # 'internal' | 'edge' | 'face', len P
    # Combined corner+phase vertex set and its fixed triangulation.  Every point
    # is a real phase in the field (corners + generated phases); the stored
    # triangulation makes "which phases form here" deterministic even when the
    # point set is geometrically degenerate.
    points_constrained: np.ndarray      # (C+P, dim)   corners first, then phases
    points_standard: np.ndarray         # (C+P, nelements)
    point_labels: np.ndarray            # 'corner' | 'internal' | 'edge' | 'face'
    point_formulas: list                # len C+P
    triangulation: Delaunay             # fixed Delaunay over points_constrained
    seed: int = None

    def summary(self):
        """Return a short human-readable description string."""
        ion_str = ', '.join(f'{el}{_charge_str(q)}' for el, q in self.species.items())
        counts = {lab: int((self.phase_labels == lab).sum())
                  for lab in ('internal', 'edge', 'face')}
        lines = [
            f"TestPhaseField  structure={self.structure}  dim={self.dim}",
            f"  ions      : {ion_str}",
            f"  corners   : {len(self.corner_formulas)}  ({', '.join(self.corner_formulas)})",
            f"  phases    : {len(self.phase_formulas)}  "
            f"(internal={counts['internal']}, edge={counts['edge']}, face={counts['face']})",
            f"  simplices : {len(self.triangulation.simplices)}  "
            f"(fixed triangulation over {len(self.points_constrained)} vertices)",
        ]
        return '\n'.join(lines)

    def is_neutral(self):
        """True if every generated phase satisfies charge neutrality."""
        if len(self.phases_standard) == 0:
            return True
        return bool(self.phase_field.check_neutrality(self.phases_standard))

    def _to_constrained(self, point):
        """Coerce a query point (constrained vec, standard vec, or Composition)
        to a constrained-basis coordinate."""
        if hasattr(point, 'get_atomic_fraction'):          # pymatgen Composition
            std = np.array([point.get_atomic_fraction(el)
                            for el in self.phase_field.elements], dtype=float)
            return self.phase_field.convert_to_constrained_basis(std)
        p = np.asarray(point, dtype=float)
        if p.shape[-1] == self.dim:                        # already constrained
            return p
        if p.shape[-1] == self.phase_field.nelements:      # standard fractions
            return self.phase_field.convert_to_constrained_basis(p)
        raise ValueError(
            f"query point has length {p.shape[-1]}; expected {self.dim} "
            f"(constrained) or {self.phase_field.nelements} (standard)")

    def find_forming_phases(self, point):
        """
        Find the phases that would form at a sampled composition.

        Locates the simplex of the fixed triangulation that contains ``point``
        and returns its ``dim + 1`` vertices (3 in 2-D, 4 in 3-D) — the phases
        the sample decomposes into — together with their barycentric mixing
        weights.  Because the triangulation is fixed at generation, the answer
        is deterministic even where the point set is degenerate.

        Parameters
        ----------
        point : ndarray or pymatgen Composition
            The sampled composition, as a constrained-basis vector, a standard
            atomic-fraction vector, or a pymatgen ``Composition``.

        Returns
        -------
        FormingPhases or None
            ``None`` if the point lies outside the phase field's convex hull.
        """
        point_c = self._to_constrained(point)
        s_idx = int(self.triangulation.find_simplex(np.atleast_2d(point_c))[0])
        if s_idx < 0:
            return None
        verts = [int(v) for v in self.triangulation.simplices[s_idx]]
        weights = _barycentric(self.triangulation, s_idx, point_c)
        return FormingPhases(
            indices=verts,
            points_constrained=self.points_constrained[verts],
            formulas=[self.point_formulas[v] for v in verts],
            labels=[str(self.point_labels[v]) for v in verts],
            weights=weights,
            simplex_index=s_idx)

    def random_sample_point(self, seed=None):
        """Draw a uniform-ish random composition inside the field (constrained
        basis) — a convenience for exercising :meth:`find_forming_phases`."""
        rng = np.random.default_rng(seed)
        w = rng.dirichlet(np.ones(len(self.corners_constrained)))
        return w @ self.corners_constrained


@dataclass
class FormingPhases:
    """The phases a sampled composition decomposes into (one simplex)."""
    indices: list                   # indices into TestPhaseField.points_*
    points_constrained: np.ndarray  # (dim+1, dim)
    formulas: list                  # len dim+1
    labels: list                    # 'corner'/'internal'/'edge'/'face'
    weights: np.ndarray             # barycentric, len dim+1, sums to 1
    simplex_index: int

    def __repr__(self):
        parts = ', '.join(f'{f}:{w:.3f}'
                          for f, w in zip(self.formulas, self.weights))
        return f'FormingPhases({parts})'


def _barycentric(tri, simplex_idx, point):
    """Barycentric coordinates of ``point`` within a Delaunay simplex."""
    T = tri.transform[simplex_idx]
    dim = point.shape[0]
    b = T[:dim, :dim] @ (point - T[dim])
    bary = np.append(b, 1.0 - b.sum())
    return np.clip(bary, 0.0, None)


def _charge_str(q):
    """'+2' / '-1' style superscript-free charge label."""
    return f'{q:+d}' if float(q).is_integer() else f'{q:+g}'


# ── species selection ────────────────────────────────────────────────────────

def choose_species(structure, rng, dim=None):
    """
    Randomly pick the ions for a structure and return ``{element: charge}``.

    ``n_cations`` cations are drawn from the combined +1/+2/+3 pools and
    ``n_anions`` anions from the -1/-2 pools, all distinct, so the charge
    tiers are sampled naturally.  A valid charge-neutral region always exists
    because there is at least one cation and one anion.

    Parameters
    ----------
    structure : str
        One of the keys in :data:`STRUCTURES` (``'2c2a'``, ``'3c1a'``,
        ``'4c1a'``, ``'3c2a'``).
    rng : numpy.random.Generator
    dim : int, optional
        If given, validates that ``structure`` belongs to that dimension.
    """
    d = _dim_of_structure(structure)
    if dim is not None and d != dim:
        raise ValueError(f"structure {structure!r} is dim {d}, not {dim}")
    n_cat, n_an = STRUCTURES[d][structure]

    cation_pool = [(el, q) for q, els in CATIONS.items() for el in els]
    anion_pool = [(el, q) for q, els in ANIONS.items() for el in els]

    cat_idx = rng.choice(len(cation_pool), size=n_cat, replace=False)
    an_idx = rng.choice(len(anion_pool), size=n_an, replace=False)

    species = {}
    for i in cat_idx:
        el, q = cation_pool[i]
        species[el] = q
    for i in an_idx:
        el, q = anion_pool[i]
        species[el] = q
    return species


def _dim_of_structure(structure):
    for d, table in STRUCTURES.items():
        if structure in table:
            return d
    raise ValueError(
        f"unknown structure {structure!r}; choose from "
        f"{[s for t in STRUCTURES.values() for s in t]}")


# ── polytope geometry ────────────────────────────────────────────────────────

def _polytope_faces(corners_c):
    """
    Group the convex-hull facets of a 3-D corner set into planar faces.

    ``ConvexHull`` triangulates every facet, so coplanar triangles (e.g. the two
    halves of a rectangular prism face) share a hull equation.  Returns a list
    of faces, each a list of vertex-index triples (triangles that tile it).
    """
    hull = ConvexHull(corners_c)
    faces = {}
    for simplex, eq in zip(hull.simplices, hull.equations):
        key = tuple(np.round(eq, 5))
        faces.setdefault(key, []).append([int(v) for v in simplex])
    return list(faces.values())


def _polytope_edges_from_faces(faces):
    """
    Recover the true polytope edges from triangulated planar faces.

    Within one planar face the perimeter segments appear in exactly one triangle
    while triangulation diagonals appear in two, so keeping the count-one
    segments of each face and taking the union across faces drops the diagonals
    that ``ConvexHull`` introduces across non-triangular faces.
    """
    from collections import Counter
    true_edges = set()
    for tris in faces:
        counts = Counter()
        for i, j, k in tris:
            for a, b in ((i, j), (j, k), (i, k)):
                counts[tuple(sorted((a, b)))] += 1
        true_edges.update(e for e, c in counts.items() if c == 1)
    return sorted(true_edges)


def _sample_in_triangle(a, b, c, rng):
    """Uniform random point inside the triangle (a, b, c)."""
    r1, r2 = rng.random(), rng.random()
    if r1 + r2 > 1.0:
        r1, r2 = 1.0 - r1, 1.0 - r2
    return a + r1 * (b - a) + r2 * (c - a)


def _triangle_area(a, b, c):
    """Area of a triangle given by three points in 2-D or 3-D."""
    return 0.5 * np.linalg.norm(np.cross(b - a, c - a))


def _sample_on_face(face_tris, corners_c, rng):
    """Sample a point on a face, choosing among its triangles by area."""
    areas = np.array([_triangle_area(corners_c[i], corners_c[j], corners_c[k])
                      for i, j, k in face_tris])
    if areas.sum() <= 0:
        i, j, k = face_tris[0]
    else:
        i, j, k = face_tris[rng.choice(len(face_tris), p=areas / areas.sum())]
    return _sample_in_triangle(corners_c[i], corners_c[j], corners_c[k], rng)


# ── main generator ───────────────────────────────────────────────────────────

def generate_test_phase_field(dim=2, structure=None, *, n_internal=5, n_edge=1,
                              n_face=1, species=None, resolution=None,
                              n_points=None, edge_margin=0.08, seed=None):
    """
    Build a random charged phase field and populate it with test phases.

    Parameters
    ----------
    dim : int, default 2
        Constrained dimensionality, 2 or 3.  Ignored when ``structure`` or
        ``species`` fixes it.
    structure : str, optional
        Cation/anion layout: ``'2c2a'`` or ``'3c1a'`` (2-D); ``'4c1a'`` or
        ``'3c2a'`` (3-D).  If omitted, a valid structure for ``dim`` is chosen
        at random.
    n_internal : int, default 5
        Total number of random phases placed strictly inside the polytope.
    n_edge : int, default 1
        Total number of edge phases; each is placed on a randomly chosen edge.
    n_face : int, default 1
        Total number of face phases; each is placed on a randomly chosen face
        (3-D only; ignored in 2-D).
    species : dict, optional
        Explicit ``{element: charge}`` mapping.  Overrides ``structure``/``dim``
        (both are inferred from the number of cations/anions).
    resolution, n_points : int, optional
        Passed through to :meth:`Phase_Field.setup_charged` to size the grid.
    edge_margin : float, default 0.08
        Keep edge phases at least this far (in barycentric fraction) from the
        corners, so they are clearly on the edge interior.
    seed : int, optional
        Seed for reproducible ion choice and phase placement.

    Returns
    -------
    TestPhaseField
    """
    rng = np.random.default_rng(seed)

    # 1. Resolve species -----------------------------------------------------
    if species is not None:
        species = dict(species)
        n_cat = sum(1 for q in species.values() if q > 0)
        n_an = sum(1 for q in species.values() if q < 0)
        dim = len(species) - 2
        structure = _match_structure(dim, n_cat, n_an) or f'{n_cat}c{n_an}a'
    else:
        if dim not in STRUCTURES:
            raise ValueError(f"dim must be one of {list(STRUCTURES)}, got {dim}")
        if structure is None:
            structure = rng.choice(sorted(STRUCTURES[dim]))
        elif structure not in STRUCTURES[dim]:
            raise ValueError(
                f"structure {structure!r} is not valid for dim {dim}; "
                f"choose from {sorted(STRUCTURES[dim])}")
        species = choose_species(structure, rng, dim=dim)

    # 2. Build the charged phase field --------------------------------------
    # Phase_Field draws its constrained basis from numpy's *global* RNG
    # (phase_field.py: rand_vec = np.random.rand(...)), so without intervention
    # the omega grid orientation — and therefore every downstream result —
    # varies from process to process.  Seed the global RNG deterministically
    # from this field's rng (saving/restoring state so nothing else is affected)
    # to make generation fully reproducible.
    pf = Phase_Field()
    _global_state = np.random.get_state()
    np.random.seed(int(rng.integers(0, 2**31 - 1)))
    try:
        pf.setup_charged(species, resolution=resolution, n_points=n_points)
    finally:
        np.random.set_state(_global_state)
    if pf.constrained_dim != dim:
        raise RuntimeError(
            f"expected constrained_dim {dim} but got {pf.constrained_dim} "
            f"for species {species}")

    corners_s = np.asarray(pf.corners, dtype=float)
    corners_c = pf.convert_to_constrained_basis(corners_s)
    corner_formulas = [_formula(c) for c in pf.convert_standard_to_pymatgen(corners_s)]
    n_corners = len(corners_c)

    # Resolve boundary structure.  In 3-D derive true edges from the faces so
    # that hull-triangulation diagonals are not mistaken for polytope edges; in
    # 2-D the hull simplices already are the polygon edges.
    if dim == 3:
        faces = _polytope_faces(corners_c)
        edges = _polytope_edges_from_faces(faces)
    else:
        faces = []
        edges = [tuple(int(v) for v in e) for e in pf.edges]

    # 3. Generate random phases in constrained space (neutral by construction)
    pts_c, labels = [], []

    # interior — convex combination of all corners with positive weights
    for _ in range(n_internal):
        w = rng.dirichlet(np.ones(n_corners))
        pts_c.append(w @ corners_c)
        labels.append('internal')

    # edges — n_edge phases total, each on a randomly chosen edge
    if edges and n_edge > 0:
        for _ in range(n_edge):
            i, j = edges[rng.choice(len(edges))]
            t = rng.uniform(edge_margin, 1.0 - edge_margin)
            pts_c.append((1.0 - t) * corners_c[i] + t * corners_c[j])
            labels.append('edge')

    # faces — n_face phases total, each on a randomly chosen face (3-D only)
    if dim == 3 and faces and n_face > 0:
        for _ in range(n_face):
            face_tris = faces[rng.choice(len(faces))]
            pts_c.append(_sample_on_face(face_tris, corners_c, rng))
            labels.append('face')

    pts_c = np.array(pts_c) if pts_c else np.empty((0, dim))
    pts_s = (pf.convert_to_standard_basis(pts_c) if len(pts_c)
             else np.empty((0, pf.nelements)))
    comps = list(pf.convert_standard_to_pymatgen(pts_s)) if len(pts_s) else []
    formulas = [_formula(c) for c in comps]

    # 4. Combined vertex set (corners first, then phases) and its fixed
    #    triangulation.  Computing it once here — rather than per query — is what
    #    makes "which phases form at this sample" deterministic and reproducible
    #    when the point set is degenerate.
    points_c = np.vstack([corners_c, pts_c]) if len(pts_c) else corners_c.copy()
    points_s = np.vstack([corners_s, pts_s]) if len(pts_s) else corners_s.copy()
    point_labels = np.array(['corner'] * n_corners + list(labels))
    point_formulas = list(corner_formulas) + formulas
    triangulation = Delaunay(points_c)

    result = TestPhaseField(
        phase_field=pf, species=species, structure=structure, dim=dim,
        corners_constrained=corners_c, corners_standard=corners_s,
        corner_formulas=corner_formulas,
        phases_constrained=pts_c, phases_standard=pts_s,
        phase_compositions=comps, phase_formulas=formulas,
        phase_labels=np.array(labels),
        points_constrained=points_c, points_standard=points_s,
        point_labels=point_labels, point_formulas=point_formulas,
        triangulation=triangulation, seed=seed)

    # 5. Safety check — every generated phase must be charge neutral ---------
    if not result.is_neutral():
        raise RuntimeError("generated a non-neutral phase — this is a bug")
    return result


def _match_structure(dim, n_cat, n_an):
    for name, (c, a) in STRUCTURES.get(dim, {}).items():
        if (c, a) == (n_cat, n_an):
            return name
    return None


def _formula(composition):
    """Reduced formula string for a pymatgen Composition."""
    try:
        return composition.reduced_formula
    except Exception:
        return str(composition)


# ── sets of phase fields ─────────────────────────────────────────────────────

def _as_values(x):
    """
    Normalise a phase-count argument to a list of candidate values.

    * ``int``            → a single fixed value, e.g. ``5`` → ``[5]``
    * ``range``          → its values, e.g. ``range(2, 5)`` → ``[2, 3, 4]``
    * ``(lo, hi)`` tuple → the **inclusive** range, e.g. ``(2, 5)`` → ``[2..5]``
    * ``list``/iterable  → the explicit values as given, e.g. ``[1, 4]`` → ``[1, 4]``
    """
    if isinstance(x, numbers.Integral) and not isinstance(x, bool):
        return [int(x)]
    if isinstance(x, range):
        return list(x)
    if isinstance(x, tuple) and len(x) == 2:
        lo, hi = int(x[0]), int(x[1])
        return list(range(lo, hi + 1))
    return [int(v) for v in x]


def generate_phase_field_set(dim=2, *, n_internal=5, n_edge=2, n_face=1,
                             repeats=1, structure=None, resolution=None,
                             n_points=None, seed=None):
    """
    Build a set of test phase fields, all of the same dimensionality.

    Each phase-count argument (``n_internal``, ``n_edge``, ``n_face``) may be a
    fixed value or a range of values — see :func:`_as_values` for the accepted
    forms.  One phase field is generated for **every unique combination** of the
    ranged values, and ``repeats`` independent draws are produced for each
    combination.

    * All arguments fixed  → the set is exactly ``repeats`` fields.
    * Any argument ranged  → the set is ``(n_combinations) * repeats`` fields.

    In 2-D, ``n_face`` is irrelevant and does not contribute combinations.

    Parameters
    ----------
    dim : int, default 2
        Constrained dimensionality shared by every field, 2 or 3.
    n_internal, n_edge, n_face : int or range-like
        Total phase counts (see :func:`generate_test_phase_field`), each either
        fixed or a range.
    repeats : int, default 1
        Number of independent fields per unique combination; equivalently the
        total number of fields when no argument is ranged.
    structure : str, optional
        Fix the cation/anion layout for every field (e.g. ``'2c2a'``).  If
        omitted, each field draws a random structure valid for ``dim``.
    resolution, n_points : int, optional
        Grid sizing, passed through to each field.
    seed : int, optional
        Base seed; each field is given a distinct, reproducible child seed so
        repeats of the same combination differ.

    Returns
    -------
    list of TestPhaseField
    """
    if dim not in STRUCTURES:
        raise ValueError(f"dim must be one of {list(STRUCTURES)}, got {dim}")
    if repeats < 1:
        raise ValueError(f"repeats must be >= 1, got {repeats}")

    internal_vals = _as_values(n_internal)
    edge_vals = _as_values(n_edge)
    face_vals = _as_values(n_face) if dim == 3 else [0]

    combos = list(itertools.product(internal_vals, edge_vals, face_vals))
    child_seeds = np.random.SeedSequence(seed).spawn(len(combos) * repeats)

    fields, k = [], 0
    for ni, ne, nf in combos:
        for _ in range(repeats):
            fseed = int(child_seeds[k].generate_state(1, dtype=np.uint32)[0])
            k += 1
            fields.append(generate_test_phase_field(
                dim=dim, structure=structure, n_internal=ni, n_edge=ne,
                n_face=nf, resolution=resolution, n_points=n_points,
                seed=fseed))
    return fields



# ── evaluation: hide one phase, recover it with PICIP ────────────────────────

@dataclass
class BatchResult:
    """One batch of suggestions within a trial.

    In a one-shot run (``n_batches=1``) a trial has exactly one of these.  In an
    iterative run there is one per round, and ``min_distance`` across the list is
    the convergence curve plotted by :func:`distance_vs_batch_figure`.
    """
    index: int                          # 0-based batch number
    points_constrained: np.ndarray      # (batch_size, dim) suggestions
    min_distance: float                 # closest suggestion to the true unknown
    #: Samples observed *before* this batch was suggested.  Round 0 sees the
    #: single seed composition; later rounds also see the previous batch's
    #: points, minus any that did not produce the unknown phase.
    n_samples: int = 0
    n_skipped: int = 0                  # suggestions that did not form it


@dataclass
class TrialResult:
    """Outcome of one hide-and-recover trial for a single unknown phase.

    A *trial* hides one phase, seeds PICIP with one composition that forms it,
    and then runs ``n_batches`` rounds of suggestion.  ``min_distance`` is the
    score: the closest suggestion to the true unknown phase, over the final
    batch.  ``batches`` carries the per-round detail.
    """
    phase_field_index: int
    unknown_index: int                  # index into TestPhaseField.points_*
    unknown_formula: str
    unknown_constrained: np.ndarray
    repeat: int
    version: str                        # 'dirichlet' or 'gaussian'
    batch_method: str                   # 'kmedoids' or 'iid'
    experimental_error: float
    batch_size: int
    n_batches: int
    n_knowns: int
    known_formulas: list
    sample_constrained: np.ndarray      # the seed composition
    predicted_error: float = None       # σ handed to PICIP
    min_distance: float = None          # closest suggestion, final batch
    best_distance: float = None         # closest suggestion, any batch
    batches: list = field_(default_factory=list)   # list of BatchResult
    batch_constrained: np.ndarray = None           # final batch's points
    samples_constrained: np.ndarray = None         # every observed composition
    #: Everything :func:`diagnostic_figure` needs, captured at the end of the
    #: run: the field, the PICIP samples with their stored stage-by-stage
    #: arrays, and the densities.  ``None`` unless ``diagnostic=True``.
    diagnostic: object = None
    error: str = None                   # populated if the PICIP run failed

    def __repr__(self):
        d = 'FAIL' if self.min_distance is None else f'{self.min_distance:.4f}'
        return (f'TrialResult(pf{self.phase_field_index} '
                f'u{self.unknown_index} r{self.repeat} '
                f'batches={len(self.batches)} min_distance={d})')


@dataclass
class DiagnosticData:
    """Everything needed to redraw one trial stage by stage, after the fact."""
    test_field: object                  # TestPhaseField
    samples: list                       # PICIP Sample objects, in order
    individual_densities: list          # per-sample density over omega
    combined_densities: list            # cumulative product after each sample
    batches: list                       # BatchResult list, mirrors TrialResult
    known_molar: list                   # true molar weights, per sample
    mass_true: list                     # true mass weights, per sample
    mass_noisy: list                    # measured (errored) weights, per sample
    unknown_index: int


# ── measurement model ────────────────────────────────────────────────────────

def _mass_from_molar(molar_weights, compositions):
    """Convert molar mixing weights to normalised mass weights."""
    M = np.array([c.weight for c in compositions], dtype=float)
    m = M * np.asarray(molar_weights, dtype=float)
    s = m.sum()
    return np.zeros_like(m) if s <= 0 else m / s


def _molar_from_mass(mass_weights, compositions):
    """Inverse of :func:`_mass_from_molar` — mass fractions back to molar."""
    m = np.asarray(mass_weights, dtype=float) / np.array(
        [c.weight for c in compositions], dtype=float)
    s = m.sum()
    return np.zeros_like(m) if s <= 0 else m / s


def _dirichlet_noise(mass_true, experimental_error, rng, floor=1e-3):
    """
    Perturb true mass weights by a Dirichlet whose concentration is set by
    ``experimental_error`` (smaller error → tighter → less noise), matching
    PICIP's own ``alpha = weights / predicted_error`` convention.
    """
    p = np.maximum(np.asarray(mass_true, dtype=float), floor)
    p = p / p.sum()
    return rng.dirichlet(p / max(experimental_error, 1e-9))


def _simplex_volume(pts):
    """Volume of a (dim+1)-vertex simplex given as a (dim+1, dim) array."""
    M = np.asarray(pts[1:], dtype=float) - np.asarray(pts[0], dtype=float)
    return abs(np.linalg.det(M)) / math.factorial(len(M))


def _sample_forming_unknown(tpf, unknown_idx, rng, max_tries=25):
    """
    Draw a random composition guaranteed to form the phase ``unknown_idx``.

    Picks a triangulation simplex that has the unknown as a vertex (weighted by
    volume), samples a strictly-interior point, and confirms via the fixed
    triangulation that the unknown is among the forming phases.  Returns
    ``(sample_c, FormingPhases)`` or ``(None, None)`` if no such simplex exists.
    """
    tri = tpf.triangulation
    simplices = [i for i, s in enumerate(tri.simplices) if unknown_idx in s]
    if not simplices:
        return None, None
    vols = np.array([_simplex_volume(tpf.points_constrained[tri.simplices[i]])
                     for i in simplices])
    vols = vols / vols.sum() if vols.sum() > 0 else None
    for _ in range(max_tries):
        s_idx = simplices[rng.choice(len(simplices), p=vols)]
        verts = tri.simplices[s_idx]
        w = rng.dirichlet(np.ones(tpf.dim + 1))
        sample_c = w @ tpf.points_constrained[verts]
        forming = tpf.find_forming_phases(sample_c)
        if forming is not None and unknown_idx in forming.indices:
            return sample_c, forming
    return None, None


def _observe(tpf, unknown_idx, sample_c, experimental_error, predicted_error,
             rng, name):
    """
    Turn one composition into a PICIP :class:`~picip.picip.Sample`.

    Reads off the phases that form at ``sample_c`` via the fixed triangulation,
    hides ``unknown_idx``, converts the remaining phases' molar mixing weights
    to mass weights, and corrupts them with Dirichlet measurement error — the
    same information a real experiment would hand over.

    Returns ``(sample, info)``, or ``(None, reason)`` when the composition does
    not produce the unknown phase and therefore teaches PICIP nothing about it.
    """
    pf = tpf.phase_field
    forming = tpf.find_forming_phases(sample_c)
    if forming is None:
        return None, 'composition lies outside the phase field'
    if unknown_idx not in forming.indices:
        return None, 'composition does not produce the unknown phase'

    order = list(forming.indices)
    keep = [i for i in range(len(order)) if order[i] != unknown_idx]
    k_idx = [order[i] for i in keep]
    molar = forming.weights[keep]
    molar = molar / molar.sum()
    comps = list(pf.convert_standard_to_pymatgen(tpf.points_standard[k_idx]))

    mass_true = _mass_from_molar(molar, comps)
    mass_noisy = _dirichlet_noise(mass_true, experimental_error, rng)

    sample_s = pf.convert_to_standard_basis(np.asarray(sample_c)[np.newaxis])[0]
    s = Sample(name, pf.convert_standard_to_pymatgen(sample_s))
    s.add_knowns([c.formula for c in comps])
    s.add_mass_weights(mass_noisy.tolist())
    s.set_predicted_error(predicted_error)

    return s, dict(known_indices=k_idx,
                   known_formulas=[tpf.point_formulas[i] for i in k_idx],
                   known_comps=comps, known_molar=molar,
                   mass_true=mass_true, mass_noisy=mass_noisy)


def _suggest(picip, prediction, n, method):
    """Draw a batch of ``n`` suggestions from a prediction."""
    if method == 'kmedoids':
        # past_samples=[] asks for the batch that best covers the posterior on
        # its own, rather than one chosen to complement the existing samples.
        return picip.suggest_kmedoids(prediction, n=n, past_samples=[])
    if method in ('iid', 'suggest'):
        return picip.suggest(prediction, n=n)
    raise ValueError(f"unknown batch_method {method!r}; "
                     f"choose 'kmedoids' or 'iid'")


def _run_one_trial(tpf, pf_index, unknown_idx, repeat, *, version, batch_size,
                   n_batches, experimental_error, predicted_error, batch_method,
                   n_l, n_p, diagnostic, rng):
    """
    Hide one phase, seed PICIP with a composition that forms it, and run
    ``n_batches`` rounds of suggestion.

    Round 0 runs on the seed composition alone.  Each later round observes every
    suggestion from the previous batch — discarding any that turn out not to
    produce the unknown phase — adds them as samples, and re-runs PICIP on the
    product of all densities so far.
    """
    pf = tpf.phase_field
    pred_err = experimental_error if predicted_error is None else predicted_error
    unknown_c = tpf.points_constrained[unknown_idx]
    base = dict(
        phase_field_index=pf_index, unknown_index=unknown_idx,
        unknown_formula=tpf.point_formulas[unknown_idx],
        unknown_constrained=unknown_c, repeat=repeat, version=version,
        batch_method=batch_method, experimental_error=experimental_error,
        predicted_error=pred_err, batch_size=batch_size, n_batches=n_batches)

    seed_c, _ = _sample_forming_unknown(tpf, unknown_idx, rng)
    if seed_c is None:
        return TrialResult(**base, n_knowns=0, known_formulas=[],
                           sample_constrained=unknown_c,
                           error='no simplex produces this phase')

    picip = PICIP(pf)
    samples, densities, infos = [], [], []
    batches, observed = [], [np.asarray(seed_c)]
    pending = [np.asarray(seed_c)]           # compositions still to be observed
    result = None

    try:
        for b in range(n_batches):
            # 1. observe every pending composition and add it to PICIP
            skipped = 0
            for c in pending:
                s, info = _observe(tpf, unknown_idx, c, experimental_error,
                                   pred_err, rng, f'pf{pf_index}_b{b}_'
                                                  f'n{len(samples)}')
                if s is None:
                    skipped += 1
                    continue
                picip.add_sample(s)
                densities.append(picip.run(sample_indexes=len(picip.samples) - 1,
                                           version=version, n_l=n_l,
                                           n_p=n_p).prob_density)
                samples.append(s)
                infos.append(info)

            if not samples:
                # Nothing observed so far produced the unknown phase, so there is
                # no density to suggest from.
                if result is None:
                    result = TrialResult(
                        **base, n_knowns=0, known_formulas=[],
                        sample_constrained=np.asarray(seed_c),
                        error='no observed composition produced the unknown phase')
                break

            # 2. combine every round so far (product of experts) and suggest
            p_comb = np.prod(np.array(densities), axis=0)
            if p_comb.sum() <= 0:
                raise ZeroPException()
            p_comb = p_comb / p_comb.sum()

            prediction = Prediction(f'combined_b{b}', p_comb, list(samples),
                                    individual_densities=list(densities))
            sugg = _suggest(picip, prediction, batch_size, batch_method)
            batch = np.asarray(sugg.constrained)

            batches.append(BatchResult(
                index=b, points_constrained=batch,
                min_distance=float(
                    np.linalg.norm(batch - unknown_c, axis=1).min()),
                n_samples=len(samples), n_skipped=skipped))

            pending = [np.asarray(p) for p in batch]
            observed.extend(pending)
    except (Exception, ZeroPException) as exc:
        if not batches:
            return TrialResult(**base, n_knowns=0, known_formulas=[],
                               sample_constrained=np.asarray(seed_c),
                               error=f'{type(exc).__name__}: {exc}')
        # Partial success: keep the batches that completed and record why the
        # rest did not, rather than discarding usable rounds.
        base['error'] = f'{type(exc).__name__}: {exc}'

    if result is not None:
        return result
    if not batches:
        return TrialResult(**base, n_knowns=0, known_formulas=[],
                           sample_constrained=np.asarray(seed_c),
                           error=base.pop('error', 'no batch completed'))

    err = base.pop('error', None)
    trial = TrialResult(
        **base, n_knowns=len(infos[0]['known_formulas']),
        known_formulas=infos[0]['known_formulas'],
        sample_constrained=np.asarray(seed_c),
        min_distance=batches[-1].min_distance,
        best_distance=min(bb.min_distance for bb in batches),
        batches=batches, batch_constrained=batches[-1].points_constrained,
        samples_constrained=np.array([s.comp_constrained for s in samples]),
        error=err)

    if diagnostic:
        cumulative, running = [], None
        for d in densities:
            running = d if running is None else running * d
            tot = running.sum()
            cumulative.append(running / tot if tot > 0 else running)
        trial.diagnostic = DiagnosticData(
            test_field=tpf, samples=samples, individual_densities=densities,
            combined_densities=cumulative, batches=batches,
            known_molar=[i['known_molar'] for i in infos],
            mass_true=[i['mass_true'] for i in infos],
            mass_noisy=[i['mass_noisy'] for i in infos],
            unknown_index=unknown_idx)
    return trial


def _trial_rng(base_seed, pf_index, unknown_idx, repeat):
    """Deterministic, config-independent RNG for one trial.

    Seeding from the trial's identity (not a shared sequential stream) makes the
    result independent of execution order, so serial and parallel runs are
    identical and the same (field, unknown, repeat) is drawn identically across
    every method / predicted_error configuration.
    """
    return np.random.default_rng([int(base_seed), pf_index, unknown_idx, repeat])


# Worker-process state, populated once per process by the pool initializer so
# the (potentially large) field list is pickled once per worker, not per task.
_WORKER = {}


def _worker_init(fields, base_seed, cfg):
    _WORKER['fields'] = fields
    _WORKER['base_seed'] = base_seed
    _WORKER['cfg'] = cfg


def _worker_trial(task):
    pf_index, unknown_idx, repeat = task
    return _run_one_trial(_WORKER['fields'][pf_index], pf_index, unknown_idx,
                          repeat, rng=_trial_rng(_WORKER['base_seed'], pf_index,
                                                 unknown_idx, repeat),
                          **_WORKER['cfg'])


def evaluate(fields, *, version='dirichlet', batch_size=5, n_batches=1,
             experimental_error=0.1, predicted_error=None,
             batch_method='kmedoids', num_repeats=3, n_l=50, n_p=50,
             diagnostic=False, n_jobs=1, seed=None):
    """
    Hide each internal phase in turn and score PICIP's recovery of it.

    For every phase field, every internal phase is treated as the unknown, and
    ``num_repeats`` independent trials are run for each.  A trial draws a random
    composition that produces the unknown, reads off the phases that form and
    their ratios, corrupts the observed weights with Dirichlet measurement
    error, hands them to PICIP, and asks for a batch of ``batch_size``
    suggestions.  The score is the distance from the closest suggestion to the
    true position of the unknown phase.

    With ``n_batches=1`` this is a **one-shot** run: PICIP sees one composition
    and proposes once.  With ``n_batches > 1`` it becomes an **iterative** run:
    every suggestion in a batch is observed in turn, added as a further sample,
    and the next batch is drawn from the product of all densities so far — so
    PICIP chooses where to look next and the loop follows it.  Suggestions that
    turn out not to produce the unknown phase teach nothing about it and are
    discarded, counted in ``BatchResult.n_skipped``.

    Parameters
    ----------
    fields : TestPhaseField or list of TestPhaseField
        Field(s) from :func:`generate_test_phase_field` /
        :func:`generate_phase_field_set`.
    version : {'dirichlet', 'gaussian'}, default 'dirichlet'
        PICIP inference method — Dirichlet simplex vs. legacy Gaussian.
    batch_size : int, default 5
        Number of suggestions per batch.
    n_batches : int, default 1
        Rounds of suggestion.  ``1`` is a one-shot run; ``> 1`` is iterative.
    experimental_error : float, default 0.1
        Dirichlet error applied to the observed weights (smaller → tighter
        measurement).  Used as PICIP's ``predicted_error`` unless the latter is
        given explicitly.
    predicted_error : float, optional
        σ handed to PICIP.  Defaults to ``experimental_error`` (the
        well-calibrated case); set it independently to study mis-calibration.
    batch_method : {'kmedoids', 'iid'}, default 'kmedoids'
        Batch-selection strategy.  ``'kmedoids'`` minimises the posterior-
        weighted distance to the closest batch point; ``'iid'`` returns the
        posterior mean plus independent draws.
    num_repeats : int, default 3
        Trials per (phase field, unknown phase).
    n_l, n_p : int, default 50
        PICIP discretisation resolution passed through to ``PICIP.run``.
    diagnostic : bool, default False
        Retain the per-sample stage-by-stage arrays needed by
        :func:`diagnostic_figure`.  Off by default because it keeps every
        sample's projected ray points in memory.
    n_jobs : int, default 1
        Worker processes.  ``1`` runs serially; ``-1`` uses all CPU cores.
        Results are identical regardless of ``n_jobs``, since each trial seeds
        its own RNG from its identity.
    seed : int, optional
        Base RNG seed for reproducible sampling and measurement noise.

    Returns
    -------
    list of TrialResult
        In trial order: field-major, then unknown phase, then repeat.  Trials
        that failed carry ``min_distance=None`` and a populated ``error``.

    Examples
    --------
    >>> field = generate_test_phase_field(dim=2, n_internal=3, seed=0)
    >>> results = evaluate(field, batch_size=5, num_repeats=2, seed=0)
    >>> scored = [r.min_distance for r in results if r.min_distance is not None]

    See Also
    --------
    diagnostic_figure : inspect one trial stage by stage.
    distance_vs_batch_figure : plot the closest-hit distance against batch.
    """
    if isinstance(fields, TestPhaseField):
        fields = [fields]
    if n_batches < 1:
        raise ValueError(f"n_batches must be at least 1, got {n_batches}")
    if batch_size < 1:
        raise ValueError(f"batch_size must be at least 1, got {batch_size}")

    base_seed = 0 if seed is None else int(seed)
    cfg = dict(version=version, batch_size=batch_size, n_batches=n_batches,
               experimental_error=experimental_error,
               predicted_error=predicted_error, batch_method=batch_method,
               n_l=n_l, n_p=n_p, diagnostic=diagnostic)

    tasks = [(pf_index, unknown_idx, repeat)
             for pf_index, tpf in enumerate(fields)
             for unknown_idx in (i for i, lab in enumerate(tpf.point_labels)
                                 if lab == 'internal')
             for repeat in range(num_repeats)]
    if not tasks:
        return []

    if n_jobs is None or n_jobs < 0:
        n_jobs = os.cpu_count() or 1
    n_jobs = max(1, min(n_jobs, len(tasks)))

    if n_jobs == 1:
        return [_run_one_trial(fields[i], i, u, r,
                               rng=_trial_rng(base_seed, i, u, r), **cfg)
                for (i, u, r) in tasks]

    from concurrent.futures import ProcessPoolExecutor
    # Fork workers inherit the parent's BLAS thread setting, so single-threaded
    # workers require the parent to be single-threaded before the first BLAS
    # call.  Importing this module before numpy (or exporting OMP_NUM_THREADS=1)
    # achieves that; this module does so at its top.
    with ProcessPoolExecutor(max_workers=n_jobs, initializer=_worker_init,
                             initargs=(fields, base_seed, cfg)) as ex:
        return list(ex.map(_worker_trial, tasks, chunksize=1))


# ── figures ──────────────────────────────────────────────────────────────────
#
# Every function here returns a plotly Figure and writes nothing.  Use
# save_figure() to write one self-contained HTML file.

#: Marker sizes per dimensionality.  The 2-D plotter works on a larger visual
#: scale than the 3-D one (its own defaults are 15 for points and 5 for
#: densities, against 6 and 3), so one set of sizes cannot serve both.
_SIZES = {
    3: dict(support=4, projected=1.5, density=3, anchor=7, known=9,
            sample=8, prev=7, batch=7, truth=13),
    2: dict(support=8, projected=2.5, density=5, anchor=12, known=15,
            sample=14, prev=12, batch=12, truth=20),
}

_COLOURS = dict(known='#E8A838', sample='#FFFFFF', previous='#9E9E9E',
                truth='#2CA02C', estimated='#D62728', batch='#1F77B4')


def save_figure(fig, path):
    """
    Write a figure to one self-contained HTML file.

    The plotly library is inlined, so the file opens offline with no
    ``plotly.min.js`` sidecar to keep beside it.  This makes each file large
    (~5 MB); that is the trade for a single portable artefact.

    Parameters
    ----------
    fig : plotly.graph_objects.Figure
    path : str
        Destination path.  ``.html`` is appended if absent, and any missing
        parent directories are created.

    Returns
    -------
    str
        The path written.
    """
    if not path.endswith('.html'):
        path += '.html'
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    fig.write_html(path, include_plotlyjs=True)
    return path


def _boundary_point(picip, start, through):
    """Where the ray start→through leaves the phase field."""
    lines = np.array(picip.phase_field._get_constraint_lines())
    cons = {'A': lines[:, :-1], 'b': -1.0 * lines[:, -1]}
    return picip._find_intersection(cons, np.asarray(start), np.asarray(through))


def diagnostic_figure(trial, sample_index=-1, batch_index=-1, title=None):
    """
    Draw one trial stage by stage, as a single figure of toggleable layers.

    Requires the trial to have been run with ``evaluate(..., diagnostic=True)``.

    The layers follow the algorithm in order, so switching them on one at a time
    walks through what PICIP actually did:

    ``Dirichlet support on known simplex``
        The Dirichlet likelihood evaluated over the discretised known simplex —
        the k1-k2 *line* in 2-D, the k1-k2-k3 *triangle* in 3-D.  This is the
        belief about the knowns' mixing ratio, before any projection.
    ``projected p_density``
        The raw ray points that carry that support outward through the sample,
        before interpolation.
    ``interpolated p_density (sN)``
        This sample's density after interpolation onto the omega grid.
        Hidden on load.
    ``combined p_density``
        The product-of-experts density over every sample up to this one.
        Hidden on load.  Absent for a single-sample trial.
    ``true direction`` / ``estimated direction``
        Lines from the average known through the sample to the boundary, built
        from the noiseless molar weights and from the measured (errored) weights
        respectively.  The true line passes exactly through the unknown phase, so
        the gap between the two lines *is* the measurement error, made visible.
    ``batch (bN)``
        The suggestions PICIP returned.
    ``unknown (truth)``
        Where the unknown phase actually is.

    Parameters
    ----------
    trial : TrialResult
        A trial from ``evaluate(..., diagnostic=True)``.
    sample_index : int, default -1
        Which observed sample to draw the stage layers for.  The default is the
        most recent.  Earlier samples remain visible as ``previous samples``.
    batch_index : int, default -1
        Which batch of suggestions to draw.
    title : str, optional
        Figure title.  A descriptive default is generated if omitted.

    Returns
    -------
    plotly.graph_objects.Figure

    Raises
    ------
    ValueError
        If the trial carries no diagnostic data.
    """
    d = trial.diagnostic
    if d is None:
        raise ValueError(
            'this trial carries no diagnostic data — re-run with '
            'evaluate(..., diagnostic=True)')

    tpf = d.test_field
    pf = tpf.phase_field
    dim = pf.constrained_dim
    sz = _SIZES[dim]
    n = range(len(d.samples))[sample_index]     # normalise negative indices
    s = d.samples[n]
    pl = make_plotter(pf)
    picip = PICIP(pf)

    unknown_c = tpf.points_constrained[d.unknown_index]
    knowns_c = np.asarray(s.knowns_constrained)
    sample_c = np.asarray(s.comp_constrained)

    # ── stage 1: Dirichlet support over the discretised known simplex ───────
    pts, pdf = np.asarray(s._simplex_points), np.asarray(s._simplex_pdf)
    pl.plot_p(pdf, omega=pts, s=sz['support'],
              name='Dirichlet support on known simplex')

    # ── stage 2: the projection rays, before interpolation ──────────────────
    # _project_known_simplex lays the ray points out ray-major, n_p per ray,
    # then appends anchor points.  The emitted ray count is not the number of
    # simplex points: the 3-D path skips rays starting on the boundary, and the
    # 2-D path adds extra corner-derived ones — hence _n_rays_emitted.
    ppts = np.asarray(s._projected_points)
    pval = np.asarray(s._projected_p)
    block = (s._n_rays_emitted or 0) * (s._n_p or 0)
    if block and block <= len(ppts):
        pl.plot_p(pval[:block], omega=ppts[:block], s=sz['projected'],
                  name='projected p_density', opacity=0.30)

    # ── stage 3: densities on the omega grid, hidden on load ────────────────
    pl.plot_p(d.individual_densities[n], s=sz['density'],
              name=f'interpolated p_density (s{n + 1})', visible='legendonly')
    if len(d.samples) > 1:
        tag = '+'.join(f's{i + 1}' for i in range(n + 1))
        pl.plot_p(d.combined_densities[n], s=sz['density'],
                  name=f'combined p_density ({tag})', visible='legendonly')

    # ── stage 4: the direction the measured weights imply ───────────────────
    # sample = w_u * unknown + (1 - w_u) * c, where c is the average known on
    # the simplex, so the ray c → sample passes through the unknown exactly when
    # c is built from the noiseless molar weights.
    comps = list(pf.convert_standard_to_pymatgen(
        np.asarray(s.knowns_standard)))
    for weights, colour, label in (
            (d.known_molar[n], _COLOURS['truth'], 'true'),
            (_molar_from_mass(d.mass_noisy[n], comps), _COLOURS['estimated'],
             'estimated')):
        c_pt = np.asarray(weights) @ knowns_c
        end = _boundary_point(picip, c_pt, sample_c)
        if end is not None:
            pl.plot_line(c_pt, end, c=colour, w=5, name=f'{label} direction')
        pl.plot_points(np.array([c_pt]), color=colour,
                       name=f'{label} average known',
                       custom_labels=[f'<b>{label} average known</b>'],
                       s=sz['anchor'], border_color='#222222', border_width=2)

    # ── context: knowns, samples, batch, truth ──────────────────────────────
    pl.plot_points(knowns_c, color=_COLOURS['known'], name='known phases',
                   custom_labels=[f'<b>known</b><br>{f}'
                                  for f in trial.known_formulas],
                   marker='diamond', border_color='#222222',
                   border_width=2, s=sz['known'])

    if n > 0:
        prev = np.array([np.asarray(x.comp_constrained) for x in d.samples[:n]])
        pl.plot_points(prev, color=_COLOURS['previous'],
                       name=f'previous samples (s1..s{n})',
                       custom_labels=[f'<b>sample s{i + 1}</b>'
                                      for i in range(n)],
                       border_color='#222222', border_width=2, s=sz['prev'])

    pl.plot_points(np.array([sample_c]), color=_COLOURS['sample'],
                   name=f'sample (s{n + 1})',
                   custom_labels=[f'<b>sample s{n + 1}</b>'],
                   border_color='#222222', border_width=2, s=sz['sample'])

    if d.batches:
        b = range(len(d.batches))[batch_index]
        batch = np.asarray(d.batches[b].points_constrained)
        pl.plot_points(batch, color=_COLOURS['batch'],
                       name=f'batch (b{b + 1}, closest '
                            f'{d.batches[b].min_distance:.4f})',
                       custom_labels=[f'<b>batch {b + 1}.{i + 1}</b>'
                                      for i in range(len(batch))],
                       border_color='#222222', border_width=2, s=sz['batch'])

    pl.plot_points(np.array([unknown_c]), color=_COLOURS['truth'],
                   name='unknown (truth)',
                   custom_labels=[f'<b>unknown (truth)</b><br>'
                                  f'{trial.unknown_formula}'],
                   marker='diamond-open', border_color='#222222',
                   border_width=2, s=sz['truth'])

    if title is None:
        title = (f'{dim}-D {tpf.structure} — hiding {trial.unknown_formula}; '
                 f'sample s{n + 1} of {len(d.samples)}, '
                 f'version={trial.version}, error={trial.experimental_error}. '
                 f'Densities hidden — toggle in legend.')
    fig = pl.show(show=False, return_fig=True)
    fig.update_layout(title=dict(text=title))
    return fig


#: Qualitative palette for comparing several runs on one set of axes.
_SERIES_COLOURS = ['#1F77B4', '#D62728', '#2CA02C', '#E8A838',
                   '#9467BD', '#17BECF']


def _as_series(results):
    """Normalise the figure input to an ordered ``{label: results}`` mapping."""
    if isinstance(results, dict):
        return dict(results)
    return {'trials': results}


def distance_vs_batch_figure(results, title=None, show_trials=True,
                             colour=None):
    """
    Plot closest-hit distance against batch number.

    The closest-hit distance is the distance from the true unknown phase to the
    nearest point in a batch of suggestions — i.e. how close the best
    composition in that batch came to the answer.

    The figure adapts to the kind of run it is given:

    * **One-shot runs** (``n_batches=1``) have only one batch, so there is
      nothing to plot against.  The figure is a **box plot** of the per-trial
      distances instead, showing median, quartiles and outliers.
    * **Iterative runs** (``n_batches > 1``) get a line of the mean against
      batch number with a ±1 SE band.  A falling curve means feeding results
      back is paying for itself.  Every trial contributes one value per batch,
      so the comparison across batches is paired within a trial.

    Pass a ``dict`` to put several runs on the same axes — side-by-side boxes
    for one-shot runs, overlaid lines for iterative ones.

    Parameters
    ----------
    results : list of TrialResult, or dict of {label: list of TrialResult}
        Output of :func:`evaluate`.  Trials that failed are ignored.  A dict
        draws one series per entry, labelled by its key.
    title : str, optional
        Figure title.  A descriptive default is generated if omitted.
    show_trials : bool, default True
        Iterative runs only: draw one faint line per trial behind the mean.
        Turn off for large runs.  Box plots always show their own points.
    colour : str or list of str, optional
        Colour per series, in order.  Defaults to a built-in palette.

    Returns
    -------
    plotly.graph_objects.Figure

    Raises
    ------
    ValueError
        If no trial produced a scored batch.

    Examples
    --------
    >>> fig = distance_vs_batch_figure({'dirichlet': d_runs,
    ...                                 'gaussian': g_runs})
    """
    import plotly.graph_objects as go

    series = _as_series(results)
    curves = {label: [[b.min_distance for b in r.batches]
                      for r in runs if r.batches]
              for label, runs in series.items()}
    curves = {k: v for k, v in curves.items() if v}
    if not curves:
        raise ValueError('no trial produced a scored batch')

    if colour is None:
        colours = _SERIES_COLOURS
    elif isinstance(colour, str):
        colours = [colour]
    else:
        colours = list(colour)
    colours = [colours[i % len(colours)] for i in range(len(curves))]

    n_max = max(len(c) for cs in curves.values() for c in cs)
    fig = go.Figure()

    if n_max == 1:
        # ── one-shot: a box plot per series, on shared axes ─────────────────
        for (label, cs), col in zip(curves.items(), colours):
            vals = [c[0] for c in cs]
            fig.add_trace(go.Box(
                y=vals, name=f'{label} (n={len(vals)})',
                marker=dict(color=col, size=6,
                            line=dict(color='#222222', width=1)),
                line=dict(color=col, width=2),
                fillcolor='rgba(0,0,0,0)',
                boxpoints='all',      # show every trial, not just the outliers
                jitter=0.5, pointpos=0, boxmean=True,
                hovertemplate='closest hit %{y:.4f}<extra></extra>'))
        xaxis = dict(showgrid=False, zeroline=False)
    else:
        # ── iterative: mean vs batch number, one line per series ────────────
        for (label, cs), col in zip(curves.items(), colours):
            if show_trials and len(cs) > 1:
                for c in cs:
                    fig.add_trace(go.Scatter(
                        x=list(range(1, len(c) + 1)), y=c, mode='lines',
                        line=dict(color='rgba(150,150,150,0.30)', width=1),
                        hoverinfo='skip', showlegend=False))

            n_b = max(len(c) for c in cs)
            xs = list(range(1, n_b + 1))
            mean, se, counts = [], [], []
            for b in range(n_b):
                colv = np.array([c[b] for c in cs if len(c) > b])
                counts.append(len(colv))
                mean.append(float(colv.mean()))
                se.append(float(colv.std(ddof=1) / np.sqrt(len(colv)))
                          if len(colv) > 1 else 0.0)
            mean, se = np.array(mean), np.array(se)

            if np.any(se > 0):
                rgb = tuple(int(col.lstrip('#')[i:i + 2], 16)
                            for i in (0, 2, 4))
                fig.add_trace(go.Scatter(
                    x=xs + xs[::-1],
                    y=list(mean + se) + list((mean - se)[::-1]),
                    fill='toself',
                    fillcolor=f'rgba({rgb[0]},{rgb[1]},{rgb[2]},0.18)',
                    line=dict(color='rgba(0,0,0,0)'), hoverinfo='skip',
                    name=f'{label} ± 1 SE', showlegend=len(curves) == 1))
            fig.add_trace(go.Scatter(
                x=xs, y=mean, mode='lines+markers',
                line=dict(color=col, width=3),
                marker=dict(size=9, line=dict(color='#222222', width=1)),
                customdata=np.stack([counts, se], axis=-1),
                name=f'{label} (n={len(cs)})',
                hovertemplate='batch %{x}<br>mean %{y:.4f}'
                              '<br>SE %{customdata[1]:.4f}'
                              '<br>n=%{customdata[0]}<extra></extra>'))
        xaxis = dict(title='batch number', dtick=1, tickmode='linear',
                     showgrid=True, gridcolor='rgba(0,0,0,0.08)',
                     zeroline=False)

    if title is None:
        n_tot = sum(len(cs) for cs in curves.values())
        title = ('Closest hit — one-shot, {n} trials'.format(n=n_tot)
                 if n_max == 1 else
                 f'Closest hit vs batch number — {n_tot} trials')
    fig.update_layout(
        title=dict(text=title),
        xaxis=xaxis,
        yaxis=dict(title='distance from unknown phase to closest suggestion',
                   rangemode='tozero', showgrid=True,
                   gridcolor='rgba(0,0,0,0.08)', zeroline=False),
        font=dict(family='Arial', size=13),
        hoverlabel=dict(bgcolor='rgba(25,25,25,0.88)', font_size=13,
                        font_color='white', bordercolor='rgba(0,0,0,0)'),
        plot_bgcolor='white',
        showlegend=True,
        legend=dict(bgcolor='rgba(255,255,255,0.92)', bordercolor='#CCCCCC',
                    borderwidth=1))
    return fig


def summarise(results):
    """
    Reduce a list of trials to a short text summary.

    Reports how many trials scored, the closest-hit distance at the first and
    final batch, and any failure reasons — the numbers worth reading before
    opening a figure.

    Parameters
    ----------
    results : list of TrialResult

    Returns
    -------
    str
    """
    from collections import Counter

    scored = [r for r in results if r.min_distance is not None]
    lines = [f'{len(results)} trials, {len(scored)} scored']
    if not scored:
        return '\n'.join(lines + ['  no trial produced a scored batch'])

    n_max = max(len(r.batches) for r in scored)
    lines.append(f'  batch size {scored[0].batch_size}, '
                 f'{n_max} batch(es) per trial')
    for b in range(n_max):
        col = np.array([r.batches[b].min_distance for r in scored
                        if len(r.batches) > b])
        se = col.std(ddof=1) / np.sqrt(len(col)) if len(col) > 1 else 0.0
        lines.append(f'  batch {b + 1}: closest hit mean {col.mean():.4f} '
                     f'+/- {se:.4f}  median {np.median(col):.4f}  n={len(col)}')

    skipped = sum(b.n_skipped for r in scored for b in r.batches)
    if skipped:
        lines.append(f'  {skipped} suggestion(s) did not produce the hidden '
                     f'phase and were discarded')
    failures = Counter(r.error for r in results if r.error)
    for reason, k in failures.most_common():
        lines.append(f'  {k} x {reason}')
    return '\n'.join(lines)


# ── self-test / demo ─────────────────────────────────────────────────────────

if __name__ == '__main__':
    import tempfile

    print('1. generate a 2-D field')
    field = generate_test_phase_field(dim=2, structure='2c2a', n_internal=3,
                                      n_edge=2, resolution=30, seed=3)
    print(field.summary())

    print('\n2a. one-shot run (n_batches=1)')
    one_shot = evaluate(field, batch_size=5, num_repeats=2, seed=0)
    print(summarise(one_shot))

    print('\n2b. iterative run (n_batches=4)')
    iterative = evaluate(field, batch_size=3, n_batches=4, num_repeats=2,
                         diagnostic=True, seed=0)
    print(summarise(iterative))

    out = tempfile.mkdtemp(prefix='picip_eval_')
    print(f'\n3. diagnostic figure  -> {save_figure(diagnostic_figure(iterative[0]), os.path.join(out, "diagnostic"))}')
    print(f'4. convergence figure -> {save_figure(distance_vs_batch_figure(iterative), os.path.join(out, "convergence"))}')
