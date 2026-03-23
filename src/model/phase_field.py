import numpy as np
import multiprocessing as mp
import random
import pandas as pd
from pymatgen.core import Composition
from scipy.optimize import nnls
from scipy.spatial import cKDTree
from scipy.stats import dirichlet, multivariate_normal
from scipy.spatial import ConvexHull, Delaunay
from scipy.interpolate import griddata 
from itertools import combinations
from .exceptions import ZeroPException
import matplotlib.pyplot as plt
from math import dist as deu
import time
import os

class Phase_Field:

    def _init(self):
        """Initialise empty element/dimension bookkeeping before a setup_* call populates the field."""
        self.elements = None #The elements in the phase field
        self.nelements = None #The number of elements in the phase field

    def setup_charged(self, species_dict, n_points=None, resolution=None, precalculated_basis=None, print_basis=False, precursors=None):
        """
        Set up the phase field for a system with fixed formal charges on each species.

        Use this entry point when every element has a single, well-defined oxidation state
        (e.g. ``{'Cu': 2, 'S': -2}``).  The charge-neutrality constraint reduces
        ``constrained_dim`` by one relative to the number of elements.  For mixed-valence
        species with a *range* of allowed charges use :meth:`setup_charge_ranges` instead.

        Parameters
        ----------
        species_dict : dict
            Mapping of element symbol (str) to its formal charge (int or float).
            Example: ``{'Mg': 2, 'Al': 3, 'S': -2}``.
        n_points : int, optional
            Target number of grid points spanning the phase field.  When omitted the
            default is 10 000 for 2-D fields and 6 000 for 3-D fields.  Takes
            precedence over ``resolution`` if both are supplied.
        resolution : int, optional
            Grid spacing expressed as ``1/resolution`` in each constrained-basis
            direction.  Ignored when ``n_points`` is supplied.
        precalculated_basis : ndarray of shape (constrained_dim, nelements), optional
            Pre-computed orthonormal basis for the constrained subspace.  Supply this
            to ensure reproducible grid layouts across runs; if omitted a random
            basis is generated each time.  Use ``print_basis=True`` once to capture
            the basis and pass it back as this argument.
        print_basis : bool, default False
            Print the basis vectors to stdout so they can be saved and reused as
            ``precalculated_basis`` in future runs.
        precursors : list of str, optional
            Composition strings for the precursor phases (e.g. ``['MgS', 'Al2S3']``).
            Two effects are possible depending on the geometry:

            * **Grid restriction** — if the precursors span the full constrained
              dimensionality, ``omega`` is trimmed to the convex hull of the
              precursor set.
            * **Dimension reduction** — if the precursors all share a zero
              coordinate (e.g. Cu=0 in every precursor), the effective
              ``constrained_dim`` is reduced automatically and the
              :class:`~visualise_square.Square` or :class:`~visualise_cube.Cube`
              class for the *new* dimension should be used for plotting.

        Returns
        -------
        None
        """
        charge_vector = []
        elements = []
        for key, value in species_dict.items():
            charge_vector.append(value)
            elements.append(key)
        self.elements = elements
        self.nelements = len(elements)
        self._setup(charge_vector, n_points=n_points, resolution=resolution, precalculated_basis=precalculated_basis, print_basis=print_basis, precursors=precursors)

    def setup_uncharged(self, species_list, n_points=None, resolution=None, precalculated_basis=None, print_basis=False, precursors=None):
        """
        Set up the phase field for a system with no charge constraints.

        Use this entry point when you only care about compositional mixing without
        any charge-neutrality requirement (e.g. alloy or oxide phase fields where
        charges are not tracked).  The only equality constraint is that element
        fractions sum to one, so ``constrained_dim = len(species_list) - 1``.
        For charged systems use :meth:`setup_charged` or :meth:`setup_charge_ranges`.

        Parameters
        ----------
        species_list : list of str
            Element symbols in the phase field, e.g. ``['Mg', 'Al', 'Sn', 'S']``.
        n_points : int, optional
            Target number of grid points spanning the phase field.  When omitted the
            default is 10 000 for 2-D fields and 6 000 for 3-D fields.  Takes
            precedence over ``resolution`` if both are supplied.
        resolution : int, optional
            Grid spacing expressed as ``1/resolution`` in each constrained-basis
            direction.  Ignored when ``n_points`` is supplied.
        precalculated_basis : ndarray of shape (constrained_dim, nelements), optional
            Pre-computed orthonormal basis for the constrained subspace.  Supply this
            to ensure reproducible grid layouts across runs; if omitted a random
            basis is generated each time.  Use ``print_basis=True`` once to capture
            the basis and pass it back as this argument.
        print_basis : bool, default False
            Print the basis vectors to stdout so they can be saved and reused as
            ``precalculated_basis`` in future runs.
        precursors : list of str, optional
            Composition strings for the precursor phases (e.g. ``['MgS', 'Al2S3']``).
            Two effects are possible depending on the geometry:

            * **Grid restriction** — if the precursors span the full constrained
              dimensionality, ``omega`` is trimmed to the convex hull of the
              precursor set.
            * **Dimension reduction** — if the precursors all share a zero
              coordinate for some element, the effective ``constrained_dim`` is
              reduced automatically and the plotter class for the *new* dimension
              should be used.

        Returns
        -------
        None
        """
        self.elements = species_list
        self.nelements = len(self.elements)
        self._setup(n_points=n_points, resolution=resolution, precalculated_basis=precalculated_basis, precursors=precursors, print_basis=print_basis)

    def setup_charge_ranges(self, species_dict, n_points=None, resolution=None, precalculated_basis=None, print_basis=False, precursors=None):
        """
        Set up the phase field for a system where one or more species has a range of allowed charges.

        Use this entry point for mixed-valence systems, e.g. Fe which can be +2 or +3.
        Elements with a fixed charge are given as a single int or float; elements with
        a variable charge are given as a two-element list ``[min_charge, max_charge]``.
        The grid is constructed without a charge-equality constraint (same as
        :meth:`setup_uncharged`) and then filtered post-hoc to retain only compositions
        whose total charge can fall within the allowed range for each species.

        Parameters
        ----------
        species_dict : dict
            Mapping of element symbol (str) to its charge specification.  Fixed
            charges are a scalar (int or float); variable charges are a list
            ``[min_charge, max_charge]``.
            Example: ``{'Cu': [1, 2], 'Zn': 2, 'S': -2}``.
        n_points : int, optional
            Target number of grid points spanning the phase field.  When omitted the
            default is 10 000 for 2-D fields and 6 000 for 3-D fields.  Takes
            precedence over ``resolution`` if both are supplied.
        resolution : int, optional
            Grid spacing expressed as ``1/resolution`` in each constrained-basis
            direction.  Ignored when ``n_points`` is supplied.
        precalculated_basis : ndarray of shape (constrained_dim, nelements), optional
            Pre-computed orthonormal basis for the constrained subspace.  Supply this
            to ensure reproducible grid layouts across runs; if omitted a random
            basis is generated each time.
        print_basis : bool, default False
            Print the basis vectors to stdout so they can be saved and reused as
            ``precalculated_basis`` in future runs.
        precursors : list of str, optional
            Composition strings for the precursor phases.  See :meth:`setup_uncharged`
            for a full description of the two possible effects (grid restriction vs
            dimension reduction).

        Returns
        -------
        None
        """
        self.charge_ranges=True
        species_list=species_dict.keys()
        self.species_dict=species_dict
        self.setup_uncharged(species_list, n_points=n_points, resolution=resolution, precalculated_basis=precalculated_basis, precursors=precursors, print_basis=print_basis)

    def setup_custom(self, species, custom_con, n_points=None, resolution=None, precalculated_basis=None, print_basis=False, precursors=None):
        """
        Set up the phase field with one or more arbitrary additional linear equality constraints.

        Use this entry point when neither a simple charge constraint nor a charge-range
        suffices.  Any number of extra equality constraints of the form ``c · x = 0``
        can be supplied via ``custom_con`` in addition to the implicit sum-to-one
        constraint (and the charge constraint if ``species`` is a dict with scalar
        charges).  Each extra constraint reduces ``constrained_dim`` by one.

        Parameters
        ----------
        species : list of str or dict
            If a **list**, element symbols only — no charge constraint is added
            automatically (only the supplied ``custom_con`` constraints and sum=1).
            If a **dict** mapping element symbol to charge (scalar int/float or list
            ``[min, max]``), a charge-neutrality (or charge-range) constraint is
            added on top of ``custom_con``.
        custom_con : array_like of shape (k, nelements) or (nelements,)
            One or more rows, each being the normal vector ``c`` of an additional
            equality constraint ``c · x = 0``.  Pass a 1-D array for a single
            extra constraint, or a 2-D array for multiple.
        n_points : int, optional
            Target number of grid points spanning the phase field.  When omitted the
            default is 10 000 for 2-D fields and 6 000 for 3-D fields.  Takes
            precedence over ``resolution`` if both are supplied.
        resolution : int, optional
            Grid spacing expressed as ``1/resolution`` in each constrained-basis
            direction.  Ignored when ``n_points`` is supplied.
        precalculated_basis : ndarray of shape (constrained_dim, nelements), optional
            Pre-computed orthonormal basis for the constrained subspace.  Supply this
            to ensure reproducible grid layouts across runs; if omitted a random
            basis is generated each time.
        print_basis : bool, default False
            Print the basis vectors to stdout so they can be saved and reused as
            ``precalculated_basis`` in future runs.
        precursors : list of str, optional
            Composition strings for the precursor phases.  See :meth:`setup_uncharged`
            for a full description of the two possible effects (grid restriction vs
            dimension reduction).

        Returns
        -------
        None

        Raises
        ------
        Exception
            If ``species`` is neither a list nor a dict.
        """

        #check if charges are provided
        if isinstance(species, dict): 
            has_ranges=False
            for value in species.values():
                if isinstance(value, list):
                    has_ranges=True
            if has_ranges:
                self.charge_ranges=True
                self.species_dict=species
                elements = []
                for key in species.keys():
                    elements.append(key)
                self.elements = elements
                self.nelements = len(elements)
                normalb = np.ones(self.nelements)
                normal_vectors = np.vstack((custom_con, normalb))
            else:
                charge_vector = []
                elements = []
                for key, value in species.items():
                    charge_vector.append(value)
                    elements.append(key)
                self.elements = elements
                self.nelements = len(elements)
                normalb = np.ones(self.nelements)
                normala = np.array(charge_vector)
                normal_vectors = np.vstack((normala, custom_con, normalb))
        elif isinstance(species, list):
            self.elements=species
            self.nelements=len(species)
            normalb = np.ones(self.nelements)
            normal_vectors = np.vstack((custom_con, normalb))
        else:
            raise Exception('species must be either list or dict')

        if precursors is not None:
            self.add_precursors_standard(precursors)
        self.normal_vectors = normal_vectors
        self.normal_vectors_b=np.zeros(len(self.normal_vectors))
        self.normal_vectors_b[-1]=1
        self.resolution = None if n_points is not None else resolution
        self._n_points = n_points
        self._create_omega_constrained(precalculated_basis=precalculated_basis,print_basis=print_basis)
        if precursors is not None:
            self.precursors_constrained=self.convert_to_constrained_basis(
                    self.precursors_standard)
            self._constrain_omega_precursors()

    def _setup(self, charge_vector=None, n_points=None, resolution=None, precalculated_basis=None, print_basis=False, precursors=None):

        """
        Creates the grid of points.

        Parameters
        ----------
        charge_vector : array_like of shape (n,), optional
            Formal charge for each of the `n` elements.
        resolution : int, optional, default=30
            Points are spaced `1/resolution` apart in each direction.
        precalculated_basis : ndarray of shape (n, m), optional
            The orthonormal vectors that determine the orientation of the constrained
            space. If not provided, it will be calculated at random.
        print_basis : bool, default=False
            Whether or not to print the orthonormal vectors that are used.
        precursors : List of strings , optional, default=None
            An optinal list of precursors which the phase field is constrained by.

        Returns
        -------
        None
        """
        normalb = np.ones(self.nelements)
        if charge_vector is not None:
            normala = np.array(charge_vector)
            normal_vectors = np.vstack((normala, normalb))
        else:
            normal_vectors = np.array([normalb])
        self.normal_vectors = normal_vectors
        self.normal_vectors_b=np.zeros(len(self.normal_vectors))
        self.resolution = None if n_points is not None else resolution
        self._n_points = n_points
        if precursors is not None:
            self.add_precursors_standard(precursors)
        self._create_omega_constrained(precalculated_basis=precalculated_basis,print_basis=print_basis)
        if precursors is not None:
            self.precursors_constrained=self.convert_to_constrained_basis(
                    self.precursors_standard)
            self._constrain_omega_precursors()

    def add_precursors_standard(self, precursors):
        """
        Parse precursor composition strings and store them in standard-basis coordinates.

        This is called automatically by the ``setup_*`` methods when ``precursors`` is
        supplied.  It only *stores* the precursor compositions; the actual grid
        restriction (or dimension reduction) is applied afterwards by
        :meth:`_constrain_omega_precursors` (called inside ``_setup`` /
        :meth:`setup_custom`).

        If you want to add precursors *after* a field has been set up, call
        :meth:`add_precursors` instead, which also applies the hull filter to
        ``self.omega`` immediately.

        Parameters
        ----------
        precursors : list of str
            Composition strings for the precursor phases, interpreted by pymatgen
            ``Composition``.  Every element in each precursor must appear in
            ``self.elements``.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If a precursor contains an element that was not in the original species list.
        """
        precursors=[Composition(x) for x in precursors]
        for p in precursors:
            for el in p.keys():
                if str(el) not in self.elements:
                    print(self.elements)
                    raise ValueError(f"Precursor {p} contains {el} which was not in provided elements")
        self.precursors_size = [x.num_atoms for x in precursors]
        precursors_standard = []
        precursors_label = []
        for k in precursors:
            precursors_standard.append(
                [k.get_atomic_fraction(e) for e in self.elements]
            )
            precursors_label.append(k.reduced_formula)
        precursors_standard = np.array(precursors_standard)
        self.precursors_standard = precursors_standard


    def _constrain_omega_precursors(self):
        """Filter omega to only points inside the convex hull of precursors_constrained."""
        hull = ConvexHull(self.precursors_constrained)
        A = hull.equations[:, :-1]
        b = hull.equations[:, -1]
        self.omega = self.omega[np.all(np.dot(self.omega, A.T) + b <= 1e-12, axis=1)]
        self.precursor_corners = self.precursors_constrained
        edges = set()
        for simplex in hull.simplices:
            for i in range(len(simplex)):
                for j in range(i + 1, len(simplex)):
                    edges.add(tuple(sorted((simplex[i], simplex[j]))))
        self.precursor_edges = list(edges)


    def _find_orthonormal(self, A):
        """
        Finds a vector orthonormal to the column space of A.

        Parameters
        ----------
        A : ndarray of shape (m, n)
            A matrix for which the returned vector is orthogonal to its columns.

        Returns
        -------
        ndarray
            An orthonormal vector orthogonal to the column space of `A`, with length `m`.
        """
        rand_vec = np.random.rand(A.shape[0], 1)
        A = np.hstack((A, rand_vec), dtype=np.double)
        b = np.zeros(A.shape[1], dtype=np.double)
        b[-1] = 1
        x = np.linalg.lstsq(A.T, b, rcond=None)[0]
        return x / np.linalg.norm(x)

    def _create_omega_constrained(
            self, precalculated_basis=None, print_basis=False, add_corners_to_knowns=False):
        """
        Create `omega`, the grid of points.

        Constructs a finite grid spanning the phase field, using linear equations
        to reduce dimensionality of the grid.

        Parameters
        ----------
        precalculated_basis : ndarray, optional
            Basis to use for constructing the grid. If not provided, it will be computed at random.
        print_basis : bool, optional, default=False
            Whether or not to print the randomly computed basis.

        Returns
        -------
        None
        """

        normal_vectors = self.normal_vectors
        resolution = self.resolution
        dim = normal_vectors.shape[1]
        self.constrained_dim = dim - len(normal_vectors)
        self.precursors_reduced_dimension=False

        if hasattr(self, 'precursors_standard') and self.precursors_standard.size!=0:
            provided_corners = np.array(self.precursors_standard)
            center_precursors = np.mean(provided_corners, axis=0)
            diffs = provided_corners - center_precursors
            rank = np.linalg.matrix_rank(diffs)
            new_constraint_count = dim - rank
            additional_constraints_needed = new_constraint_count - len(normal_vectors)
            if additional_constraints_needed > 0:
                if rank==2:
                    print('Precursors reduced dimension to two: use the Square class for plotting.')
                elif rank==3:
                    print('Precursors reduced dimension to three: use the Cube class for plotting.')
                else:
                    print(f'Precursors reduced dimension to {rank}: plotting not supported.')
                # Prefer unit-vector constraints for elements that are zero in every
                # precursor (e.g. Cu=0) — SVD ordering is ambiguous and can pick the
                # wrong null vector, breaking the vertex LP.
                zero_cols = np.where(np.all(provided_corners < 1e-10, axis=0))[0]
                if len(zero_cols) >= additional_constraints_needed:
                    additional_vectors = np.eye(dim)[zero_cols[:additional_constraints_needed]]
                else:
                    U, S, Vt = np.linalg.svd(diffs, full_matrices=True)
                    svd_null = Vt[rank:]  # shape: (dim-rank, dim)
                    # Prepend any unit-vector constraints, then fill from SVD
                    svd_needed = additional_constraints_needed - len(zero_cols)
                    if len(zero_cols) > 0:
                        additional_vectors = np.vstack(
                            [np.eye(dim)[zero_cols], svd_null[:svd_needed]]
                        )
                    else:
                        additional_vectors = svd_null[:additional_constraints_needed]
                normal_vectors = np.vstack([additional_vectors,normal_vectors])
                self.normal_vectors=normal_vectors
                self.constrained_dim=rank
                self.precursors_reduced_dimension=True
                precalculated_basis=None

        if self.constrained_dim == 1:
            raise ValueError(
                f"The resulting phase field is 1D (constrained_dim=1) for elements "
                f"{self.elements}. PICIP requires at least 2D."
            )

        # Build an orthonormal basis for the constrained subspace.  All bases that
        # span the same subspace are valid; the random construction means omega
        # will look "rotated" in different runs unless precalculated_basis is fixed.
        # Use print_basis=True once, capture the output, and pass it back via
        # precalculated_basis to get reproducible grid layouts.
        if precalculated_basis is None:
            x = np.empty((self.constrained_dim, dim))
            A = normal_vectors.T
            for i in range(self.constrained_dim):
                x[i] = self._find_orthonormal(A)
                A = np.hstack((A, np.array([x[i]]).T))
            self.basis = x
            if print_basis:
                print("Basis used:")
                print("np." + repr(self.basis))
        else:
            self.basis = precalculated_basis

        # Pre-compute charge bounds before vertex-finding so the LP can include them
        _charge_ranges = hasattr(self, 'charge_ranges') and self.charge_ranges
        if _charge_ranges:
            has_range = [isinstance(x, list) for x in self.species_dict.values()]
            ranges_list = [x for x, y in zip(self.species_dict.values(), has_range) if y]
            fixed = [x if not y else 0 for x, y in zip(self.species_dict.values(), has_range)]
            self.charge_max = np.array(fixed, dtype=float)
            self.charge_max[has_range] = np.array(ranges_list)[:, 1]
            self.charge_min = np.array(fixed, dtype=float)
            self.charge_min[has_range] = np.array(ranges_list)[:, 0]

        self._find_vertices_and_edges()
        vertices = self.corners
        center = np.mean(vertices, axis=0)
        center = center / sum(center)
        self.contained_point = center
        vertices = self.convert_to_constrained_basis(vertices)
        center = np.mean(vertices, axis=0)

        n_points = getattr(self, '_n_points', None)
        if resolution is None:
            # Derive a resolution that hits the n_points target by computing the
            # polytope volume and back-solving for the grid spacing.  The fallback
            # of resolution=30 is used when ConvexHull fails (degenerate geometry).
            if n_points is None:
                # Default density: ~10 000 points for 2-D, ~6 000 for 3-D.
                # Increase n_points (or use the resolution parameter directly)
                # for finer-grained inference at the cost of memory and compute.
                n_points = 10000 if self.constrained_dim == 2 else 6000
            try:
                vol = ConvexHull(vertices).volume
                # If precursors restrict the field, scale n_points up so the target
                # is met after the precursor hull filter is applied.
                if hasattr(self, 'precursors_standard') and self.precursors_standard.size != 0:
                    prec_c = self.convert_to_constrained_basis(self.precursors_standard)
                    prec_vol = ConvexHull(prec_c).volume
                    if prec_vol > 0:
                        n_points = round(n_points * vol / prec_vol)
                resolution = max(1, round((n_points / vol) ** (1 / self.constrained_dim)))
            except Exception:
                resolution = 30

        distances = vertices - center
        max_distances = np.max(distances, axis=0)
        min_distances = np.min(distances, axis=0)
        dimensions = len(center)
        line_coefficients = self._get_constraint_lines()

        def _build(res):
            # Build a Cartesian meshgrid at spacing 1/res in the constrained basis,
            # then discard points that violate any positivity constraint (x_i >= 0)
            # or charge-range bounds.
            grids = [
                np.arange(center[i] + min_distances[i], center[i] + max_distances[i], 1/res)
                for i in range(dimensions)
            ]
            w = np.vstack([np.ravel(m) for m in np.meshgrid(*grids)]).T
            for line in line_coefficients:
                w = np.delete(w, np.where(np.einsum("...i,i->...", w, line[:-1]) < -1 * line[-1]), axis=0)
            if _charge_ranges:
                # For charge-range fields the grid is first built uncharged, then
                # filtered here so only charge-neutral-compatible compositions remain.
                w_s = self.convert_to_standard_basis(w, norm=1)
                keep = (
                    (np.einsum('...i,i->...', w_s, self.charge_min) <= 0) &
                    (np.einsum('...i,i->...', w_s, self.charge_max) >= 0)
                )
                w = w[keep]
            return w

        omega = _build(resolution)
        # If the initial resolution undershoots n_points (can happen when the
        # polytope is very small), increment resolution until the target is met.
        if n_points is not None:
            while omega.shape[0] < n_points:
                resolution += 1
                omega = _build(resolution)

        self.omega = omega  # the grid of points
        self.normal_vectors = normal_vectors  # normal vectors
        self.resolution = resolution  # resolution of discretisation
        if add_corners_to_knowns:
            knowns_mat = self.convert_standard_to_pymatgen(self.corners)
            self.add_knowns(knowns_mat)
        if self.precursors_reduced_dimension:
            for i in range(additional_constraints_needed)[::-1]:
                b=np.dot(self.convert_to_standard_basis(self.omega[0]),self.normal_vectors[i])
                self.normal_vectors_b=np.append([b],self.normal_vectors_b)

        if hasattr(self,'charge_ranges') and self.charge_ranges:
            self.set_charge_extreme_constraints()

    def cartesian(self, arrays, out=None):
        """Compute the Cartesian product of a sequence of 1-D arrays into a 2-D array of all combinations."""
        arrays = [np.asarray(x) for x in arrays]
        dtype = arrays[0].dtype
        n = np.prod([x.size for x in arrays])
        if out is None:
            out = np.zeros([n, len(arrays)], dtype=dtype)
        m = int(n / arrays[0].size)
        out[:, 0] = np.repeat(arrays[0], m)
        if arrays[1:]:
            self.cartesian(arrays[1:], out=out[0:m, 1:])
            for j in range(1, arrays[0].size):
                out[j * m:(j + 1) * m, 1:] = out[0:m, 1:]
        return out

    def set_charge_extreme_constraints(self, charge_scalar=0):
        """Compute and store the extreme-charge half-plane constraints for a charge-range field, then filter omega."""
        #this function finds the two hyperplanes corresponding to max and min charges
        has_range = [isinstance(x, list) for x in self.species_dict.values()]
        if np.all(not has_range):
            raise Exception(
                'Error, to get charge extremes at least one species must have '
                + 'a range of charges')
        ranges = [
            x for x, y in zip(self.species_dict.values(), has_range) if y
        ]
        fixed = [
            x if not y else 0
            for x, y in zip(self.species_dict.values(), has_range)
        ]
        self.charge_max = np.array(fixed)
        self.charge_max[has_range] = np.array(ranges)[:, 1]
        self.charge_min = np.array(fixed)
        self.charge_min[has_range] = np.array(ranges)[:, 0]
        self.constrain_charge_omega(charge_scalar)
        constraints = []
        combinations = self.cartesian(ranges)
        for c in combinations:
            f = np.array(fixed)
            f[has_range] = c
            constraints.append(f)
        self.charge_extreme_constraints = constraints

    def constrain_charge_omega(self, charge_scalar=0, method='b'):
        """Filter omega to retain only points whose total charge lies within the allowed range."""
        #default charge_scalar changed from 3 to 0 (charge neutral = sum to 0)
        if method == 'b':
            omega_s = self.convert_to_standard_basis(self.omega, norm=1)
            keep = (
                (np.einsum('...i,i->...', omega_s, self.charge_min) <= charge_scalar) &
                (np.einsum('...i,i->...', omega_s, self.charge_max) >= charge_scalar)
            )
            self.omega = self.omega[keep]
        return

    def convert_to_standard_basis(self, points, norm=1):
        """
        Convert points from the constrained basis back to standard (element-fraction) coordinates.

        The **standard basis** is the full-dimensional element-fraction representation,
        i.e. a length-``nelements`` vector where entry *i* is the mole fraction of
        element *i*.  Points in the standard basis lie on the charge-neutral simplex
        face and are comparable to published phase diagrams.

        The **constrained basis** is a lower-dimensional Cartesian coordinate system
        embedded in the plane defined by all equality constraints (charge neutrality,
        sum=1, etc.).  It has dimension ``constrained_dim = nelements - n_constraints``
        and is what :attr:`omega`, :attr:`corners`, and all PICIP probability arrays
        use internally.

        Parameters
        ----------
        points : array_like of shape (constrained_dim,) or (m, constrained_dim)
            One or more points in the constrained basis.
        norm : int or float, optional, default 1
            L1 norm to rescale the output to.  The default of 1 gives mole fractions
            summing to 1.  Use ``norm=100`` for at.-% output, for instance.

        Returns
        -------
        ndarray of shape (nelements,) or (m, nelements)
            Corresponding points in the standard element-fraction representation.
        """

        A = self.basis
        p_standard = self.contained_point + np.einsum(
            "ji,...j->...i", A, points
        )
        if p_standard.ndim == 1:
            p_standard = norm * p_standard / np.sum(p_standard)
        else:
            norma = np.sum(p_standard, axis=1)
            p_standard = (p_standard.T / norma).T
            p_standard *= norm
        return p_standard

    def convert_to_constrained_basis(self, points):
        """
        Convert points from standard (element-fraction) coordinates to the constrained basis.

        The **constrained basis** is the low-dimensional Cartesian space in which
        the phase field is represented after all equality constraints (sum=1, charge
        neutrality, etc.) have been applied.  Its dimension is ``constrained_dim``.
        This is the coordinate system used by :attr:`omega`, probability density
        arrays, and all visualiser classes.

        See :meth:`convert_to_standard_basis` for the inverse transformation and a
        fuller description of both bases.

        Parameters
        ----------
        points : array_like of shape (nelements,) or (m, nelements)
            One or more compositions in the standard element-fraction representation.
            Each row is normalised to sum to 1 before projection.

        Returns
        -------
        ndarray of shape (constrained_dim,) or (m, constrained_dim)
            Corresponding points in the constrained basis.
        """
        point = np.array(points)
        if point.ndim == 1:
            point = np.array(point) / sum(point)
            point_s = point - self.contained_point
            point = np.einsum("ij,j", self.basis, point_s)
            return point
        else:
            norma = np.sum(point, axis=1)
            point = (point.T / norma).T
            point = point - self.contained_point
            point = np.einsum("ij,...j->...i", self.basis, point)
            return point

    def add_constraints(self, ratio, minimum_amount, omega=None):
        """
        Remove grid points from ``omega`` that violate a linear inequality on element fractions.

        Applies the constraint ``ratio · x >= minimum_amount`` (in standard
        element-fraction coordinates) and discards any grid point that fails it.
        This is the low-level building block used by :meth:`constrain_minimum_amount`;
        call that method instead when you simply want a floor on an element's fraction.

        Can be called multiple times to stack several constraints.  Each call
        permanently removes rows from ``self.omega``; the operation is not reversible
        without re-running a ``setup_*`` method.

        Parameters
        ----------
        ratio : array_like of shape (nelements,)
            Coefficients ``a`` in the inequality ``a · x >= b``, expressed in
            standard (element-fraction) coordinates.  For example,
            ``[1, 0, 0, 0]`` constrains the fraction of the first element.
        minimum_amount : float
            Threshold ``b`` in the inequality.  Fractions lie in ``[0, 1]`` so
            ``minimum_amount`` should be in the same range for single-element
            constraints.
        omega : ndarray of shape (N, constrained_dim), optional
            Grid to filter.  Defaults to ``self.omega``; pass ``self.omega_cut``
            if you want to operate on a previously trimmed sub-grid.

        Returns
        -------
        None
        """
        if omega is None:
            omega=self.omega
        omega_s =self.convert_to_standard_basis(omega, norm=1)
        omega = np.delete(
            omega,
            np.where(
                np.einsum("...i,i->...", omega_s, ratio) < minimum_amount
            ),
            axis=0,
        )
        self.omega = omega
        self.omega_cut = omega

    def add_precursors(self, precursors, constrain_omega=True):
        """
        Adds precursor phases to the phase field and eliminates points from omega that are not inside
        the polytope defined by the precursors.

        Parameters
        ----------
        precursors : list
            List of precursor phases as pymatgen Composition objects).

        constrain_omega : bool, optional
            Whether to constrain omega based on the precursors' polytope. Defaults to True.

        Returns
        -------
        None
        """
        self.add_knowns(precursors)
        self.precursors_size = [x.num_atoms for x in precursors]
        precursors_standard = []
        precursors_label = []
        for k in precursors:
            precursors_standard.append(
                [k.get_atomic_fraction(e) for e in self.elements]
            )
            precursors_label.append(k.reduced_formula)
        precursors_standard = np.array(precursors_standard)
        precursors_constrained = self.convert_to_constrained_basis(precursors_standard)
        self.precursors_standard = precursors_standard
        self.precursors_constrained = precursors_constrained

        # Compute the convex hull of the precursors in the constrained basis
        hull = ConvexHull(precursors_constrained)

        # Extract the inequalities (half-space representation) from the convex hull
        # The inequalities are in the form (A x) + b <= 0
        inequalities = hull.equations  # Each row is [normal_vector, -offset]
        A = inequalities[:, :-1]
        b = inequalities[:, -1]

        # Store the precursor corners (vertices of the polytope)
        self.precursor_corners = precursors_constrained

        # Extract edges from the convex hull for visualization or further analysis
        edges = set()
        for simplex in hull.simplices:
            # Each simplex is a facet (in N-1 dimensions)
            for i in range(len(simplex)):
                for j in range(i + 1, len(simplex)):
                    edge = tuple(sorted((simplex[i], simplex[j])))
                    edges.add(edge)
        self.precursor_edges = list(edges)

        # If omega needs to be constrained, eliminate points outside the polytope
        if constrain_omega:
            omega = self.omega  # Assuming omega is already in the constrained basis
            # Check which points in omega satisfy all the inequalities
            # Since inequalities are (A x) + b <= 0, we can compute A x + b
            omega_in_polytope = omega[
                np.all(np.dot(omega, A.T) + b <= 1e-12, axis=1)
            ]
            self.omega_cut = omega_in_polytope
        else:
            self.omega_cut = self.omega


    def add_knowns(self, knowns, make_plotting_df=False, tol=0.05):
        """
        Adds known compositions to the phase field.

        Known compositions are specified as string to be interpreted as pymatgen Composition objects

        Parameters
        ----------
        knowns : list of string
            List of known composition strings to be interpreted by pymatgen Composition.
        make_plotting_df : bool, optional, default=False
            Whether or not to create a plotting DataFrame of the knowns.
        tol : float, optional, default=0.05
            Tolerance for checking the validity of the compositions.

        Returns
        -------
        None
        """
        knowns=[Composition(x) for x in knowns]
        knowns_standard = []
        knowns_label = []
        for k in knowns:
            ks = [k.get_atomic_fraction(e) for e in self.elements]
            if self.normal_vectors.shape[0] > 1:
                if abs(np.dot(ks, self.normal_vectors[0])) > tol:
                    res = np.dot(ks, self.normal_vectors[0])
                    name = self.get_composition_strings(ks, out="txt")
                    print(f"Ignoring unbalanced known: {name} ({res}>{tol})")
                else:
                    knowns_standard.append(ks)
                    knowns_label.append(k.reduced_formula)
            else:
                knowns_standard.append(ks)
                knowns_label.append(k.reduced_formula)

        if knowns_standard:
            knowns_constrained = self.convert_to_constrained_basis(
                knowns_standard
            )
            if make_plotting_df:
                df = pd.DataFrame()
                df["Formula"] = knowns_label
                df["x"] = knowns_constrained[:, 0]
                df["y"] = knowns_constrained[:, 1]
                if self.constrained_dim == 3:
                    df["z"] = knowns_constrained[:, 2]
                df["Composition"]=self.get_composition_strings(knowns_standard)
                self.plotting_df = df
            if hasattr(self, "knowns_constrained"):
                self.knowns_constrained = np.vstack(
                    (knowns_constrained, self.knowns_constrained)
                )
            else:
                self.knowns_constrained = knowns_constrained

    def add_knowns_standard(
            self, knowns_standard, make_plotting_df=False, dp=2):
        """
        Takes known compositions in standard representation and adds them to the phase field.

        Parameters
        ----------
        knowns_standard : array_like or ndarray of shape (m, n)
            Known compositions in standard representation.
        make_plotting_df : bool, optional, default=False
            Whether or not to create a plotting DataFrame.
        dp : int, optional, default=2
            Decimal precision for rounding values in labels.

        Returns
        -------
        None
        """
        knowns_label = []
        for k in knowns_standard:
            kn = np.array(k) / sum(k)
            if self.normal_vectors.shape[0] > 1:
                if abs(np.dot(kn, self.normal_vectors[0])) > 0.05:
                    print(k)
                    print(np.dot(kn, self.normal_vectors[0]))
                    print(
                        "Ignoring unbalanced known: ",
                        self.get_composition_strings(k, out="txt"),
                    )
            rep = ""
            for i, j in zip(self.elements, k):
                if j != 0:
                    rep += i
                    if j != 1:
                        rep += "<sub>" + str(round(j, dp)) + "</sub>"
            knowns_label.append(rep)

        knowns_constrained = self.convert_to_constrained_basis(knowns_standard)
        if make_plotting_df:
            df = pd.DataFrame()
            df["Label"] = knowns_label
            df["x"] = knowns_constrained[:, 0]
            df["y"] = knowns_constrained[:, 1]
            if self.constrained_dim == 3:
                df["z"] = knowns_constrained[:, 2]
            self.knowns_plotting_df = df
        if hasattr(self, "knowns_constrained"):
            self.knowns_constrained = np.vstack(
                (knowns_constrained, self.knowns_constrained)
            )
        else:
            self.knowns_constrained = knowns_constrained

    def convert_standard_to_pymatgen(self, points):
        """
        Convert standard-basis compositions to pymatgen ``Composition`` objects.

        Parameters
        ----------
        points : array_like of shape (nelements,) or (m, nelements)
            One or more compositions in standard element-fraction coordinates.

        Returns
        -------
        pymatgen.core.Composition or list of pymatgen.core.Composition
            A single ``Composition`` when one point is supplied, or a list when
            multiple points are supplied.
        """
        point = np.array(points)
        point[np.isclose(point,0)]=0
        if point.ndim == 1:
            point_d = {}
            for el, x in zip(self.elements, point):
                point_d[el] = x
            point = Composition(point_d)
            return point
        else:
            points = []
            for p in point:
                points.append(self.convert_standard_to_pymatgen(p))
            return points

    def get_composition_strings(
        self, points, norm=1, out="html", prec=2,
        simple_ratio_lim=30, rtol=1e-8,
    ):
        """
        Format one or more compositions as human-readable strings.

        Accepts points in either the standard (element-fraction) or constrained
        basis and auto-detects which one is supplied based on the last dimension.
        When a composition is close to a simple integer ratio (e.g. 1:2:3) that
        ratio is used; otherwise decimal fractions rounded to ``prec`` places are
        shown.

        This method is used internally to populate hover-text labels in the
        visualiser classes; call it directly when you want nicely formatted
        composition labels for your own plots or CSV output.

        Parameters
        ----------
        points : array_like of shape (nelements,), (constrained_dim,),
                 (m, nelements), or (m, constrained_dim)
            One or more compositions in either basis.
        norm : int or float, optional, default 1
            L1 norm used when displaying decimal fractions (has no effect when
            the composition is representable as a simple ratio).
        out : {'html', 'txt', 'txt_short'}, optional, default 'html'
            Output format.  ``'html'`` wraps subscript numbers in ``<sub>`` tags
            suitable for Plotly hovertext.  ``'txt'`` uses space-separated plain
            text (e.g. ``'Mg 0.25 Al 0.25 S 0.5'``).  ``'txt_short'`` is the
            same but without spaces (e.g. ``'Mg0.25Al0.25S0.5'``), useful for
            filenames.
        prec : int, optional, default 2
            Number of decimal places shown when a simple integer ratio cannot be
            found within ``simple_ratio_lim``.
        simple_ratio_lim : int, optional, default 30
            Maximum integer multiplier to try when looking for a simple ratio
            representation.  Increase this for systems with large stoichiometric
            coefficients (e.g. ``Cu31Zn8``).
        rtol : float, optional, default 1e-8
            Relative tolerance used when testing whether a scaled composition is
            close to integers.

        Returns
        -------
        str
            If a single point was supplied.
        list of str
            If multiple points were supplied.

        Raises
        ------
        Exception
            If ``out`` is not one of the recognised format strings.
        """
        points = np.array(points)
        if points.ndim == 1:
            points = np.array([points])
        if points.shape[1]==self.constrained_dim:
            points=self.convert_to_standard_basis(points)
        if points.shape[1]!=self.nelements:
            raise Exception(
                    "Points must be provided in standard or constrained basis")
        comps = []
        for point in points:
            point = point / sum(point)
            best_norm=99999
            for t_norm in range(1, simple_ratio_lim):
                pointt = t_norm * point
                if np.allclose(np.round(pointt), pointt, rtol=rtol):
                    if t_norm < best_norm:
                        best_norm = t_norm
            if best_norm < simple_ratio_lim:
                point = np.round(point * best_norm)
                comp = ""
                if out == "html":
                    for x, el in zip(point, self.elements):
                        if x > 1:
                            comp += el + "<sub>"
                            comp += str(int(x)) + "</sub>"
                        if x == 1:
                            comp += el
                    comps.append(comp)
                elif out == "txt":
                    for x, el in zip(point, self.elements):
                        if x > 1:
                            comp += el + " "
                            comp += str(int(x)) + " "
                        if x == 1:
                            comp += el + " "
                    comps.append(comp)
                elif out == "txt_short":
                    for x, el in zip(point, self.elements):
                        if x > 1:
                            comp += el
                            comp += str(int(x))
                        if x == 1:
                            comp += el
                    comps.append(comp)
                else:
                    print(out)
                    raise Exception("Unknwon output format")
            else:
                point = point * norm
                comp = ""
                if out == "html":
                    for x, el in zip(point, self.elements):
                        x = round(x, prec)
                        if x != 0:
                            comp += el + "<sub>"
                            comp += str(x) + "</sub>"
                    comps.append(comp)
                elif out == "txt":
                    for x, el in zip(point, self.elements):
                        x = round(x, prec)
                        if x != 0:
                            comp += el + " "
                            comp += str(x) + " "
                    comps.append(comp)
                elif out == "txt_short":
                    for x, el in zip(point, self.elements):
                        x = round(x, prec)
                        if x != 0:
                            comp += el
                            comp += str(x)
                    comps.append(comp)
                else:
                    raise Exception("Unknwon output format")
        if len(comps) == 1:
            return comps[0]
        else:
            return comps

    def _get_constraint_lines(self):
        """
        Finds the planes corresponding to `x_i > 0` in constrained space using the assigned basis vectors.

        Returns
        -------
        list of ndarray
            List of arrays representing the coefficients of the constraint lines.
        """
        # function that uses the assigned basis vectors to find the planes
        # corresponding to x_i>0 in constrained space
        line_coefficients = []
        for i in range(self.basis.shape[1]):
            coefficients = [self.basis[j][i] for j in range(self.basis.shape[0])]
            # Skip elements that have no projection into the constrained space
            # (e.g. Cu in a dim-reduced Cu=0 field). Floating-point near-zero
            # values would otherwise produce a spurious half-space filter.
            if np.linalg.norm(coefficients) < 1e-10:
                continue
            coefficients.append(self.contained_point[i])
            line_coefficients.append(np.array(coefficients))
        return line_coefficients


    def convert_string_to_standard(self, composition):
        """
        Converts a composition string to standard representation.

        Parameters
        ----------
        composition : str
            Composition string representing elements.

        Returns
        -------
        ndarray
            Composition in standard representation.
        """
        c = Composition(composition)
        s = [c.get_atomic_fraction(el) for el in self.elements]
        return s

    def check_neutrality(self, points, show=False):
        """
        Checks if the points are charge neutral.

        Parameters
        ----------
        points : array_like or ndarray of shape (n,) or (m, n)
            Points to check for neutrality.
        show : bool, optional, default=False
            Whether or not to print the neutrality check results.

        Returns
        -------
        None
        """
        if self.normal_vectors.shape[0] == 1:
            return True  # no charge constraint — uncharged field, nothing to check
        points = np.array(points)
        if points.ndim == 1:
            points = [points]
        for i in points:
            res = np.dot(self.normal_vectors[0], i)
            if abs(res) > 0.01:
                print(f"Error, point {i} is not balanced. ({res})")
                return False
            else:
                if show:
                    print(res)
        return True




    def _find_vertices_and_edges(self):
        """
        Finds the vertices and edges of the phase field polytope.

        Uses standard representation constraints to compute the vertices and edges of
        the phase field polytope based on shared constraints between vertices.

        Returns
        -------
        tuple
            A tuple containing:
            - vertices: ndarray of unique vertices of the polytope in standard representation.
            - edges: list of tuples representing edges between vertices.

        Notes
        -----
        The method is currently limited to single-charge species.
        """
        def _edges_from_vertices(verts):
            """Compute edges from a vertex set via ConvexHull projection into the polytope subspace."""
            try:
                d = self.constrained_dim
                center = np.mean(verts, axis=0)
                _, _, Vt = np.linalg.svd(verts - center, full_matrices=False)
                coords = (verts - center) @ Vt[:d].T
                hull = ConvexHull(coords)
                edges_set = set()
                for simplex in hull.simplices:
                    for i, j in combinations(simplex, 2):
                        edges_set.add(tuple(sorted((i, j))))
                return list(edges_set)
            except Exception:
                n = len(verts)
                if n == 2:
                    return [(0, 1)]
                elif n == 3:
                    return [(0, 1), (1, 2), (0, 2)]
                else:
                    return []

        num_variables = self.nelements

        # Inequality constraints: x_i >= 0 for all i
        A_ub = -np.eye(num_variables)
        b_ub = np.zeros(num_variables)

        # Charge range half-planes: charge_min · x <= 0  and  charge_max · x >= 0
        if hasattr(self, 'charge_max') and hasattr(self, 'charge_min'):
            A_ub = np.vstack([A_ub, self.charge_min, -self.charge_max])
            b_ub = np.hstack([b_ub, np.zeros(2)])

        # Equality constraints: sum(x_i) = 1
        A_eq = np.ones((1, num_variables))
        b_eq = np.array([1])

        # Custom constraint coefficients (all rows except the last sum=1 row)
        if len(self.normal_vectors) > 1:
            for custom_c in self.normal_vectors[:-1]:
                A_eq = np.vstack([A_eq, custom_c])
                b_eq = np.append(b_eq, 0)

        # Determine the effective dimension (constrained dimension) of the polytope
        num_equality_constraints = np.linalg.matrix_rank(A_eq)
        polytope_dimension = num_variables - num_equality_constraints

        vertices = []

        # Each vertex lies where polytope_dimension inequalities are active,
        # combined with all equalities.  Iterate over inequality combos only.
        for comb in combinations(range(len(A_ub)), polytope_dimension):
            try:
                A_sub = np.vstack([A_ub[list(comb)], A_eq])
                b_sub = np.hstack([b_ub[list(comb)], b_eq])
                vertex = np.linalg.solve(A_sub, b_sub)
                if (
                    np.all(np.dot(A_ub, vertex) - b_ub <= 1e-4)
                    and np.allclose(np.dot(A_eq, vertex), b_eq, atol=1e-4)
                ):
                    vertices.append(vertex)
            except np.linalg.LinAlgError:
                continue

        vertices = np.unique(np.round(vertices, 8), axis=0)
        edges = _edges_from_vertices(vertices)

        self.corners = vertices
        self.edges=edges
        corner_compositions = self.get_composition_strings(vertices)
        self.corner_compositions = corner_compositions
                
    def constrain_minimum_amount(
            self, min_amount, el_indexes=None, use_cut_omega=False):
        """
        Require each specified element to have at least ``min_amount`` mole fraction in every grid point.

        Useful for excluding near-binary or near-ternary corners of the phase field
        from the inference grid, for example to focus PICIP on genuinely quaternary
        compositions.  Internally calls :meth:`add_constraints` once per element.

        Parameters
        ----------
        min_amount : float
            Lower bound on the element mole fraction (standard basis, so values
            are in ``[0, 1]``).  For example, ``0.05`` excludes compositions
            where any selected element is below 5 at.-%.
        el_indexes : list of int, optional
            Indices (0-based, into ``self.elements``) of the elements to constrain.
            If omitted, the constraint is applied to *all* elements.  To constrain
            only e.g. the second element use ``el_indexes=[1]``.
        use_cut_omega : bool, default False
            If ``True``, filter ``self.omega_cut`` (a previously trimmed sub-grid)
            rather than the full ``self.omega``.

        Returns
        -------
        None
        """
        if el_indexes is None:
            el_indexes=range(self.nelements)
        for i in el_indexes:
            if use_cut_omega:
                omega=self.omega_cut
            else:
                omega=self.omega
            ratio=[0]*self.nelements
            ratio[i]=1
            self.add_constraints(ratio,min_amount,omega=omega)


    def create_new_omega(
            self, resolution, minimum_amount=None, el_indexes=None,
            precursors=None):
        """
        Build a fresh grid at a different resolution while reusing the current field's charge constraints.

        Creates a temporary :class:`Phase_Field` with the same elements and charge
        vector and returns its ``omega`` (or ``omega_cut`` if precursors are given)
        without modifying the current object.

        Parameters
        ----------
        resolution : int
            Grid spacing expressed as ``1/resolution`` in each constrained-basis
            direction.
        minimum_amount : float, optional
            If supplied, passed to :meth:`constrain_minimum_amount` on the
            temporary field.
        el_indexes : list of int, optional
            Indices of elements to apply the minimum-amount constraint to.
            Passed through to :meth:`constrain_minimum_amount`.
        precursors : list of pymatgen Composition, optional
            Precursor phases (as Composition objects, *not* strings) used to
            trim the grid to a convex hull.

        Returns
        -------
        ndarray of shape (N, constrained_dim)
            The newly built grid in the constrained basis.
        """
        species_dic={}
        for el,c in zip(self.elements,self.normal_vectors[0]):
            species_dic[el]=c
        temp_field=Phase_Field()
        temp_field.setup_charged(species_dic,resolution=resolution,basis=self.basis)
        if minimum_amount is not None:
            temp_field.constrain_minimum_amount(
                    minimum_amount, el_indexes=el_indexes)
        if precursors is not None:
            temp_field.add_precursors(precursors)
            return temp_field.omega_cut
        return temp_field.omega

    def add_corners_to_knowns(self):
        """Add the phase-field corner compositions to ``self.knowns_constrained`` for plotting."""
        self.add_knowns_standard(self.corners)


    def random_point_constrained(self, n=1):
        """
        Draw one or more grid points uniformly at random from ``omega``.

        Parameters
        ----------
        n : int, optional, default 1
            Number of points to draw (without replacement).

        Returns
        -------
        ndarray of shape (constrained_dim,) if n==1, else (n, constrained_dim)
            Randomly selected grid point(s) in the constrained basis.
        """
        choice = random.randrange(len(self.omega))
        indicis = range(len(self.omega))
        r_indicis = np.random.choice(indicis,n,replace=False)
        ps=self.omega[r_indicis]
        if n==1:
            return ps[0]
        else:
            return ps

    def convert_standards_to_formulas(self, standard_reps):
        """
        Format a list of standard-basis compositions as plain-text element-amount strings.

        Parameters
        ----------
        standard_reps : array_like of shape (m, nelements)
            Compositions in standard element-fraction coordinates.

        Returns
        -------
        list of str
            Each entry is a space-separated string such as ``'Mg 0.25 Al 0.25 S 0.5'``.
        """
        formulas = []
        for k in standard_reps:
            rep = ""
            for a, b in zip(k, self.elements):
                rep += b + " " + str(a) + " "
            rep = rep[:-1]
            formulas.append(rep)
        return formulas

    def get_constraint_lines(self):
            """
            Finds the planes corresponding to `x_i > 0` in constrained space using the assigned basis vectors.

            Returns
            -------
            list of ndarray
                List of arrays representing the coefficients of the constraint lines.
            """
            # function that uses the assigned basis vectors to find the planes
            # corresponding to x_i>0 in constrained space
            line_coefficients = []
            for i in range(self.basis.shape[1]):
                coefficients = []
                for j in range(self.basis.shape[0]):
                    coefficients.append(self.basis[j][i])
                coefficients.append(1 * self.contained_point[i])
                line_coefficients.append(np.array(coefficients))
            return line_coefficients





