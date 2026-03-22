"""
PICIP — Probabilistic Isolation of Inorganic Crystalline Phases
===============================================================
Bayesian inference of an unknown phase composition from measured XRD data.

Four-step workflow
------------------
1. **Set up a phase field** — choose your element system and grid::

       from setup_phase_field import Phase_Field
       pf = Phase_Field()
       pf.setup_uncharged(["Fe", "Mn", "Ti"])          # 2-D constrained space
       # or: pf.setup_charged({"Fe": 3, "O": -2})
       # or: pf.setup_charge_ranges({"Fe": [2,3], "O": -2})
       # or: pf.setup_custom(...)

2. **Create and add Sample objects** — supply the measured composition, the
   known co-existing phases identified by XRD, and their Rietveld mass
   fractions::

       from run_PICIP import PICIP, Sample
       sample = Sample("s1", "Fe2Mn4Ti4")
       sample.add_knowns(["FeMn", "FeTi"])
       sample.add_mass_weights([0.3, 0.7])      # Rietveld mass fractions
       sample.set_predicted_error(0.3)           # 0 = precise, 1 = rough
       picip = PICIP(pf)
       picip.add_sample(sample)

3. **Run inference** — returns a Prediction with a probability density over
   the phase field grid::

       pred = picip.run(n_l=50, n_p=50)

4. **Plot results** — ``make_plotter`` automatically picks 2-D or 3-D::

       from visualise_cube import make_plotter
       pl = make_plotter(pf)
       pl.plot_prediction_results(pred)
       pl.show()
"""
import numpy as np
import scipy.stats
from scipy.stats import dirichlet, multivariate_normal
from scipy.interpolate import griddata
from pymatgen.core import Composition
import pandas as pd


def _requires_samples(method):
    """Guard decorator: raises ValueError if PICIP has no samples registered."""
    def wrapper(self, *args, **kwargs):
        if not self.samples:
            raise ValueError(
                f"No samples added. Call add_sample() before calling {method.__name__}()."
            )
        return method(self, *args, **kwargs)
    wrapper.__name__ = method.__name__
    wrapper.__doc__  = method.__doc__
    return wrapper


class Suggestions:
    """
    Suggestion points drawn from a PICIP probability density.

    Attributes
    ----------
    constrained : ndarray, shape (n, dim)
        Points in constrained basis. First row is the mean.
    standard : ndarray, shape (n, nelements)
        Points in standard (elemental fraction) basis.
    labels : list of str
        'mean' for the first point, 'sampled' for the rest.
    elements : list of str
        Element names, matching columns of `standard`.
    """

    def __init__(self, points_constrained, phase_field, labels):
        self.constrained = points_constrained
        self.standard    = phase_field.convert_to_standard_basis(points_constrained)
        self.labels      = labels
        self.elements    = phase_field.elements

    def save(self, path):
        """Save suggestions to a CSV file with elemental fraction columns."""
        df = pd.DataFrame(self.standard, columns=self.elements)
        df.insert(0, 'type', self.labels)
        df.to_csv(path, index=False)
        print(f"Saved {len(df)} suggestion points to {path}")


def _orthogonal_basis(d):
    """Return a (dim × dim) orthonormal matrix whose first column is d/|d|.

    Uses successive Gram-Schmidt orthogonalisation against the standard
    basis vectors, so the result is deterministic for any input direction.
    """
    dim = len(d)
    d = np.array(d, dtype=float)
    d /= np.linalg.norm(d)
    basis = [d]
    for i in range(dim):
        e = np.zeros(dim)
        e[i] = 1.0
        for b in basis:
            e -= np.dot(e, b) * b
        n = np.linalg.norm(e)
        if n > 1e-10:
            basis.append(e / n)
        if len(basis) == dim:
            break
    return np.column_stack(basis)  # (dim, dim), first column = d

class PICIP:
    """
    Bayesian inference engine for PICIP.

    Holds a Phase_Field and a list of Sample objects.  The typical workflow is:
    add one or more samples with add_sample(), call run() to obtain a Prediction,
    then optionally call suggest() to draw next-experiment compositions.

    Parameters
    ----------
    phase_field : Phase_Field
        The phase field within which inference is performed.
    """

    def __init__(self,phase_field):
        self.phase_field=phase_field
        self.samples=[]

    def add_sample(self, sample):
        """
        Validate a Sample and register it for inference.

        Converts the sample composition and all known phases into both the
        constrained and standard bases, computes the weight-averaged known
        position (average_k), and appends the sample to self.samples.

        Parameters
        ----------
        sample : Sample
            A fully configured Sample object.  Must have add_knowns() called and
            either add_mass_weights() or add_molar_weights() called beforehand.

        Raises
        ------
        Exception
            If the sample or any known phase contains elements outside the phase
            field, or if predicted_error has not been set on the sample.
        """
        if set([x.symbol for x in sample.composition.elements]) > set(self.phase_field.elements):
            print(set([x.symbol for x in sample.composition.elements]),set(self.phase_field.elements))
            raise Exception(f"Sample {sample.name} contains elements not in phase field")
        comp_standard=np.array([sample.composition.get_atomic_fraction(el) for el in self.phase_field.elements])
        sample.comp_constrained=self.phase_field.convert_to_constrained_basis(comp_standard)
        for k in sample.knowns:
            if set([x.symbol for x in k.elements]) > set(self.phase_field.elements):
                raise Exception(f"Known phase {k} contains elements not in phase field")
        sample.knowns_standard=np.array([[k.get_atomic_fraction(el) for el in self.phase_field.elements] for k in sample.knowns])

        sample.knowns_constrained=self.phase_field.convert_to_constrained_basis(sample.knowns_standard)
        sample.average_k=np.sum(sample.weights[:,np.newaxis]*sample.knowns_constrained, axis=0)

        if sample.predicted_error is None:
            raise Exception(f"Predicted error has not bee set for sample {sample.name}")
        self.samples.append(sample)
                                
    @_requires_samples
    def run(self, sample_indexes=None, version='b', n_l=50, n_p=50, name=None):
        """
        Run PICIP inference and return a Prediction.

        For a single sample the returned probability density is the one produced
        by compute_p_dirichlet.  For multiple samples the densities are
        multiplied together (product-of-experts), renormalised, and stored in a
        single Prediction.  If the combined density is zero, the method
        automatically retries with a progressively larger predicted_error (up to
        four doublings) before raising an error.

        Parameters
        ----------
        sample_indexes : None, int, or slice
            Which samples in self.samples to use.  None (default) uses all
            samples.  An integer selects a single sample by index.  A slice
            selects a subset.
        version : str, optional
            Inference method.  ``'b'`` (default) uses the Dirichlet simplex
            method, dispatching on the number of known phases (1 known →
            Gaussian automatically).  ``'gaussian'`` forces the Gaussian
            hemisphere method regardless of the number of knowns — equivalent
            to the original PICIP algorithm from the paper.
        n_l : int, optional
            Resolution of the known simplex discretisation: number of ray
            directions per sample.  Higher values give a smoother simplex PDF
            at the cost of compute time.  Default 50.
        n_p : int, optional
            Number of interpolation points per ray when projecting the simplex
            PDF onto the omega grid.  Higher values reduce griddata artefacts
            for coarse grids.  Default 50.
        name : str or None
            Label for the returned Prediction.  Defaults to the first sample's
            name for single-sample runs, or ``'combined probability'``.

        Returns
        -------
        Prediction
            Contains the normalised probability density over omega and back-
            references to the samples used.

        Raises
        ------
        ValueError
            If the density contains NaNs, is zero everywhere, or cannot be
            rescued by relaxing predicted_error.
        """
        if sample_indexes==None:
            samples=self.samples
            if name is None:
                name=samples[0].name if len(samples)==1 else 'combined probability'
        elif isinstance(sample_indexes,int):
            samples=[self.samples[sample_indexes]]
            if name is None:
                name=samples[0].name
        else:
            samples=self.samples[sample_indexes]
            if name is None:
                name=samples[0].name if len(samples)==1 else 'combined probability'

        if version=='b':
            p_densities=[]
            for sample in samples:
                p_densities.append(self.compute_p_dirichlet(sample, n_l, n_p))
            if len(samples)==1:
                prob_density=p_densities[0]
            else:
                prob_density=np.prod(np.array(p_densities),axis=0)
        elif version=='gaussian':
            p_densities=[]
            for sample in samples:
                p_densities.append(self.compute_p_gaussian(sample, n_l, n_p))
            if len(samples)==1:
                prob_density=p_densities[0]
            else:
                prob_density=np.prod(np.array(p_densities),axis=0)

        if np.any(np.isnan(prob_density)):
            raise ValueError(
                "Probability density contains NaN values.  This usually means a "
                "Dirichlet PDF was evaluated at a degenerate point — check that "
                "mass_weights are valid and predicted_error is not too small."
            )

        if prob_density.sum() == 0:
            if len(samples) > 1:
                # Auto-retry with increasing predicted_error
                scale = 2.0
                max_retries = 4
                for attempt in range(1, max_retries + 1):
                    new_err = samples[0].predicted_error * (scale ** attempt)
                    print(
                        f"Combined density is zero — samples have no overlapping support. "
                        f"Retrying with predicted_error × {scale**attempt:.0f} "
                        f"({new_err:.4f}) [attempt {attempt}/{max_retries}]..."
                    )
                    for s in samples:
                        s.predicted_error = s.predicted_error * scale
                    p_densities = [self.compute_p_dirichlet(s, n_l, n_p) for s in samples]
                    prob_density = np.prod(np.array(p_densities), axis=0)
                    if prob_density.sum() > 0:
                        print(
                            f"  Success.  Final predicted_error values: "
                            + ", ".join(f"{s.name}={s.predicted_error:.4f}" for s in samples)
                            + ".  Consider updating your Sample objects to use these values."
                        )
                        break
                else:
                    raise ValueError(
                        f"Combined density is still zero after {max_retries} retries "
                        f"(predicted_error scaled up to {samples[0].predicted_error:.4f}). "
                        "The samples may not be consistent with a common unknown phase."
                    )
            else:
                raise ValueError(
                    "Probability density is zero everywhere — check that the sample "
                    "composition and knowns are consistent, and that predicted_error "
                    "is not too small."
                )

        prob_density=prob_density/prob_density.sum()

        # Normalise and store individual densities for multi-sample plotting
        individual_densities = [p/p.sum() for p in p_densities] if len(p_densities) > 1 else None

        prediction=Prediction(name,prob_density,samples,individual_densities)
        return prediction

    @_requires_samples
    def suggest(self, prediction, n, min_dist=None, past_samples=None):
        """
        Return n suggestion points drawn from a Prediction's probability density.

        The first point is the probability-weighted mean (snapped to the nearest
        omega grid point).  The remaining n-1 points are sampled without
        replacement from the distribution.

        Parameters
        ----------
        prediction : Prediction
        n : int
            Total number of suggestion points (must be >= 1).
        min_dist : float or None
            Minimum Euclidean distance (in constrained space) between any two
            suggestion points, and between suggestions and past sample points.
            Points within min_dist of an already-chosen point are excluded from
            the candidate pool before each draw.  If None, no constraint is applied.
        past_samples : list of ndarray or None
            Previously measured sample compositions in constrained space to
            also enforce min_dist against.  If None, uses the compositions of
            all samples in the prediction.

        Returns
        -------
        Suggestions
        """
        p     = prediction.prob_density.copy()
        p    /= p.sum()
        omega = self.phase_field.omega

        # Build initial exclusion set from past samples
        excluded = []
        if min_dist is not None:
            if past_samples is not None:
                excluded = [np.asarray(s) for s in past_samples]
            else:
                excluded = [s.comp_constrained for s in prediction.samples]

        def _mask_near(p_work, chosen):
            """Zero out omega points within min_dist of any chosen point."""
            if min_dist is None or not chosen:
                return p_work
            chosen_arr = np.array(chosen)                          # (k, dim)
            dists = np.min(
                np.linalg.norm(omega[:, None, :] - chosen_arr[None, :, :], axis=2),
                axis=1)                                            # (N,)
            p_work = p_work.copy()
            p_work[dists < min_dist] = 0.0
            return p_work

        # Mean — weighted average snapped to nearest grid point
        p_work   = _mask_near(p.copy(), excluded)
        mean     = p_work / p_work.sum() @ omega
        mean_idx = np.argmin(np.linalg.norm(omega - mean, axis=1))
        chosen   = excluded + [omega[mean_idx]]
        points   = [omega[mean_idx]]
        labels   = ['mean']

        # Sample remaining n-1 points sequentially, re-masking after each pick
        for _ in range(n - 1):
            p_work = _mask_near(p.copy(), chosen)
            if p_work.sum() == 0:
                print(f'Warning: min_dist too large — only {len(points)} '
                      f'of {n} suggestion points could be placed.')
                break
            p_work /= p_work.sum()
            idx     = np.random.choice(len(omega), p=p_work)
            chosen.append(omega[idx])
            points.append(omega[idx])
            labels.append('sampled')

        suggestions = Suggestions(np.array(points), self.phase_field, labels)
        prediction.suggestions = suggestions
        return suggestions

    def compute_p_dirichlet(self, sample, n_l, n_p, floor=1e-6):
        """
        Compute the probability density for one sample using the Dirichlet method.

        Dispatches on the number of known phases:

        * 1 known  → :meth:`compute_p_gaussian` (Fibonacci demisphere, Gaussian
          weighting by perpendicular distance to the single known).
        * 2 knowns → :meth:`discretise_known_simplex_2d` + :meth:`map_to_ab_line`
          to get barycentric coordinates along the known line segment.
        * 3 knowns → :meth:`discretise_known_simplex_3d` (vectorised
          Möller-Trumbore) to sample the full triangular known simplex.
        * >3 knowns → raises ``Exception`` (not implemented).

        After computing the Dirichlet likelihood on the known simplex, the
        simplex PDF is projected onto the omega grid via
        :meth:`project_known_simplex`.  Intermediate data are stored on the
        sample object for use by the detail-view visualiser.

        Parameters
        ----------
        sample : Sample
            Must have been registered with add_sample().
        n_l : int
            Simplex discretisation resolution (number of ray directions).
        n_p : int
            Points per projection ray.
        floor : float, optional
            Minimum value used to avoid zero-valued Dirichlet parameters.
            Increase slightly if you see Dirichlet PDF underflow.  Default 1e-6.

        Returns
        -------
        ndarray, shape (N,)
            Normalised probability density over omega.

        Raises
        ------
        Exception
            If >3 knowns are present, or if the Dirichlet PDF produces inf/nan.
        """
        weights=sample.weights
        k_c=sample.knowns_constrained
        s=sample.comp_constrained
        mass_weights=sample.mass_weights
        predicted_error=sample.predicted_error
        # Dispatch: single known → Gaussian, 2/3 knowns → Dirichlet
        if len(k_c) == 1:
            return self.compute_p_gaussian(sample, n_l, n_p)

        # Discretize the known simplex
        dim = self.phase_field.constrained_dim
        if len(k_c) == 2:
            # Angle-sweep along k1-k2 segment; dimension-agnostic
            line_points = self.discretise_known_simplex_2d(k_c[0], k_c[1], s, n_l)
            normalised_points = np.array(
                [self.map_to_ab_line(point, k_c[0], k_c[1]) for point in line_points])
        elif len(k_c) == 3:
            if dim == 3:
                # Möller-Trumbore solid-angle sweep (requires np.cross, 3D only)
                line_points, normalised_points = self.discretise_known_simplex_3d(
                    k_c[0], k_c[1], k_c[2], s, n_l)
            else:
                # Uniform barycentric grid for 2D constrained space
                line_points, normalised_points = self._discretise_triangle_2d(
                    k_c[0], k_c[1], k_c[2], s, n_l)
        else:
            raise Exception("Method not implemented for > 3 knowns")

        # Get the probability density function (PDF) of the Dirichlet distribution
        alpha=weights/sample.predicted_error
        dirichlet_pdf = []
        dirichlet_likelihood = []
        dirichlet_likelihoodb = []
        alpha=list(alpha)
        if 0 in mass_weights:
            mass_weights=[floor if x==0 else x-floor for x in weights_mass]

        for x in normalised_points:
            if np.any(np.abs(x)<floor):
                x=[floor if p<floor else p-floor for p in x]
            x=self.convert_weights_moletomass(sample.knowns,x)
            alpha_l=np.array(x)/predicted_error
            dirichlet_likelihood.append(dirichlet.pdf(mass_weights,alpha_l))


        known_simplex=line_points
        if np.any(np.isinf(dirichlet_pdf)):
            raise Exception('infa')
        if np.any(np.isnan(dirichlet_pdf)):
            raise Exception('nana')

        dirichlet_likelihood=np.array(dirichlet_likelihood)+np.array(sample.prior)/len(dirichlet_likelihood)
        known_pdf=dirichlet_likelihood/sum(dirichlet_likelihood)

        # Store simplex data for detailed visualisation
        sample._simplex_points = np.array(known_simplex)
        sample._simplex_pdf    = known_pdf
        sample._n_p            = n_p

        p_grid = self.project_known_simplex(known_simplex, known_pdf, s, n_p)

        # Store projected lines data (set by project_known_simplex)
        sample._projected_points = np.array(self.projected_simplex)
        sample._projected_p      = np.array(self.projected_simplex_p)
        sample._n_rays           = len(known_simplex)

        return p_grid

    def compute_p_gaussian(self, sample, n_l, n_p, tol=1e-4):
        """
        Single-known Gaussian method (original PICIP algorithm).

        Fires n_l rays from the sample point in a hemisphere of directions
        facing the known phase.  Each ray is weighted by a Gaussian PDF of
        the perpendicular distance from the ray to the known composition.
        The weighted densities are interpolated onto omega via griddata.

        Automatically called by compute_p_dirichlet when only one known is
        present.  Can also be called directly as a legacy interface.

        Parameters
        ----------
        sample : Sample
            Must have comp_constrained, average_k, predicted_error set.
        n_l : int
            Number of hemisphere directions (analogous to simplex resolution).
        n_p : int
            Points per ray for griddata interpolation.
        """
        dim  = self.phase_field.constrained_dim
        s    = sample.comp_constrained
        k    = sample.average_k          # = single known in constrained space
        pred_err = sample.predicted_error

        # Isotropic covariance in constrained space
        sigma = pred_err**2 * np.eye(dim)

        # Polygon constraints for boundary intersection
        lines = np.array(self.phase_field._get_constraint_lines())
        poly  = {'A': lines[:, :-1], 'b': -lines[:, -1]}

        # Fibonacci demisphere — deterministic uniform hemisphere facing AWAY from k
        # (same as legacy: b = sample_point - avg_known, keep dot(a,b) > 0)
        direction_away = s - k
        direction_away = direction_away / np.linalg.norm(direction_away)

        if dim == 3:
            golden = (1 + 5**0.5) / 2
            idx    = np.arange(n_l)
            theta  = 2 * np.pi * idx / golden
            phi    = np.arccos(1 - 2 * (idx + 0.5) / n_l)
            dirs   = np.stack([np.cos(theta)*np.sin(phi),
                               np.sin(theta)*np.sin(phi),
                               np.cos(phi)], axis=1)          # (n_l, 3)
        elif dim == 2:
            theta = np.linspace(0, 2*np.pi, n_l, endpoint=False)
            dirs  = np.stack([np.cos(theta), np.sin(theta)], axis=1)  # (n_l, 2)
        else:
            raise Exception(f"Gaussian method not implemented for constrained_dim={dim}")

        # Keep only directions on the hemisphere pointing away from k
        dirs = dirs[dirs @ direction_away > 0]

        # ── Ray sampling ──────────────────────────────────────────────────────
        all_points  = []
        all_values  = []

        for d in dirs:
            d = d / np.linalg.norm(d)
            intersect = self.find_intersection(poly, s, s + d)
            if intersect is None:
                continue
            line_len = np.linalg.norm(s - intersect)
            if line_len < 1e-10:
                continue

            # Orthogonal basis: first column = d, rest = perpendicular subspace
            Q = _orthogonal_basis(d)   # (dim, dim)

            if dim == 2:
                perp        = Q[:, 1]
                sigma_perp  = float(perp @ sigma @ perp)
                y           = float(perp @ (s - k))
                f           = scipy.stats.norm.pdf(y, 0, np.sqrt(max(sigma_perp, 1e-30)))
            else:
                perp        = Q[:, 1:].T                           # (dim-1, dim)
                sigma_perp  = perp @ sigma @ perp.T
                y           = perp @ (s - k)
                f           = multivariate_normal.pdf(
                                  y, np.zeros(dim - 1), sigma_perp)

            ray_pts = [s + x / (n_p - 1) * (intersect - s) for x in range(n_p)]
            all_points.extend(ray_pts)
            all_values.extend([f / line_len] * n_p)

        all_points = np.array(all_points)
        all_values = np.array(all_values)
        n_rays     = len(all_points) // n_p

        # Store detail-view data on sample (simplex not used for Gaussian case)
        sample._simplex_points   = None
        sample._simplex_pdf      = None
        sample._n_p              = n_p
        sample._n_rays           = n_rays
        sample._projected_points = all_points
        sample._projected_p      = all_values

        p_grid = griddata(all_points, all_values, self.phase_field.omega,
                          method='linear', fill_value=0)

        if p_grid.sum() == 0:
            raise Exception('Gaussian density: all zero after griddata interpolation')
        if np.any(p_grid < -tol):
            raise Exception(f'Gaussian density: values below zero tol ({p_grid.min():.2e})')
        p_grid[p_grid < 0] = 0
        if np.isnan(p_grid).any():
            p_grid = np.nan_to_num(p_grid)
        return p_grid / p_grid.sum()

    def apply_omega_filter(self, omega_filter):
        """Apply an external per-point weight mask to the accumulated density."""
        omega_filter=np.array(omega_filter)
        if hasattr(self,'values_full'):
            self.values_full=np.append(self.values_full,[omega_filter],axis=0)
        else:
            self.values_full=omega_filter[np.newaxis,:]
        self.values_all=np.ones(len(omega_filter))
        for i in self.values_full:
            self.values_all*=i
        if self.values_all.sum()==0:
            raise ZeroPException()
        else:
            self.values_all/=self.values_all.sum()


    def discretise_known_simplex_2d(self, start, stop, sample, n_l):
        """
        Discretise the line segment between two known phases into n_l points.

        Works for any constrained_dim (2 or 3).  Points are spaced uniformly
        in angle from the sample rather than uniformly along the segment, so
        that the Dirichlet likelihood is evaluated at equal angular intervals.

        The sweep is performed in the plane spanned by (start - sample) and
        (stop - sample), making it dimension-agnostic.

        Parameters
        ----------
        start : ndarray, shape (dim,)
            First known phase in constrained space.
        stop : ndarray, shape (dim,)
            Second known phase in constrained space.
        sample : ndarray, shape (dim,)
            Sample composition in constrained space.
        n_l : int
            Number of discretisation points.

        Returns
        -------
        ndarray, shape (n_l, dim)
            Points on the known line segment, equally spaced in angle from
            the sample.

        Raises
        ------
        Exception
            If the sample lies on the line between the two known phases.
        """
        v1 = start - sample   # vector from sample to start (k1)
        v2 = stop  - sample   # vector from sample to stop  (k2)

        # Build a 2-D coordinate system in the plane of v1 and v2
        u1 = v1 / np.linalg.norm(v1)
        v2_perp = v2 - np.dot(v2, u1) * u1
        perp_norm = np.linalg.norm(v2_perp)
        if perp_norm < 1e-10:
            raise Exception(
                'Sample lies on the line between the two known phases — '
                'angular discretisation is degenerate.'
            )
        u2 = v2_perp / perp_norm

        # Total angle subtended by the segment from sample
        cos_theta = np.clip(
            np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)), -1.0, 1.0
        )
        total_angle = np.arccos(cos_theta)

        # For a ray from sample in direction d(θ) = cos(θ)·u1 + sin(θ)·u2,
        # the intersection parameter on start + t*(stop-start) is:
        #   t = |v1| · sin(θ) / (b·cos(θ) − a·sin(θ))
        # where a = (stop-start)·u1, b = (stop-start)·u2.
        seg = stop - start
        a = np.dot(seg, u1)
        b = np.dot(seg, u2)
        v1_norm = np.linalg.norm(v1)

        points = []
        for theta in np.linspace(0.0, total_angle, n_l):
            denom = b * np.cos(theta) - a * np.sin(theta)
            if abs(denom) < 1e-12:
                raise Exception('Degenerate ray-segment intersection in discretise_known_simplex_2d')
            t = np.clip(v1_norm * np.sin(theta) / denom, 0.0, 1.0)
            points.append(start + t * seg)

        return np.array(points)

    def _discretise_triangle_2d(self, k1, k2, k3, sample, n):
        """
        Discretise the triangle between three known phases for constrained_dim=2.

        Generates a uniform barycentric grid of (n+1)*(n+2)//2 points on the
        k1-k2-k3 triangle and returns them with their barycentric coordinates.
        Used by compute_p_dirichlet when there are 3 known phases in a 2-D
        constrained space (where np.cross is not available).

        Parameters
        ----------
        k1, k2, k3 : ndarray, shape (2,)
            Known phase coordinates in constrained space.
        sample : ndarray, shape (2,)
            Sample composition (unused in computation; kept for API consistency).
        n : int
            Grid density; total points returned is (n+1)*(n+2)//2.

        Returns
        -------
        points : ndarray, shape (M, 2)
            Points on the known simplex triangle.
        bary_coords : ndarray, shape (M, 3)
            Barycentric coordinates [w_k1, w_k2, w_k3] for each point.
        """
        k1, k2, k3 = (np.asarray(x, dtype=float) for x in (k1, k2, k3))

        n_range = np.arange(n + 1)
        i_idx, j_idx = np.meshgrid(n_range, n_range, indexing='ij')
        i_idx, j_idx = i_idx.ravel(), j_idx.ravel()
        mask = i_idx + j_idx <= n
        i_idx, j_idx = i_idx[mask], j_idx[mask]
        k_idx = n - i_idx - j_idx

        alpha = i_idx / n
        beta  = j_idx / n
        gamma = k_idx / n

        points = alpha[:, None] * k1 + beta[:, None] * k2 + gamma[:, None] * k3
        bary_coords = np.stack([alpha, beta, gamma], axis=1)

        # Drop any grid point that coincides with the sample (would give a zero-length ray)
        sample_arr = np.asarray(sample, dtype=float)
        valid = np.linalg.norm(points - sample_arr, axis=1) > 1e-8
        return points[valid], bary_coords[valid]

    def discretise_known_simplex_3d(self, k1, k2, k3, sample, n):
        """
        Returns points on the triangular simplex (k1, k2, k3) that are uniformly
        distributed in solid angle from the sample point.

        Builds a triangular grid of (n+1)*(n+2)//2 directions by spherically
        interpolating between unit vectors from the sample to each known corner,
        then recovers the exact position on the flat triangle and its barycentric
        coordinates via vectorised Möller-Trumbore ray-triangle intersection.

        Parameters
        ----------
        k1, k2, k3 : array-like, shape (3,)
            Known phase coordinates in constrained space.
        sample : array-like, shape (3,)
            Sample composition in constrained space.
        n : int
            Grid density; total points returned is (n+1)*(n+2)//2.

        Returns
        -------
        points : ndarray, shape (M, 3)
            Intersection points on the known simplex triangle.
        bary_coords : ndarray, shape (M, 3)
            Barycentric coordinates [w_k1, w_k2, w_k3] for each point,
            suitable for direct use as Dirichlet parameters.
        """
        k1, k2, k3, sample = (np.asarray(x, dtype=float) for x in (k1, k2, k3, sample))

        # Unit direction vectors from sample to each corner
        d1 = k1 - sample;  d1 /= np.linalg.norm(d1)
        d2 = k2 - sample;  d2 /= np.linalg.norm(d2)
        d3 = k3 - sample;  d3 /= np.linalg.norm(d3)

        # Triangular grid of barycentric indices — vectorised
        n_range = np.arange(n + 1)
        i_idx, j_idx = np.meshgrid(n_range, n_range, indexing='ij')
        i_idx, j_idx = i_idx.ravel(), j_idx.ravel()
        mask = i_idx + j_idx <= n
        i_idx, j_idx = i_idx[mask], j_idx[mask]
        k_idx = n - i_idx - j_idx

        alpha = i_idx / n   # weight on k1
        beta  = j_idx / n   # weight on k2
        gamma = k_idx / n   # weight on k3

        # Spherically interpolated directions: normalise(α·d1 + β·d2 + γ·d3)
        dirs = alpha[:, None] * d1 + beta[:, None] * d2 + gamma[:, None] * d3  # (M, 3)
        dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)

        # Vectorised Möller-Trumbore ray-triangle intersection
        # Ray: sample + t * dirs[i],  Triangle: k1 + u*E1 + v*E2
        E1 = k2 - k1            # (3,)
        E2 = k3 - k1            # (3,)
        s_vec = sample - k1     # (3,) — constant across all rays

        h = np.cross(dirs, E2)                       # (M, 3)
        a = np.einsum('ij,j->i', h, E1)             # (M,)

        EPS = 1e-10
        valid = np.abs(a) > EPS
        f = np.where(valid, 1.0 / a, np.inf)        # (M,)

        u = f * np.einsum('ij,j->i', h, s_vec)      # (M,)

        q = np.cross(s_vec, E1)                      # (3,) — constant
        v = f * (dirs @ q)                           # (M,)
        t = f * float(np.dot(E2, q))                 # (M,) — E2·q is a scalar

        valid &= (u >= -EPS) & (v >= -EPS) & (u + v <= 1.0 + EPS) & (t > 0)

        u, v, t = u[valid], v[valid], t[valid]
        points = sample + t[:, None] * dirs[valid]               # (M', 3)
        bary_coords = np.stack([1.0 - u - v, u, v], axis=1)     # (M', 3)

        return points, bary_coords

    def line_intersection(self,p1, p2, q1, q2):
        """
        Finds the intersection of two lines (p1->p2 and q1->q2).
        Returns the intersection point.
        """
        A1 = p2[1] - p1[1]
        B1 = p1[0] - p2[0]
        C1 = A1 * p1[0] + B1 * p1[1]

        A2 = q2[1] - q1[1]
        B2 = q1[0] - q2[0]
        C2 = A2 * q1[0] + B2 * q1[1]

        det = A1 * B2 - A2 * B1
        if det == 0:
            return None  # The lines are parallel
        else:
            x = (B2 * C1 - B1 * C2) / det
            y = (A1 * C2 - A2 * C1) / det
            return np.array([x, y])
        plot.show()

    def rotate_vector(self, v, angle, sample):
        """Rotate vector v by angle (radians) about the origin, then translate by sample."""
        rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)],
                                    [np.sin(angle),  np.cos(angle)]])
        return np.dot(rotation_matrix, v)+sample

    def map_to_ab_line(self,point, A, B, ret_t=False):
        """
        Maps a point to a parameter t in [0, 1] along the line from A to B.
        """
        AB = B - A  # Vector from A to B
        AP = point - A  # Vector from A to the point
        t = np.dot(AP, AB) / np.dot(AB, AB)  # Projection of AP onto AB, normalized by the length of AB
        if ret_t:
            return 1-t
        x = round(t,4)
        y = round(1 - t,4)
        return np.array([y,x])

    def project_known_simplex(self, known_simplex, known_pdf, sample, n_p, tol=1e-4):
        """
        Project the known-simplex PDF onto the omega grid via ray casting.

        For each point on the known simplex, a ray is fired from that point
        through the sample and on to the opposite boundary of the phase field.
        n_p equally spaced points are placed along each ray and assigned the
        simplex PDF value (scaled by 1/ray_length).  The resulting scattered
        values are interpolated onto omega using scipy griddata (linear method).

        The projected ray data are stored on self.projected_simplex and
        self.projected_simplex_p for access by the detail-view visualiser.

        Parameters
        ----------
        known_simplex : ndarray, shape (M, dim)
            Discretised points on the known simplex (output of discretise_*).
        known_pdf : ndarray, shape (M,)
            Normalised Dirichlet likelihood at each simplex point.
        sample : ndarray, shape (dim,)
            Sample composition in constrained space.
        n_p : int
            Number of points per ray.  Higher values reduce interpolation
            artefacts on coarse grids; 50 is a good default.
        tol : float, optional
            Tolerance below which small negative griddata values are clipped
            to zero rather than raising an error.  Default 1e-4.

        Returns
        -------
        ndarray, shape (N,)
            Normalised probability density over the omega grid.

        Raises
        ------
        ZeroPException
            If the projected density is zero everywhere (e.g. the sample lies
            outside the region reachable from the known simplex).
        Exception
            If griddata produces values significantly below zero.
        """
        lines=np.array(self.phase_field._get_constraint_lines())
        A=lines[:,:-1]
        b=-1*lines[:,-1]
        polygon_constraints={
                'A':A,
                'b':b
                }
        #polygon_constraints=
        points=[]
        values=[]
        for x,p in zip(known_simplex,known_pdf):
            line=[]
            direction=sample-x
            if np.linalg.norm(direction) < 1e-10:
                continue   # simplex point coincides with sample — skip
            direction=direction/np.linalg.norm(direction)
            end_point=self.find_intersection(polygon_constraints,x,sample)
            l_ps=[(sample+x/(n_p-1)*(end_point-sample))
                  for x in range(n_p)]
            points+=l_ps
            p=p/np.linalg.norm(sample-end_point)
            values+=[p]*n_p

            #plotting
            '''
            plot=self.make_fig()
            plot.plot_point(sample)
            plot.plot_point(end_point)
            plot.show()
            '''

        #plotting
        self.projected_simplex=points
        self.projected_simplex_p=values

        try:
            p = griddata(points, values, self.phase_field.omega,
                         method='linear', fill_value=0)
        except Exception:
            # Projected points are lower-dimensional (e.g. 2 knowns in 3D space
            # produces coplanar rays). Fall back to nearest-neighbour
            # interpolation, which spreads the density to nearby grid points.
            p = griddata(points, values, self.phase_field.omega,
                         method='nearest', fill_value=0)
        if sum(p)==0:
            raise ZeroPException()
        if np.any(p<0):
            if np.any(p< -1*tol):
                raise Exception('a P was < 0')
            else:
                p[p<0]=0
        if np.isnan(p).any():
                                           
            print('P contained nan')
            p=np.nan_to_num(p)
        return(p/np.sum(p))
                                           


    def rep_samples(self, weights, knowns):
        """Legacy helper: compute weight-averaged known in constrained space."""
        weights=np.array(weights)/sum(weights)
        knowns=[Composition(k) for k in knowns]
        k_s=np.array([[k.get_atomic_fraction(el) for el in self.elements] for k in knowns])
        k_c=self.convert_to_constrained_basis(k_s)
        k=weights[0]*k_c[0]+weights[1]*k_c[1]
        return (k,k_c,k_s,knowns,weights)

    def convert_weights_moletomass(self, phases, weights):
        """Convert mole-fraction weights to mass-fraction weights for the given phases."""
        molar_masses=np.array([phase.weight for phase in phases])
        mass_weights=molar_masses*weights
        weights=np.array(mass_weights)/sum(mass_weights)
        return weights

    def convert_weights_masstomole(self, phases, weights):
        """Convert mass-fraction weights to mole-fraction weights for the given phases."""
        molar_masses=mp.array([phase.weight for phase in phases])
        mole_weights=molar_masses/weights
        weights=mass_weights/sum(mass_weights)
        return weights

    def set_model_parameters(self, k):
        """Legacy: set cube_size, num_lines, and points_per_line from a parameter dict."""
        self.cube_size=k['cube_size']
        self.num_lines=k['num lines']
        self.points_per_line=k['points_per_line']

    def set_next_batch(self, batch_size, method='default', p=None, use_mean=True, cutoff_x=None):
        """Legacy batch-suggestion helper; prefer PICIP.suggest() for new code."""
        next_points=np.empty((batch_size,self.constrained_dim))

        if p is None:
            p=self.values_all

        if use_mean:
            q=self.get_mean()
        else:
            q=self.get_max()
        '''
        pc=self.omega
        matches = np.all(pc == q, axis=1)
        if np.any(matches):
            idx=np.argmax(matches)
        else:
            idx = cKDTree(pc).query(q)[1]
        p_tem=self.values_all
        u_p=p_tem[idx]
        if u_p==0:
            print('tick')
            q=self.get_max()
        else:
            print('untick')
            '''
        if method=='default':
            if np.count_nonzero(p)>=batch_size:
                next_points=self.omega[np.random.choice(
                    range(len(self.omega)),size=batch_size-1,
                    replace=False,p=p)
                ]
            else:
                next_points=self.omega[p.nonzero()]
            next_points=np.append(
                next_points,[q],axis=0)
            self.next_batch=next_points
        elif method=='spread':
            omega_reduced=self.top_probability_mass(self.omega,self.values_all,cutoff_x)
            if len(omega_reduced)>batch_size:
                fixed_points=self.s
                self.find_spreadout_points(batch_size,num_repeats=5,omega=omega_reduced,fixed_sites=fixed_points,use_knowns=False,multicore=False)
                self.next_batch=self.suggested_points
            else:
                self.next_batch=omega_reduced

    def calc_sksig(
        self,
        s_standard,
        knowns,
        weights,
        pred_error=0.02,
        sig_stretch=None,
        make_plotting_df=True,
        make_p=True,
        wt_in_moles=False,
        set_sig=None,
    ):
        """
        Calculates and saves the average known composition and corresponding error matrix.

        Parameters
        ----------
        s_standard : ndarray of shape (n,)
            Coordinates of the sample in the standard basis.
        knowns : list of ndarray
            Coordinates of knowns in the standard basis.
        weights : array_like of shape (m,)
            Weights associated with the knowns.
        pred_error : float or list, optional, default=0.02
            Prediction error for all or each of the knowns.
        sig_stretch : float, optional
            Stretch factor for sigma.
        make_plotting_df : bool, optional, default=True
            Whether or not to create a plotting-ready DataFrame for the knowns.
        make_p : bool, optional, default=True
            Whether or not to create density.
        wt_in_moles : bool, optional, default=False
            Whether the weights are in moles.
        set_sig : float or ndarray, optional
            If provided, this value will be used for sigma.

        Returns
        -------
        None
        """

        self.add_knowns_standard(knowns, make_plotting_df=make_plotting_df)
        knowns = np.array(knowns)
        if knowns.ndim == 1:
            knowns = np.array(knowns)
        # get string rep for knowns
        formulas=self.convert_standards_to_formulas(knowns)
        formulas = []
        for k in knowns:
            rep = ""
            for a, b in zip(k, self.elements):
                rep += b + " " + str(a) + " "
            rep = rep[:-1]
            formulas.append(rep)
        # set pred error
        if not isinstance(pred_error, list):
            pred_error = [pred_error] * len(weights)

        converter = wt_converter()
        moles, moles_error, formulas_standard = converter.wt_to_moles(
            formulas, weights, weights_error=pred_error, in_moles=wt_in_moles
        )
        print(f'{moles=}')
        print(f'{moles_error=}')
        k, sig = self.compute_k_sig(moles, formulas_standard, moles_error)
        if sig_stretch is not None:
            sig *= sig_stretch
        s = self.convert_to_constrained_basis(s_standard)
        if set_sig is not None:
            sig = set_sig
        self.add_sksig(s, k, sig, make_p=make_p)

    def compute_k_sig(self, moles, formulas, moles_error):
        """
        Computes the mean and sigma matrix for a sample.

        Parameters
        ----------
        moles : array_like of shape (m,)
            Relative number of moles for each phase.
        formulas : array_like of shape (m, n)
            Standard representation for each phase.
        moles_error : array_like of shape (m,)
            Estimated error for each phase.

        Returns
        -------
        tuple of ndarray
            Mean in constrained representation and associated sigma matrix.
        """
        moles = np.array(moles)
        moles_error = np.array(moles_error)
        formulas = np.array(formulas)
        formulas_projected = np.array([np.array(x) / sum(x) for x in formulas])
        if len(formulas) != len(moles) or len(moles) != len(moles_error):
            raise Exception(
                "Error, moles,formulas,moles_error of different length"
            )
        # compute average composition in standard rep and associated variance
        merged_mean = np.einsum("i,ij->j", moles, formulas_projected)
        merged_mean_var = np.einsum(
            "i,ij->j", moles_error**2, formulas_projected**2
        )
        # normalise
        merged_mean_std = merged_mean_var**0.5 / np.sum(merged_mean)
        print(merged_mean_std,'std')
        merged_mean /= np.sum(merged_mean)
        # construct sigma
        merged_sigma = np.diag(merged_mean_std**2)
        # convert to constrained
        constrained_mean = self.convert_to_constrained_basis(merged_mean)
        constrained_sigma = self.convert_sigma_to_constrained_basis(
            merged_sigma
        )
        return (constrained_mean, constrained_sigma)

    def convert_sigma_to_constrained_basis(self, sigma):
        """
        Converts the sigma matrix to the constrained basis.

        Parameters
        ----------
        sigma : ndarray of shape (n, n)
            Sigma matrix in the standard representation.

        Returns
        -------
        ndarray
            Sigma matrix in the constrained basis.
        """
        sigma = np.einsum("ij,jk,kl", self.basis, sigma, self.basis.T)
        return sigma

    def add_sksig(self, s, k, sig, make_p=False, exclude_nan=True):
        """
        Adds sample, known, and sigma information to the phase field.

        Parameters
        ----------
        s : ndarray
            Sample point in constrained representation.
        k : ndarray
            Known composition in constrained representation.
        sig : ndarray
            Sigma matrix in constrained representation.
        make_p : bool, optional, default=False
            Whether or not to create a density distribution.
        exclude_nan : bool, optional, default=True
            Whether or not to exclude NaN values in density.

        Returns
        -------
        None
        """
        if make_p:
            p = self.create_density_from_sample(
                s,
                k,
                sig,
                n=self.demi_len,
                points_per_line=self.line_len,
                fib=self.use_fib,
            )
            if exclude_nan:
                if np.isnan(p).any():
                    return
            if hasattr(self, "values_full"):
                self.values_full = np.append(self.values_full, [p], axis=0)
                self.values = np.prod(self.values_full, axis=0)
                self.values = self.values / sum(self.values)
            else:
                self.values_full = p[np.newaxis, :]
                self.values = p
        if hasattr(self, "s"):
            self.s = np.append(self.s, [s], axis=0)
        else:
            self.s = s[np.newaxis, :]
        if hasattr(self, "k"):
            self.k = np.append(self.k, [k], axis=0)
        else:
            self.k = k[np.newaxis, :]
        if hasattr(self, "sig"):
            self.sig = np.append(self.sig, [sig], axis=0)
        else:
            self.sig = sig[np.newaxis, :]

    def create_density_from_sample(
        self,
        sample_point,
        avg_known,
        sigma,
        n=800,
        points_per_line=500,
        fib=False,
        tol=1e-5,
        scale_by_line_length=True,
    ):
        """
        Creates probability density from a sample point and known average.

        Parameters
        ----------
        sample_point : ndarray
            Sample point in constrained representation.
        avg_known : ndarray
            Average known in constrained representation.
        sigma : ndarray
            Sigma matrix for the average known.
        n : int, optional, default=800
            Number of lines to construct to aproximate probability density.
        points_per_line : int, optional, default=500
            Number of points along each line.
        fib : bool, optional, default=False
            Whether or not to use Fibonacci sampling for the hemisphere.
        tol : float, optional, default=1e-5
            Tolerance for avoiding negative densities.
        scale_by_line_length : bool, optional, default=True
            Whether or not to scale the densities by line length.

        Returns
        -------
        ndarray
            Array of probability density values for points in the omega grid.
        """

        uniform_demisphere = None
        if fib:
            uniform_demisphere = self.get_uniform_demisphere_fib(
                sample_point, avg_known, n=n
            )
        else:
            uniform_demisphere = self.get_uniform_demisphere(
                sample_point, avg_known, n=n
            )
        z = []
        points = []
        lines = np.array(self._get_constraint_lines())
        A = lines[:, :-1]
        b = -1 * lines[:, -1]
        polygon_constraints = {"A": A, "b": b}

        for p in uniform_demisphere:
            v = p - sample_point
            intersect = self.find_intersection(
                polygon_constraints, sample_point, p
            )
            line_len = np.linalg.norm(sample_point - intersect)

            f = None
            if self.constrained_dim == 2:
                A = self.get_orthogonal_basis(v).T[1]
                sigma_perp = np.einsum("j,jk,k", A.T, sigma, A)
                y = np.einsum("j,j", A, sample_point - avg_known)
                f = scipy.stats.norm.pdf(y, 0, sigma_perp)
            else:
                A = self.get_orthogonal_basis(v).T[1:]
                sigma_perp = np.einsum("ij,jk,kl", A, sigma, A.T)
                y = np.einsum("ij,j", A, sample_point - avg_known)
                f = multivariate_normal.pdf(
                    y, np.zeros(self.constrained_dim - 1), sigma_perp
                )
            if scale_by_line_length:
                values = [f / line_len] * points_per_line
            else:
                values = [f] * points_per_line
            z.extend(values)
            l_ps = [
                (
                    sample_point
                    + x / (points_per_line - 1) * (intersect - sample_point)
                )
                for x in range(points_per_line)
            ]
            points.extend(l_ps)

        z = np.array(z)
        points = np.array(points)

        self.computed_p = z
        self.computed_omega = points

        if hasattr(self, "interpolation_method"):
            m = self.interpolation_method
        else:
            m = "linear"

        
        p = griddata(
            points, z, self.omega, method=m, fill_value=0
        )

        if m == "nearest":
            print("cutting")
            # Create a Delaunay triangulation of the initial points
            tri = Delaunay(points)

            # Find which points are inside the convex hull of the initial point
            inside_hull = tri.find_simplex(self.omega) >= 0

            # Set values outside the convex hull to 0
            p[~inside_hull] = 0
            print("done")

        if sum(p) == 0:
            raise ZeroPException()
        if np.any(p < 0):
            if np.any(p < -1 * tol):
                raise Exception("a P was < 0")
            else:
                p[p < 0] = 0
        if np.isnan(p).any():
            raise Exception("P contains NAN")
        return p / np.sum(p)

    def get_uniform_demisphere_fib(self, sample_point, avg_known, n=500):
        """
        Generates n uniformly distributed points on the hemisphere using the Fibonacci method.

        Parameters
        ----------
        sample_point : ndarray
            Sample point in constrained representation.
        avg_known : ndarray
            Average known point in constrained representation.
        n : int, optional, default=500
            Number of points to generate on the hemisphere.

        Returns
        -------
        ndarray
            n points "uniformly" distributed on the hemisphere in constrained representation.
        """
        if self.constrained_dim == 3:
            goldenRatio = (1 + 5**0.5) / 2
            i = np.arange(0, n)
            theta = 2 * np.pi * i / goldenRatio
            phi = np.arccos(1 - 2 * (i + 0.5) / n)
            points = np.array(
                [
                    np.cos(theta) * np.sin(phi),
                    np.sin(theta) * np.sin(phi),
                    np.cos(phi),
                ]
            )
            points = points.T
        elif self.constrained_dim == 2:
            theta = np.linspace(0, 2 * np.pi, n)
            points = np.array([np.cos(theta), np.sin(theta)])
            points = points.T

        else:
            print(
                "Warning, golden angle method not implemented for"
                + str(self.constrained_dim + 2)
                + "elements, using regular"
                + " method"
            )
            return self.get_uniform_demisphere(sample_point, avg_known, n)
        points = points / 999 + sample_point
        a = points - sample_point
        b = sample_point - avg_known
        points = np.delete(points, np.where(np.dot(a, b) <= 0), axis=0)
        return points

    def get_uniform_demisphere(self, sample_point, avg_known, n=500):
        """
        Generates n "uniformly" distributed points on the hemisphere.

        Parameters
        ----------
        sample_point : ndarray
            Sample point in constrained representation.
        avg_known : ndarray
            Average known point in constrained representation.
        n : int, optional, default=500
            Number of points to generate on the hemisphere.

        Returns
        -------
        ndarray
            Points uniformly distributed on the hemisphere in constrained representation.
        """
        points = np.random.multivariate_normal(
            np.zeros(len(sample_point)), np.eye(len(avg_known)), size=n
        )
        # Compute the Euclidean norm (L2 norm) for each point
        norms = np.linalg.norm(points, axis=1, keepdims=True)

        # Normalize each point by dividing by its norm
        points = points / norms
        points = points + sample_point
        a = points - sample_point
        b = sample_point - avg_known
        points = np.delete(points, np.where(np.dot(a, b) <= 0), axis=0)
        return points

    def set_end_points_from_s(self):
        """
        Sets the end points of lines drawn from sample points to the edge of the constrained phase field.

        Calculates the end points of each line that extends from a sample point
        in the direction defined by the known composition. Relies on the boundary being 0 of an element.

        Returns
        -------
        None
        """
        self.end_points = []
        for s, k in zip(self.s, self.k):
            point = self.convert_to_standard_basis(s)
            line = point - self.convert_to_standard_basis(k)
            np.any(line == 0)
            index_0 = np.argwhere(np.isclose(line, 0))
            if index_0 != np.argwhere(np.isclose(point, 0)):
                print("Error, 0 index not in same place for point and line")
            line_c = np.delete(line, index_0)
            point_c = np.delete(point, index_0)
            min_steps = 9999
            for pi, vi in zip(point_c, line_c):
                if vi < 0:
                    if pi / (-1 * vi) < min_steps:
                        min_steps = -1 * pi / vi
            final_point = np.add(point, line * min_steps)
            self.end_points.append(
                self.convert_to_constrained_basis(final_point)
            )

    def get_orthogonal_basis(self, estimated_direction):
        """
        Finds a basis orthogonal to the estimated direction.

        Parameters
        ----------
        estimated_direction : ndarray of shape (n,)
            Direction vector used to compute orthogonal basis.

        Returns
        -------
        ndarray of shape (n, n)
            Orthonormal basis where the first column aligns with the estimated direction.
        """
        dim = len(estimated_direction)
        # function to caluclate basis for each sample
        # basis: numpy (dim,dim) array such such that
        # basis matmul (a point) gives the point in
        # representation where first element is size in estimated_direction and
        # subsequent elements are sizes in orthogonal directions
        A = np.transpose(np.array([estimated_direction]))
        for i in range(dim - 1):
            x = self._find_orthonormal(A)
            A = np.hstack((A, np.array([x]).T))
        return A


    def _lambda_for_mass(self, p, mass_level=0.01):
        """
        Return the pdf thresholds λ that enclose the requested probability mass.

        Parameters
        ----------
        p : (m,) ndarray
            Probability–density values living on a *uniform* voxel grid.
            They do **not** need to be normalised.
        mass_levels : tuple of floats
            Desired cumulative-mass levels (must be increasing).

        Returns
        -------
        list[float]
            One λ for each entry in `mass_levels`.  All points with
            p >= λ_k together contain at least `mass_levels[k]` of the total mass.
        """
        # 1️⃣  sort pdf values from high → low
        order       = np.argsort(p)[::-1]
        p_sorted    = p[order]

        # 2️⃣  cumulative *relative* mass
        cum_frac    = np.cumsum(p_sorted)
        cum_frac   /= cum_frac[-1]          # normalise to 1.0

        # 3️⃣  locate first index where cumulative ≥ desired mass
        cut_lambda = p_sorted[np.searchsorted(cum_frac, mass_level)]
        return cut_lambda

    def resample_p(self, p, n_sample_points=200, omega=None, quant=0.8,method='nearest'):
        """
        Interpolates the probability density onto a new grid

        Grid is generated by Phase_Field.sample_points_on_hull, a uniform distribution
        on the convex hull of the top 'quant' quantile of omega by probability density

        Parameters
        ----------
        p : ndarray of shape (m,)
            Density values corresponding to the `omega` points.
        n_sample_points : int, optional, default=200
            Number of sample points to generate.
        omega : ndarray of shape (m, n), optional
            Grid of points. If not provided, uses the existing `omega` grid.
        quant : float, optional, default=0.8
            Quantile value for selecting the top points in density distribution.

        Returns
        -------
        tuple of ndarray
            High-density points, sampled points on the hull, and interpolated density values.
        """
        if omega is None:
            omega = self.omega

        if len(p) != len(omega):
            raise ValueError("The lengths of p and omega must be the same")

        cut_lambda=self._lambda_for_mass(p,quant)
        high_mask=p>=cut_lambda
        # Calculate the number of top elements to select
        n_top = int(np.ceil(len(p) * (quant)))

        # Get the indices of the top n elements in p
        top_indices = np.argsort(p)[-n_top:]

        # Extract the corresponding elements from omega and p
        omega_high = omega[high_mask]
        p_high = p[high_mask]

        print(f"Fraction cut = {len(omega_high)/len(omega):.0%}")
        print(
            f"Included zeros = {np.count_nonzero(p_high==0)/len(p_high):.0%}"
        )

        hull = ConvexHull(omega_high)

        # Sample n points on the convex hull
        sampled_points = self.sample_points_on_hull(hull, n_sample_points)

        # Interpolate the probabilities to the sampled points
        print("interpolating")
        print(
            f"Interpolating. Omega is length {len(omega)} and new points is"
            + f" length {len(sampled_points)}"
        )
        interpolated_p = griddata(omega, p, sampled_points, method=method)
        print("done")
        return omega_high, p_high, sampled_points, interpolated_p

    def sample_points_on_hull(self, hull, n_points):
        """
        Samples points uniformly on the surface of a convex hull.

        Parameters
        ----------
        hull : scipy.spatial.ConvexHull
            Convex hull of points in `omega` space.
        n_points : int
            Number of points to sample on the hull surface.

        Returns
        -------
        ndarray
            Points sampled uniformly on the surface of the convex hull.
        """
        print("actual sampling")
        sampled_points = []

        # Get vertices and simplices (triangular faces) from the hull
        simplices = hull.simplices

        # Calculate areas of the triangles to use for weighted sampling
        areas = []
        for simplex in simplices:
            a = hull.points[simplex[0]]
            b = hull.points[simplex[1]]
            c = hull.points[simplex[2]]
            area = 0.5 * np.linalg.norm(np.cross(b - a, c - a))
            areas.append(area)
        areas = np.array(areas)
        total_area = np.sum(areas)
        areas /= total_area
        for simplex, areas in zip(simplices, areas):
            points = int(areas * n_points)
            a = hull.points[simplex[0]]
            b = hull.points[simplex[1]]
            c = hull.points[simplex[2]]
            for _ in range(points):
                r1, r2 = np.random.rand(2)
                if r1 + r2 > 1:
                    r1, r2 = 1 - r1, 1 - r2
                point = (1 - r1 - r2) * a + r1 * b + r2 * c
                sampled_points.append(point)
        print("Done")
        print(len(simplices))
        return np.array(sampled_points)

    def get_mean(self, p=None, omega=None):
        """
        Computes the mean of the distribution over `omega` space.

        Parameters
        ----------
        p : ndarray of shape (m,), optional
            Density values corresponding to `omega` points.
            If not provided, uses the existing `values` attribute.
        omega : ndarray of shape (m, n), optional
            Grid of points. If not provided, uses the existing `omega` grid.

        Returns
        -------
        ndarray
            Mean point in constrained representation.
        """
        if omega is None:
            omega = self.omega
        if p is None:
            p = self.values_all
        p = p / np.sum(p)
        u = (
            np.swapaxes(np.broadcast_to(p, (omega.shape[1], len(p))), 0, 1)
            * omega
        )
        mean = np.sum(u, axis=0)
        return mean

    def get_max(self, p=None, omega=None):
        """
        Finds the point with the maximum density value.

        Parameters
        ----------
        p : ndarray of shape (m,), optional
            Density values corresponding to `omega` points.
            If not provided, uses the existing `values` attribute.
        omega : ndarray of shape (m, n), optional
            Grid of points. If not provided, uses the existing `omega` grid.

        Returns
        -------
        ndarray
            Point in constrained representation with the maximum density value.
        """
        if omega is None:
            omega = self.omega
        if p is None:
            p = self.values_all
        mean = self.omega[np.argmax(p)]
        return mean

    def sample_along_line(self, n, point_index=-1):
        """
        Samples points along the line from the point_index sample to the edge of the space.

        Adds points to self.suggested_points
        Parameters
        ----------
        n : int
            Number of points to sample along the line.
        point_index : int, optional, default=-1
            Which sample to use

        Returns
        -------
        None
        """
        self.set_end_points_from_s()
        a = self.s[point_index]
        b = self.end_points[point_index]
        samples = []
        for i in range(n):
            samples.append(a + ((i + 1) / n) * (b - a))
        self.suggested_points = samples

    def print_suggested_points(self):
        """
        Prints suggested points in the standard representation.

        Returns
        -------
        None
        """
        s_c = self.convert_to_standard_basis(self.suggested_points)
        s = self.get_composition_strings(s_c, out="txt")
        print("Suggested points")
        for i in s:
            print(i)

    def get_suggested_html(self, prec=2):
        """Return an HTML table of suggested points in standard (elemental fraction) basis."""
        p_s=self.convert_to_standard_basis(self.suggested_points)
        df=pd.DataFrame()
        for n,el in enumerate(self.elements):
            df[el]=np.round(p_s[:,n],prec)
        return df.to_html(escape=False, index=False)

    def find_intersection(self, polygon_constraints, x, sample, epsilon=1e-2):
        """
        Finds the intersection of a line with the edges of a polygon.

        Parameters
        ----------
        polygon_constraints : dict
            Dictionary with keys "A" (linear coefficients) and "b" (constant terms)
            representing the linear inequalities of the polygon.
        x : ndarray
            Starting point of the line.
        sample : ndarray
            Direction point of the line.
        epsilon : float, optional, default=1e-2
            Minimum distance threshold to avoid numerical precision issues.

        Returns
        -------
        ndarray or None
            Intersection point closest to `x` if found; otherwise, `None`.
        """
        intersections = []

        # Parametric form of the line: x + t(sample - x), for t >= 0.
        direction = sample - x

        # plotting
        """
        plot=self.make_fig()
        plot.plot_point(x)
        plot.plot_point(sample)
        plot.plot_point(sample+d)
        plot.show()
        """

        # Iterate through each edge of the polygon,
        # given by the inequalities Ax <= b
        for i, (A_row, b_val) in enumerate(
            zip(polygon_constraints["A"], polygon_constraints["b"])
        ):
            # Find t such that A @ (x + t * (sample - x)) =
            # b (i.e., intersection)
            # This simplifies to A @ direction * t = b - A @ x
            A_dot_dir = np.dot(A_row, direction)

            if (
                abs(A_dot_dir) > 1e-6
            ):  # Only if the direction and edge are not parallel
                t_intersect = (b_val - np.dot(A_row, x)) / A_dot_dir

                # We only care about positive t values (forward in direction)
                if t_intersect > 1e-6:
                    # Compute intersection point
                    intersection_point = self.line_through_point(
                        x, sample, t_intersect
                    )
                    if np.linalg.norm(intersection_point - x) > epsilon:
                        intersections.append(intersection_point)

        # Return the intersection point closest to x
        if len(intersections) > 0:
            distances = [
                np.linalg.norm(intersection - x)
                for intersection in intersections
            ]
            closest_intersection = intersections[np.argmin(distances)]
            return closest_intersection
        else:
            print("no intersection found")
            # No intersection found
            # (this should not happen in a bounded polygon)
            return None

    def line_through_point(self, x, sample, t):
        """
        Calculates a point along a parametric line.

        Parameters
        ----------
        x : ndarray
            Starting point of the line.
        sample : ndarray
            Direction point of the line.
        t : float
            Parameter used to calculate the intersection point along the line.

        Returns
        -------
        ndarray
            Point along the line at parameter `t`.
        """
        return x + t * (sample - x)

    def top_probability_mass(self, omega, p, x):
        """
        Select coordinates from omega whose cumulative probability mass
        reaches the top x% of the total probability.

        Parameters
        ----------
        omega : (N, 2) ndarray
            Array of 2D coordinates.
        p : (N,) ndarray
            Array of probabilities associated with each coordinate.
            Should sum to 1 (but will be normalised if not).
        x : float
            Percentage (0–100) of total probability mass to retain.

        Returns
        -------
        omega_reduced : (M, 2) ndarray
            Coordinates corresponding to the top x% probability mass.
        p_reduced : (M,) ndarray
            Corresponding probabilities.
        """

        omega = np.asarray(omega)
        p = np.asarray(p, dtype=float)
        p = p / p.sum()  # normalise to ensure total = 1

        # Sort by descending probability
        sort_idx = np.argsort(p)[::-1]
        omega_sorted = omega[sort_idx]
        p_sorted = p[sort_idx]

        # Compute cumulative sum
        cumsum = np.cumsum(p_sorted)

        # Keep points until cumulative probability >= x%
        cutoff = x / 100.0
        keep_mask = cumsum <= cutoff
        # Include the first point above cutoff (to guarantee ≥ x%)
        if not np.all(keep_mask):
            first_above = np.argmax(~keep_mask)
            keep_mask[first_above] = True

        omega_reduced = omega_sorted[keep_mask]

        return omega_reduced

DEFAULT_PREDICTED_ERROR = 0.2  # σ ≈ 0.2; TODO: validate via simulation study


class Sample:
    """
    Container for one measured sample and its co-existing known phases.

    Construct with a name and a composition string, then call add_knowns(),
    followed by either add_mass_weights() (primary, for Rietveld output) or
    add_molar_weights() (alternative).  Call set_predicted_error() to override
    the default uncertainty before passing the sample to PICIP.add_sample().

    Parameters
    ----------
    name : str
        Human-readable label for the sample (used in plot titles and legends).
    composition : str
        Overall measured sample composition in any format recognised by pymatgen
        (e.g. ``'Mg0.3Al0.5Sn0.2'``).

    Attributes
    ----------
    name : str
    composition : pymatgen.core.Composition
    predicted_error : float
        Dirichlet concentration parameter σ.  Defaults to DEFAULT_PREDICTED_ERROR.
        Set by set_predicted_error().
    knowns : list of pymatgen.core.Composition
        Set by add_knowns().
    weights : ndarray
        Mole fractions of the known phases (normalised).  Set by add_mass_weights()
        or add_molar_weights().
    mass_weights : ndarray
        Mass fractions of the known phases (normalised).  Set by add_mass_weights()
        or add_molar_weights().
    comp_constrained : ndarray
        Sample composition in constrained basis.  Set by PICIP.add_sample().
    comp_standard : ndarray
        Sample composition in standard (elemental fraction) basis.  Set by
        PICIP.add_sample().
    knowns_constrained : ndarray, shape (n_knowns, dim)
        Known phase compositions in constrained basis.  Set by PICIP.add_sample().
    knowns_standard : ndarray, shape (n_knowns, nelements)
        Known phase compositions in standard basis.  Set by PICIP.add_sample().
    average_k : ndarray
        Weight-averaged known composition in constrained basis.  Set by
        PICIP.add_sample().
    """

    def __init__(self, name, composition):
        self.name = name
        self.composition=Composition(composition)
        # predicted_error defaults to DEFAULT_PREDICTED_ERROR; override with
        # set_predicted_error() before running PICIP.
        self.predicted_error=DEFAULT_PREDICTED_ERROR

    def add_knowns(self, knowns):
        """
        Register the co-existing known phases for this sample.

        Parameters
        ----------
        knowns : list of str
            Phase compositions in any format recognised by pymatgen, e.g.
            ``['MgS', 'Al2S3']``.  Order must match the weights supplied to
            add_mass_weights() or add_molar_weights().
        """
        self.knowns=[Composition(k) for k in knowns]

    def add_molar_weights(self, mol_weights):
        """
        Set known-phase weights from molar (mole-fraction) quantities.

        Use add_mass_weights() instead if your phase fractions come from Rietveld
        refinement (which reports mass fractions directly).

        Parameters
        ----------
        mol_weights : array-like, shape (n_knowns,)
            Relative molar amounts for each known phase.  Need not be normalised.
            Order must match the list passed to add_knowns().
        """
        self.weights=np.array(mol_weights)/sum(mol_weights)
        molar_masses=np.array([known.weight for known in self.knowns])
        # Derive mass fractions from mole amounts × molar mass
        mass_weights=molar_masses*np.array(mol_weights)
        self.mass_weights=mass_weights/sum(mass_weights)

    def add_mass_weights(self, mass_weights):
        """
        Set known-phase weights from mass-fraction quantities (primary method).

        This is the primary weight-setting method and the one most commonly
        used in practice, since Rietveld refinement reports mass fractions.
        Internally converts to mole fractions (weights) by dividing each
        mass weight by the phase's molar mass.

        Parameters
        ----------
        mass_weights : array-like, shape (n_knowns,)
            Relative mass amounts for each known phase (e.g. Rietveld scale
            factors).  Need not be normalised.  Order must match add_knowns().
        """
        molar_masses = np.array([known.weight for known in self.knowns])
        weights = np.array(mass_weights) / molar_masses   # moles = mass / M
        self.weights = weights / sum(weights)
        self.mass_weights = np.array(mass_weights) / sum(mass_weights)

    def set_predicted_error(self, predicted_error, prior=1e-4):
        """
        Set the predicted measurement uncertainty (Dirichlet concentration σ).

        Larger values of predicted_error produce a broader, more diffuse
        probability density.  The default (DEFAULT_PREDICTED_ERROR = 0.2) is a
        reasonable starting point; see the memory file for calibration notes.

        Parameters
        ----------
        predicted_error : float
            Standard deviation of the Dirichlet likelihood.  Typical range is
            0.05–0.5; values below ~0.05 can cause near-zero densities for
            noisy data.
        prior : float, optional
            Uniform prior weight added to each simplex point before normalisation.
            The default 1e-4 keeps the density from becoming exactly zero in
            regions the simplex does not reach.  Rarely needs changing.
        """
        self.predicted_error=predicted_error
        # Note: the prior argument is accepted for API compatibility but the
        # value passed here is always overridden to 1e-4.  To use a custom
        # prior, set sample.prior directly after calling this method.
        self.prior=1e-4

    

class Prediction:
    """
    Result of a PICIP inference run.

    Created and returned by PICIP.run().  All probability values are defined
    over the same omega grid as the parent Phase_Field.

    Attributes
    ----------
    name : str
        Label for this prediction (sample name or 'combined probability').
    prob_density : ndarray, shape (N,)
        Normalised probability density over the omega grid.  Sums to 1.
    samples : list of Sample
        The Sample objects that contributed to this prediction.
    individual_densities : list of ndarray or None
        Per-sample normalised densities (only populated for multi-sample runs).
        Each entry has the same shape as prob_density.  None for single-sample runs.
    suggestions : Suggestions or None
        Set by PICIP.suggest() after calling that method; None until then.
    """

    def __init__(self, name, prob_density, samples, individual_densities=None):
        self.name = name
        self.prob_density = prob_density
        self.samples = samples
        self.individual_densities = individual_densities  # list of per-sample normalised densities, or None
        self.suggestions = None  # set by PICIP.suggest()
