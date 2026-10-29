"""
PICIP — Probabilistic Isolation of Inorganic Crystalline Phases
===============================================================
Bayesian inference of an unknown phase composition from measured XRD data.

Four-step workflow
------------------
1. **Set up a phase field** — choose your element system and grid::

       from phase_field import Phase_Field
       pf = Phase_Field()
       pf.setup_uncharged(["Fe", "Mn", "Ti"])          # 2-D constrained space
       # or: pf.setup_charged({"Fe": 3, "O": -2})
       # or: pf.setup_charge_ranges({"Fe": [2,3], "O": -2})
       # or: pf.setup_custom(...)

2. **Create and add Sample objects** — supply the measured composition, the
   known co-existing phases identified by XRD, and their Rietveld mass
   fractions::

       from picip import PICIP, Sample
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
import warnings

import numpy as np
import scipy.stats
from scipy.stats import dirichlet, multivariate_normal
from scipy.interpolate import griddata
from pymatgen.core import Composition
import pandas as pd

from picip import utils
from picip.exceptions import ZeroPException
from picip.utils.composition import composition_to_standard_basis, atom_count


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
    composition : str or pymatgen.core.Composition
        Overall measured sample composition in any format recognised by pymatgen
        (e.g. ``'Mg0.3Al0.5Sn0.2'``).

    Attributes
    ----------
    name : str
        Human-readable label (used in plot titles and legends).
    composition : pymatgen.core.Composition
        Parsed sample composition.
    predicted_error : float
        Dirichlet concentration parameter σ.  Defaults to DEFAULT_PREDICTED_ERROR.
        Override with set_predicted_error() before passing to PICIP.
    prior : float
        Uniform prior weight added to each simplex point before normalisation.
        Default 1e-4.  Set by set_predicted_error(); rarely needs changing.
    knowns : list of pymatgen.core.Composition or None
        Co-existing known phases.  Set by add_knowns().
    phase_fractions : ndarray or None
        Mole fractions of the known phases.
        Set by add_mass_weights() or add_molar_weights().
    mass_weights : ndarray or None
        Mass fractions of the known phases.
        Set by add_mass_weights() or add_molar_weights().
    comp_constrained : ndarray or None
        Sample composition in constrained basis.  Set by PICIP.add_sample().
    knowns_constrained : ndarray, shape (n_knowns, dim) or None
        Known phase compositions in constrained basis.  Set by PICIP.add_sample().
    knowns_standard : ndarray, shape (n_knowns, nelements) or None
        Known phase compositions in standard (elemental fraction) basis.
        Set by PICIP.add_sample().
    average_k : ndarray or None
        Weight-averaged known composition in constrained basis.
        Set by PICIP.add_sample().
    _simplex_points : ndarray or None
        Discretised points on the known simplex from the most recent inference run.
        Set by PICIP._compute_p_dirichlet().
    _simplex_pdf : ndarray or None
        Dirichlet likelihood at each simplex point.
        Set by PICIP._compute_p_dirichlet().
    _projected_points : ndarray or None
        Ray points projected from the simplex through the sample to the boundary.
        Set by PICIP._compute_p_dirichlet().
    _projected_p : ndarray or None
        Probability values along the projected rays.
        Set by PICIP._compute_p_dirichlet().
    _n_p : int or None
        Number of interpolation points per ray used in the last run.
        Set by PICIP._compute_p_dirichlet().
    _n_rays : int or None
        Number of discretised known-simplex points used in the last run.
        Set by PICIP._compute_p_dirichlet().
    _n_rays_emitted : int or None
        Number of projection rays actually fired, which is <= _n_rays once
        degenerate rays are skipped, and larger than it on the 2-D path where
        extra corner-derived rays are added.  The first
        ``_n_rays_emitted * _n_p`` entries of _projected_points are the ray
        block, laid out ray-major; anything after that is anchor points.
        Set by PICIP._compute_p_dirichlet().
    """

    def __init__(self, name, composition):
        self.name = name
        self.composition = Composition(composition)
        self.predicted_error = DEFAULT_PREDICTED_ERROR
        self.prior = 1e-4
        self.knowns = None
        self.phase_fractions = None
        self.mass_weights = None
        self.comp_constrained = None
        self.knowns_constrained = None
        self.knowns_standard = None
        self.average_k = None
        self._simplex_points = None
        self._simplex_pdf = None
        self._projected_points = None
        self._projected_p = None
        self._n_p = None
        self._n_rays = None
        self._n_rays_emitted = None

    def add_knowns(self, knowns):
        """
        Register the co-existing known phases for this sample.

        Parameters
        ----------
        knowns : list of str
            Phase compositions in any format recognised by pymatgen, e.g.
            ``['MgS', 'Al2S3']``.  Order must match the phase_fractions supplied to
            add_mass_weights() or add_molar_weights().
        """
        if not knowns:
            raise ValueError("knowns must not be empty.")
        self.knowns=[Composition(k) for k in knowns]

    def _atom_counts(self):
        """
        Count atoms in each known phase.

        Returns
        -------
        np.ndarray : Numpy array of shape (n_knowns,)
            Total number of atoms per formula unit for each known phase.
        """
        return atom_count(self.knowns)

    def add_molar_weights(self, mol_weights):
        """
        Set known-phase phase_fractions from molar (mole-fraction) quantities.

        Use add_mass_weights() instead if your phase fractions come from Rietveld
        refinement (which reports mass fractions directly).

        Parameters
        ----------
        mol_weights : array-like, shape (n_knowns,)
            Relative molar amounts for each known phase.  Need not be normalised.
            Order must match the list passed to add_knowns().
        """
        if len(mol_weights) != len(self.knowns):
            raise ValueError(
                f"mol_weights length ({len(mol_weights)}) does not match "
                f"knowns length ({len(self.knowns)})."
            )
        if any(w <= 0 for w in mol_weights):
            raise ValueError("All mol_weights must be positive.")
        scaled_weights = np.array(mol_weights) * self._atom_counts()
        self.phase_fractions= scaled_weights / sum(scaled_weights)
        molar_masses=np.array([known.weight for known in self.knowns])
        # Derive mass fractions from mole amounts × molar mass
        mass_weights=molar_masses*np.array(mol_weights)
        self.mass_weights=mass_weights/sum(mass_weights)

    def add_mass_weights(self, mass_weights):
        """
        Set known-phase phase_fractions from relative mass quantities (primary method).

        This is the primary weight-setting method and the one most commonly
        used in practice, since Rietveld refinement reports mass fractions.
        Internally converts to relative amounts of substance (rel_amounts_of_substance)
        by dividing each mass weight by the phase's molar mass.

        Parameters
        ----------
        mass_weights : array-like, shape (n_knowns,)
            Relative mass amounts for each known phase (e.g. Rietveld scale
            factors).  Need not be normalised.  Order must match add_knowns().
        """
        if len(mass_weights) != len(self.knowns):
            raise ValueError(
                f"mass_weights length ({len(mass_weights)}) does not match "
                f"knowns length ({len(self.knowns)})."
            )
        if any(w <= 0 for w in mass_weights):
            raise ValueError("All mass_weights must be positive.")
        molar_masses = np.array([known.weight for known in self.knowns])
        # This is moles of individual atoms
        # moles = mass / M
        rel_amounts_of_substance = np.array(mass_weights) * self._atom_counts() / molar_masses
        # Moles of formula units in phase i / Sum over j moles of formula units in phase j
        self.phase_fractions = rel_amounts_of_substance / sum(rel_amounts_of_substance)
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
        if predicted_error <= 0:
            raise ValueError(f"predicted_error must be positive, got {predicted_error}.")
        self.predicted_error = predicted_error
        self.prior = prior

    



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

    def print_table(self, prec=3):
        """
        Print the suggested compositions as an aligned table.

        One row per suggestion: its label ('mean' or 'sampled') followed by the
        element fractions, using whichever elements the phase field was built
        from.

        Parameters
        ----------
        prec : int, optional, default=3
            Decimal places shown for each element fraction.
        """
        width = max((len(label) for label in self.labels), default=0)
        for label, row in zip(self.labels, self.standard):
            fractions = "  ".join(
                f"{el}={v:.{prec}f}" for el, v in zip(self.elements, row)
            )
            print(f"  {label:{width}s}  {fractions}")

    def save(self, path):
        """Save suggestions to a CSV file with elemental fraction columns."""
        df = pd.DataFrame(self.standard, columns=self.elements)
        df.insert(0, 'type', self.labels)
        df.to_csv(path, index=False)
        print(f"Saved {len(df)} suggestion points to {path}")




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

    Attributes
    ----------
    phase_field : Phase_Field
        The phase field within which inference is performed.
    samples : list of Sample
        Samples registered via add_sample().
    projected_simplex : list or None
        Ray points produced by the most recent _project_known_simplex call.
        Overwritten on each call; copied onto the Sample object immediately after.
    projected_simplex_p : list or None
        Ray probability values corresponding to projected_simplex.
        Overwritten on each call; copied onto the Sample object immediately after.
    """

    def __init__(self, phase_field):
        self.phase_field = phase_field
        self.samples = []
        self.projected_simplex = None
        self.projected_simplex_p = None

    def _warn_if_outside_field(self, comp_standard, label):
        """
        Warn if a standard-basis composition violates the phase field's constraints.

        Parameters
        ----------
        comp_standard : ndarray of shape (nelements,)
            Composition in standard (element-fraction) representation.
        label : str
            How to name the composition in the warning, e.g. ``"sample s1"``.
        """
        failures = self.phase_field.constraint_failures(comp_standard)
        if failures:
            print(
                f"Warning, {label} " + "; ".join(failures)
                + " — it will be projected onto the phase field, so the result "
                  "will describe a slightly different composition."
            )

    def add_sample(self, sample):
        """
        Validate a Sample and register it for inference.

        Converts the sample composition and all known phases into both the
        constrained and standard bases, computes the weight-averaged known
        position (average_k), and appends the sample to self.samples.

        The sample composition and every known are checked against the phase
        field's constraints (charge neutrality, charge ranges, custom
        constraints) *before* conversion, since converting to the constrained
        basis projects any violation away.  A violation is reported as a
        warning, not an exception — measured compositions often sit slightly
        off the constraint surface — but the inference then describes the
        projected composition rather than the one supplied.

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
        if sample.knowns is None:
            raise Exception(f"Sample {sample.name} has no knowns — call add_knowns() first")
        if sample.phase_fractions is None:
            raise Exception(
                f"Sample {sample.name} has no phase_fractions — call add_mass_weights() "
                "or add_molar_weights() first"
            )
        field_elements = set(self.phase_field.elements)
        # Set difference, not '>': a strict-superset test only fires when the
        # sample happens to mention every element in the field as well.
        extra = set([x.symbol for x in sample.composition.elements]) - field_elements
        if extra:
            raise Exception(
                f"Sample {sample.name} contains elements not in phase field: {sorted(extra)}"
            )
        comp_standard=np.array(composition_to_standard_basis(sample.composition, self.phase_field.elements))
        # Check in the standard basis, before converting.  The constrained basis
        # spans only the subspace satisfying the constraints, so converting
        # projects any violation away — this is the last point at which a
        # composition outside the phase field can still be detected.
        self._warn_if_outside_field(comp_standard, f"sample {sample.name}")
        sample.comp_constrained=self.phase_field.convert_to_constrained_basis(comp_standard)
        for k in sample.knowns:
            extra = set([x.symbol for x in k.elements]) - field_elements
            if extra:
                raise Exception(
                    f"Known phase {k} contains elements not in phase field: {sorted(extra)}"
                )

        # Rows: known phase; columns: element
        # Fractions of  atoms in phase i which are element j
        sample.knowns_standard=composition_to_standard_basis(sample.knowns, self.phase_field.elements)
        for k, k_s in zip(sample.knowns, sample.knowns_standard):
            self._warn_if_outside_field(k_s, f"known {k.reduced_formula}")

        # Convert to array of constrained-space vectors
        # Rows: knowns; columns: constrained-space basis vectors
        sample.knowns_constrained=self.phase_field.convert_to_constrained_basis(sample.knowns_standard)

        # Column-wise dot products giving average k as a row vector
        sample.average_k=np.sum(sample.phase_fractions[:,np.newaxis] * sample.knowns_constrained, axis=0)

        if sample.predicted_error is None:
            raise Exception(f"Predicted error has not been set for sample {sample.name}")
        self.samples.append(sample)
                                
    @_requires_samples
    def run(self, sample_indexes=None, version='gaussian', n_l=50, n_p=50, name=None):
        """
        Run PICIP inference and return a Prediction.

        For a single sample the returned probability density is the one produced
        by _compute_p_dirichlet.  For multiple samples the densities are
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
            Inference method.  ``'gaussian'`` (default) is the original PICIP
            algorithm from the paper — the well-tested path, valid for any
            number of knowns in any dimension.

            ``'dirichlet'`` opts into the Dirichlet simplex method, which is
            implemented for exactly two configurations: 2-D with 2 known
            phases, and 3-D with 3 known phases.  Requesting it anywhere else
            emits a ``RuntimeWarning`` and falls back to Gaussian.  The warning
            only appears when ``'dirichlet'`` was asked for explicitly — the
            default never warns, since Gaussian is valid everywhere.
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

        if version not in ('dirichlet', 'gaussian'):
            raise ValueError(f"Unknown version '{version}'. Use 'dirichlet' or 'gaussian'.")

        # Pick the density function once, so every later recomputation (the
        # zero-density retry below) stays on the method the caller asked for.
        compute = (self._compute_p_dirichlet if version == 'dirichlet'
                   else self._compute_p_gaussian)

        p_densities = [compute(sample, n_l, n_p) for sample in samples]
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
                    p_densities = [compute(s, n_l, n_p) for s in samples]
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
        if n < 1:
            raise ValueError(f"n must be at least 1, got {n}.")
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

    def suggest_kmedoids(self, prediction, n, past_samples=None,
                         exclude_radius=None, max_candidates=1000):
        """
        Bayes-optimal batch for the metric  E_p[ min_i ||c_i - x*|| ].

        Greedy weighted k-medoids over the omega grid: at each step pick the
        omega point that minimises the posterior-weighted expected distance
        to the *closest* already-chosen point.  Equivalent to greedy facility
        location with the posterior as customer phase_fractions — a 2-approximation
        to the optimal batch under this objective.

        Parameters
        ----------
        prediction : Prediction
        n : int
            Batch size.
        past_samples : list of ndarray or None
            Past sample compositions (constrained basis) treated as already-
            chosen — the batch is selected to *complement* them.  If None,
            uses the input samples on the prediction.
        exclude_radius : float or None
            If set, omega points within this radius of any past sample are
            removed from candidacy (not just from the cost function).
        max_candidates : int, optional
            Restrict the candidate pool to the top-``max_candidates`` omega
            points by posterior probability.  The optimal medoid sits in a
            high-probability region, so this is a near-lossless speedup for
            large omega grids.  Set to ``None`` to disable.
        """
        from scipy.spatial.distance import cdist
        if n < 1:
            raise ValueError(f"n must be at least 1, got {n}.")
        p     = prediction.prob_density.astype(float).copy()
        s     = p.sum()
        if s <= 0:
            raise ZeroPException()
        p    /= s
        omega = self.phase_field.omega                              # (N, dim)
        N     = len(omega)

        if past_samples is None:
            past = [s.comp_constrained for s in prediction.samples]
        else:
            past = [np.asarray(x) for x in past_samples]

        # Candidate pool: top-K omega points by posterior probability.
        if max_candidates is not None and max_candidates < N:
            cand_indices = np.argpartition(-p, max_candidates)[:max_candidates]
        else:
            cand_indices = np.arange(N)

        # Optional radius exclusion against past samples
        if exclude_radius is not None and past:
            d_past = cdist(omega[cand_indices], np.array(past)).min(axis=1)
            cand_indices = cand_indices[d_past >= exclude_radius]

        # Distance from every omega point to its nearest past sample
        if past:
            d_prev = cdist(omega, np.array(past)).min(axis=1)
        else:
            d_prev = np.full(N, np.inf)

        # Distances candidate→all-omega: a single (K, N) matrix reused each step.
        D_cand_omega = cdist(omega[cand_indices], omega)            # (K, N)
        cand_alive = np.ones(len(cand_indices), dtype=bool)

        chosen_idx = []
        for _ in range(n):
            alive = np.where(cand_alive)[0]
            if len(alive) == 0:
                break
            # Cost(c) = Σ_x p(x) · min(d_prev(x), D[c, x])
            new_min = np.minimum(d_prev[None, :], D_cand_omega[alive])
            cost = new_min @ p
            best_local = alive[int(np.argmin(cost))]
            best = int(cand_indices[best_local])
            chosen_idx.append(best)
            cand_alive[best_local] = False
            d_prev = np.minimum(d_prev, D_cand_omega[best_local])

        labels = ['kmedoid'] * len(chosen_idx)
        suggestions = Suggestions(omega[chosen_idx], self.phase_field, labels)
        prediction.suggestions = suggestions
        return suggestions

    def _compute_p_dirichlet(self, sample, n_l, n_p, floor=1e-6):
        """
        Compute the probability density for one sample.

        Dispatch:

        * 2-D constrained space with exactly 2 knowns → Dirichlet line method
          (:meth:`_discretise_known_simplex_2d` + :meth:`_map_to_ab_line`
          + :meth:`_project_known_simplex`).
        * 3-D constrained space with exactly 3 knowns → Dirichlet triangle
          method (:meth:`_compute_p_dirichlet_3d`) — the same geometry one
          dimension up.
        * Every other combination of dimension and known count → emits a
          ``RuntimeWarning`` that the Dirichlet method is untested there, and
          falls back to :meth:`_compute_p_gaussian` (Fibonacci demisphere of
          rays from the sample, Gaussian-weighted by perpendicular distance to
          the centroid of the knowns).

        The Gaussian fallback uses ``sample.average_k`` (the molar-weighted
        centroid of the known compositions) as the single effective known —
        valid for any m and any dim, at the cost of treating the known
        simplex as a point.  Pass ``version='gaussian'`` to ``run`` to select it
        deliberately, without the warning.

        Parameters
        ----------
        sample : Sample
            Must have been registered with add_sample().
        n_l : int
            Number of ray directions.
        n_p : int
            Points per projection ray.
        floor : float, optional
            Minimum value used to avoid zero-valued Dirichlet parameters in
            the 2-D / 2-knowns path.  Default 1e-6.

        Returns
        -------
        ndarray, shape (N,)
            Normalised probability density over omega.
        """
        weights=sample.phase_fractions
        k_c=sample.knowns_constrained
        s=sample.comp_constrained
        mass_weights=sample.mass_weights
        predicted_error=sample.predicted_error

        dim = self.phase_field.constrained_dim
        m   = len(k_c)

        # Dispatch:
        #   2-D + exactly 2 knowns  → Dirichlet over the k1–k2 line segment.
        #   3-D + exactly 3 knowns  → Dirichlet over the k1–k2–k3 triangle
        #                            (Möller-Trumbore discretisation): the same
        #                            geometry one dimension up.
        #   everything else         → Gaussian fallback (perpendicular-distance
        #                            kernel from the sample to the known
        #                            centroid; well-defined for any m, any dim).
        if dim == 3 and m == 3:
            return self._compute_p_dirichlet_3d(sample, n_l, n_p, floor=floor)
        if not (dim == 2 and m == 2):
            warnings.warn(
                f"The Dirichlet method (version='dirichlet') is implemented only for "
                f"2-D with 2 known phases and 3-D with 3 known phases. Sample "
                f"{sample.name!r} is {dim}-D with {m} known phase"
                f"{'' if m == 1 else 's'}, which is untested — falling back to "
                f"the Gaussian method. Pass version='gaussian' to select it "
                f"explicitly and silence this warning.",
                RuntimeWarning, stacklevel=3)
            return self._compute_p_gaussian(sample, n_l, n_p)

        # m == 2, dim == 2: existing Dirichlet line method
        line_points = self._discretise_known_simplex_2d(k_c[0], k_c[1], s, n_l)
        normalised_points = np.array(
            [self._map_to_ab_line(point, k_c[0], k_c[1]) for point in line_points])

        # Get the probability density function (PDF) of the Dirichlet distribution
        alpha=weights/sample.predicted_error
        dirichlet_likelihood = []
        alpha=list(alpha)

        for x in normalised_points:
            if np.any(np.abs(x)<floor):
                x=[floor if p<floor else p-floor for p in x]
            x=self._convert_phase_fractions_to_mass_fractions(sample.knowns, x)
            alpha_l=np.array(x)/predicted_error
            dirichlet_likelihood.append(dirichlet.pdf(mass_weights,alpha_l))


        known_simplex=line_points
        dirichlet_likelihood=np.array(dirichlet_likelihood)+np.array(sample.prior)/len(dirichlet_likelihood)
        known_pdf=dirichlet_likelihood/sum(dirichlet_likelihood)

        # Store simplex data for detailed visualisation
        sample._simplex_points = np.array(known_simplex)
        sample._simplex_pdf    = known_pdf
        sample._n_p            = n_p

        # Add phase-field corners explicitly so the interpolation hull covers
        # the diagram.  For each corner C, intersect the line through (s, C)
        # with the k1-k2 segment.  If the back-extension lands on the segment
        # (u in [0, 1]), the corner is reachable by a projection ray from a
        # virtual c on the segment — extend known_simplex/known_pdf with that
        # (c, pdf) pair so the existing loop fires a ray ending at C.  If not,
        # the corner sits on the "knowns side" of the sample and we add it as
        # a zero-density anchor point only.
        k1, k2 = k_c[0], k_c[1]
        seg = k2 - k1
        extra_zero_corners = []
        corners_c = self.phase_field.convert_to_constrained_basis(
            self.phase_field.corners)
        seg_norm_sq = float(seg @ seg)
        extra_c, extra_pdf = [], []
        for corner in corners_c:
            d = corner - s
            if np.linalg.norm(d) < 1e-10:
                continue
            # Solve s + t*d = k1 + u*seg  for t, u  (2x2 linear system).
            M = np.column_stack([d, -seg])
            det = M[0, 0] * M[1, 1] - M[0, 1] * M[1, 0]
            if abs(det) < 1e-12:
                extra_zero_corners.append((corner, 0.0))
                continue
            t_u = np.linalg.solve(M, k1 - s)
            t_val, u_val = float(t_u[0]), float(t_u[1])
            # Need u on the segment AND the projection to be a *back*-extension
            # (sample is past c on the ray, so t > 1 for s + t*d == corner with
            # c = k1 + u*seg lying on the opposite side of the sample).
            if t_val > 0 and 0.0 <= u_val <= 1.0:
                c_back = k1 + u_val * seg
                x_bary = np.array([1 - u_val, u_val])
                if np.any(np.abs(x_bary) < floor):
                    x_bary = np.array([floor if v < floor else v - floor
                                       for v in x_bary])
                x_mass = self._convert_phase_fractions_to_mass_fractions(sample.knowns, x_bary)
                alpha_c = np.array(x_mass) / predicted_error
                try:
                    pdf_c = float(dirichlet.pdf(mass_weights, alpha_c))
                except (ValueError, ZeroDivisionError):
                    pdf_c = 0.0
                if not np.isfinite(pdf_c):
                    pdf_c = 0.0
                extra_c.append(c_back)
                extra_pdf.append(pdf_c)
            else:
                extra_zero_corners.append((corner, 0.0))

        if extra_c:
            extra_c = np.array(extra_c)
            extra_pdf = np.array(extra_pdf)
            # Renormalise extra pdf into the same scale as known_pdf so each
            # ray contributes commensurate mass to griddata.
            denom = float(np.sum(dirichlet_likelihood))
            if denom > 0:
                extra_pdf_norm = extra_pdf / denom
            else:
                extra_pdf_norm = extra_pdf
            known_simplex_all = np.vstack([known_simplex, extra_c])
            known_pdf_all     = np.concatenate([known_pdf, extra_pdf_norm])
        else:
            known_simplex_all, known_pdf_all = known_simplex, known_pdf

        p_grid = self._project_known_simplex(
            known_simplex_all, known_pdf_all, s, n_p,
            extra_points=extra_zero_corners or None)

        # Store projected lines data (set by _project_known_simplex)
        sample._projected_points = np.array(self.projected_simplex)
        sample._projected_p      = np.array(self.projected_simplex_p)
        sample._n_rays           = len(known_simplex)
        sample._n_rays_emitted   = self.projected_n_rays

        return p_grid

    def _compute_p_dirichlet_3d(self, sample, n_l, n_p, floor=1e-6):
        """
        Dirichlet method for exactly 3 knowns in a 3-D constrained space
        (``m == dim == 3``) — the direct analogue of the 2-D/2-known line method.

        Discretises the k1-k2-k3 triangle into points that are uniform in solid
        angle from the sample (Möller-Trumbore, via
        :meth:`_discretise_known_simplex_3d`), evaluates the Dirichlet likelihood
        of the observed mass phase_fractions at each triangle point, then projects that
        PDF onto omega with the shared :meth:`_project_known_simplex`.

        Falls back to the Gaussian method if the triangle discretisation yields
        no valid points (e.g. a degenerate sample-triangle configuration).
        """
        k_c             = sample.knowns_constrained
        s               = sample.comp_constrained
        mass_weights    = sample.mass_weights
        predicted_error = sample.predicted_error

        tri_points, bary = self._discretise_known_simplex_3d(
            k_c[0], k_c[1], k_c[2], s, n_l)

        if len(tri_points) == 0:
            return self._compute_p_gaussian(sample, n_l, n_p)

        # Dirichlet likelihood at each triangle point (mirrors the 2-D loop):
        # floor the barycentric phase_fractions, convert mole→mass, form the Dirichlet
        # concentration alpha = mass / predicted_error, evaluate at the observed
        # mass phase_fractions.
        dirichlet_likelihood = []
        for x in bary:
            if np.any(np.abs(x) < floor):
                x = [floor if v < floor else v - floor for v in x]
            x_mass  = self._convert_phase_fractions_to_mass_fractions(sample.knowns, x)
            alpha_l = np.array(x_mass) / predicted_error
            try:
                pdf = float(dirichlet.pdf(mass_weights, alpha_l))
            except (ValueError, ZeroDivisionError):
                pdf = 0.0
            if not np.isfinite(pdf):
                pdf = 0.0
            dirichlet_likelihood.append(pdf)

        dirichlet_likelihood = (np.array(dirichlet_likelihood)
                                + np.array(sample.prior) / len(dirichlet_likelihood))
        known_pdf = dirichlet_likelihood / np.sum(dirichlet_likelihood)

        # Store simplex data for the detail-view visualiser
        sample._simplex_points = np.array(tri_points)
        sample._simplex_pdf    = known_pdf
        sample._n_p            = n_p

        # Anchor the phase-field corners at zero density so the griddata
        # interpolation hull covers the whole diagram (as the Gaussian path does).
        corners_c = self.phase_field.convert_to_constrained_basis(
            self.phase_field.corners)
        extra = [(c, 0.0) for c in corners_c]

        p_grid = self._project_known_simplex(
            tri_points, known_pdf, s, n_p, extra_points=extra)

        sample._projected_points = np.array(self.projected_simplex)
        sample._projected_p      = np.array(self.projected_simplex_p)
        sample._n_rays           = len(tri_points)
        sample._n_rays_emitted   = self.projected_n_rays
        return p_grid

    def _compute_p_gaussian(self, sample, n_l, n_p, tol=1e-4):
        """
        Single-known Gaussian method (original PICIP algorithm).

        Fires n_l rays from the sample point in a hemisphere of directions
        facing the known phase.  Each ray is weighted by a Gaussian PDF of
        the perpendicular distance from the ray to the known composition.
        The weighted densities are interpolated onto omega via griddata.

        Automatically called by _compute_p_dirichlet when only one known is
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

        # Flag directions that face *away* from k (the unknown-side hemisphere).
        # We still cast rays in the other half of the fan with zero density
        # so the interpolation hull covers the whole phase field and griddata
        # doesn't zero out corners that happen to lie on the known side.
        is_away = dirs @ direction_away > 0

        # ── Ray sampling ──────────────────────────────────────────────────────
        all_points  = []
        all_values  = []

        for d, away in zip(dirs, is_away):
            d = d / np.linalg.norm(d)
            intersect = self._find_intersection(poly, s, s + d)
            if intersect is None:
                continue
            line_len = np.linalg.norm(s - intersect)
            if line_len < 1e-10:
                continue

            # Place n_p points strictly past the sample (avoids n_l duplicates
            # at s under griddata's Delaunay triangulation — see chord method).
            ray_pts = [s + (i / n_p) * (intersect - s)
                       for i in range(1, n_p + 1)]

            if not away:
                # Wrong side of the hemisphere — zero density, ray exists only
                # to extend the interpolation hull.
                all_points.extend(ray_pts)
                all_values.extend([0.0] * n_p)
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

            all_points.extend(ray_pts)
            all_values.extend([f / line_len] * n_p)

        # Ray endpoints land on the boundary *edges*, not the corners — so the
        # convex hull of ``all_points`` inscribes a polygon strictly inside
        # the phase field and griddata's fill_value zeroes out the corner
        # regions of the diagram.  Add each phase-field corner explicitly
        # with the same Gaussian-PDF / hemisphere logic.
        corners_c = self.phase_field.convert_to_constrained_basis(
            self.phase_field.corners)
        for corner in corners_c:
            d_corner = corner - s
            len_corner = float(np.linalg.norm(d_corner))
            if len_corner < 1e-10:
                continue
            d_corner = d_corner / len_corner
            if d_corner @ direction_away <= 0:
                all_points.append(corner)
                all_values.append(0.0)
                continue
            Q_c = _orthogonal_basis(d_corner)
            if dim == 2:
                perp_c       = Q_c[:, 1]
                sigma_perp_c = float(perp_c @ sigma @ perp_c)
                y_c          = float(perp_c @ (s - k))
                f_c          = scipy.stats.norm.pdf(
                                   y_c, 0, np.sqrt(max(sigma_perp_c, 1e-30)))
            else:
                perp_c       = Q_c[:, 1:].T
                sigma_perp_c = perp_c @ sigma @ perp_c.T
                y_c          = perp_c @ (s - k)
                f_c          = multivariate_normal.pdf(
                                   y_c, np.zeros(dim - 1), sigma_perp_c)
            all_points.append(corner)
            all_values.append(float(f_c / len_corner))

        all_points = np.array(all_points)
        all_values = np.array(all_values)
        n_rays     = len(all_points) // n_p

        # Store detail-view data on sample (simplex not used for Gaussian case)
        sample._simplex_points   = None
        sample._simplex_pdf      = None
        sample._n_p              = n_p
        sample._n_rays           = n_rays
        sample._n_rays_emitted   = n_rays
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

    def _discretise_known_simplex_2d(self, start, stop, sample, n_l):
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
                raise Exception('Degenerate ray-segment intersection in _discretise_known_simplex_2d')
            t = np.clip(v1_norm * np.sin(theta) / denom, 0.0, 1.0)
            points.append(start + t * seg)

        return np.array(points)

    def _discretise_triangle_2d(self, k1, k2, k3, sample, n):
        """
        Discretise the triangle between three known phases for constrained_dim=2.

        Generates a uniform barycentric grid of (n+1)*(n+2)//2 points on the
        k1-k2-k3 triangle and returns them with their barycentric coordinates.
        Used by _compute_p_dirichlet when there are 3 known phases in a 2-D
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

    def _discretise_known_simplex_3d(self, k1, k2, k3, sample, n):
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
        # Rays parallel to the triangle give a == 0.  np.where still evaluates
        # 1.0 / a for every element, and the resulting inf then propagates into
        # u, v and t as inf/nan — all harmless, because `valid` masks those
        # entries out below (every comparison against nan is False).  Silence
        # the numpy warnings so they do not surface to the user as if something
        # had gone wrong.
        with np.errstate(divide='ignore', invalid='ignore'):
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

    def _fibonacci_sphere(self, n):
        """Approximately uniform unit vectors on S^2 (Fibonacci lattice)."""
        golden = (1 + 5 ** 0.5) / 2
        idx = np.arange(n)
        theta = 2 * np.pi * idx / golden
        phi = np.arccos(1 - 2 * (idx + 0.5) / n)
        return np.stack([
            np.cos(theta) * np.sin(phi),
            np.sin(theta) * np.sin(phi),
            np.cos(phi),
        ], axis=1)

    def _chord_back_simplex_2d(self, s, direction, K, eps=1e-9):
        """
        Intersect the line ``s + t·direction`` with the K-triangle (m=3 corners
        in 2-D) and return the back-side chord (t ≤ 0).

        Returns
        -------
        (t_in, t_out) or None
            ``t_in ≤ t_out ≤ 0``; the chord runs from ``s + t_in·direction`` to
            ``s + t_out·direction``.  ``t_out = 0`` when the sample lies inside
            the K-triangle along this line.  ``None`` if no back-side chord.
        """
        K = np.asarray(K)
        ts = []
        m = len(K)
        for i in range(m):
            a = K[i]
            b = K[(i + 1) % m]
            M = np.column_stack([direction, -(b - a)])
            det = M[0, 0] * M[1, 1] - M[0, 1] * M[1, 0]
            if abs(det) < eps:
                continue
            t, u = np.linalg.solve(M, a - s)
            if -eps <= u <= 1 + eps:
                ts.append(float(t))
        ts_neg = sorted(t for t in ts if t < -eps)
        ts_pos = sorted(t for t in ts if t > eps)
        if not ts_neg:
            return None
        if len(ts_neg) >= 2:
            return ts_neg[0], ts_neg[-1]
        if ts_pos:                        # sample inside K-triangle along line
            return ts_neg[0], 0.0
        return None

    def _chord_back_simplex_3d(self, s, direction, K, eps=1e-9):
        """
        Intersect the line ``s + t·direction`` with the K-tetrahedron (m=4
        corners in 3-D) and return the back-side chord (t ≤ 0).

        Iterates over the four triangular faces using Möller-Trumbore.
        """
        from itertools import combinations
        K = np.asarray(K)
        ts = []
        for face in combinations(range(len(K)), 3):
            v0, v1, v2 = K[list(face)]
            e1 = v1 - v0
            e2 = v2 - v0
            h = np.cross(direction, e2)
            a = float(np.dot(e1, h))
            if abs(a) < eps:
                continue
            f = 1.0 / a
            s_vec = s - v0
            u = f * float(np.dot(s_vec, h))
            if u < -eps or u > 1 + eps:
                continue
            q = np.cross(s_vec, e1)
            v = f * float(np.dot(direction, q))
            if v < -eps or u + v > 1 + eps:
                continue
            t = f * float(np.dot(e2, q))
            ts.append(t)
        ts_neg = sorted(t for t in ts if t < -eps)
        ts_pos = sorted(t for t in ts if t > eps)
        if not ts_neg:
            return None
        if len(ts_neg) >= 2:
            return ts_neg[0], ts_neg[-1]
        if ts_pos:
            return ts_neg[0], 0.0
        return None

    def _compute_p_dirichlet_chord(self, sample, n_l, n_p, floor=1e-6):
        """
        Chord-integration path for ``m == d+1`` (full-dimensional K-simplex).

        For each direction ``d`` in an angularly-uniform fan through the sample,
        the line ``s + t·d`` cuts a chord through the K-simplex on the back side
        of ``s``.  Every ``c`` on that chord is a "mixture-of-knowns" position
        whose forward ray (from ``c`` through ``s``) hits the same forward
        boundary point — so all those rays coincide and the per-line likelihood
        is the chord-integrated Dirichlet PDF::

            Λ(ℓ) = ∫_{c ∈ chord(ℓ)}  Dir(w_obs_mass | mass(α(c))/σ)  dτ

        The integrated likelihood is distributed uniformly along the forward
        ray within Ω (mass ``Λ/L_fwd`` per unit length), then interpolated onto
        the omega grid via griddata — the same final stage as the m≤d path.

        Used for (d=2, m=3) and (d=3, m=4).
        """
        s          = np.asarray(sample.comp_constrained, dtype=float)
        K          = np.asarray(sample.knowns_constrained, dtype=float)  # (m, dim)
        m, dim     = K.shape
        sigma      = sample.predicted_error
        mass_obs   = sample.mass_weights

        # Closed-form barycentric inverse: solve [K^T; 1] α = [c; 1].
        K_aug = np.vstack([K.T, np.ones(m)])      # (dim+1, m), square for m=dim+1
        K_aug_inv = np.linalg.inv(K_aug)

        # Forward boundary intersection setup
        lines = np.array(self.phase_field._get_constraint_lines())
        poly  = {'A': lines[:, :-1], 'b': -lines[:, -1]}

        # Direction fan
        if dim == 2:
            theta = np.linspace(0, 2 * np.pi, n_l, endpoint=False)
            directions = np.stack([np.cos(theta), np.sin(theta)], axis=1)
        elif dim == 3:
            directions = self._fibonacci_sphere(n_l)
        else:
            raise Exception(f"Chord method not supported for dim={dim}")

        n_chord       = max(n_l, 20)
        all_points    = []
        all_values    = []
        chord_pts_viz = []
        chord_pdf_viz = []

        for d_fwd in directions:
            d_fwd = d_fwd / np.linalg.norm(d_fwd)
            if dim == 2:
                chord = self._chord_back_simplex_2d(s, d_fwd, K)
            else:
                chord = self._chord_back_simplex_3d(s, d_fwd, K)

            # Forward ray endpoint and length are needed regardless of whether
            # the line crosses the K-simplex — without this fallback, any
            # direction with no back-chord leaves a wedge of the phase field
            # uncovered by ``all_points`` and griddata's fill_value zeroes the
            # corners of the diagram in those directions.
            end_point = self._find_intersection(poly, s, s + d_fwd)
            if end_point is None:
                continue
            L_fwd = float(np.linalg.norm(end_point - s))
            if L_fwd < 1e-10:
                continue
            # Skip i=0 to avoid n_l duplicate points at the sample under griddata.
            ray_pts = [s + (i / n_p) * (end_point - s) for i in range(1, n_p + 1)]

            if chord is None or (chord[1] - chord[0]) < 1e-10:
                # Line through sample misses the K-simplex along this direction:
                # no Dirichlet support → contribute the ray with zero density
                # so the interpolation hull still covers the diagram.
                all_points.extend(ray_pts)
                all_values.extend([0.0] * n_p)
                continue

            t_in, t_out = chord
            ts        = np.linspace(t_in, t_out, n_chord)
            dtau      = (t_out - t_in) / max(n_chord - 1, 1)
            cs        = s + ts[:, None] * d_fwd                   # (n_chord, dim)

            # Bary coords of every chord point in one shot
            cs_aug    = np.column_stack([cs, np.ones(len(cs))]).T  # (dim+1, n_chord)
            alphas    = (K_aug_inv @ cs_aug).T                     # (n_chord, m)

            Lambda = 0.0
            for c, alpha in zip(cs, alphas):
                a = np.where(np.abs(alpha) < floor, floor, alpha)
                a = a / a.sum()
                mass_alpha = self._convert_phase_fractions_to_mass_fractions(sample.knowns, a)
                try:
                    lj = float(dirichlet.pdf(mass_obs, mass_alpha / sigma))
                except (ValueError, ZeroDivisionError):
                    lj = 0.0
                if not np.isfinite(lj):
                    lj = 0.0
                Lambda += lj * dtau
                chord_pts_viz.append(c)
                chord_pdf_viz.append(lj)

            all_points.extend(ray_pts)
            all_values.extend([Lambda / L_fwd] * n_p)

        # Ensure each phase-field corner is included so the interpolation hull
        # spans the diagram — without this, omega corner points fall outside
        # the convex hull of ``all_points`` and griddata zeroes them out.
        corners_c = self.phase_field.convert_to_constrained_basis(
            self.phase_field.corners)
        for corner in corners_c:
            d_corner = corner - s
            len_corner = float(np.linalg.norm(d_corner))
            if len_corner < 1e-10:
                continue
            d_corner = d_corner / len_corner
            if dim == 2:
                chord_c = self._chord_back_simplex_2d(s, d_corner, K)
            else:
                chord_c = self._chord_back_simplex_3d(s, d_corner, K)
            if chord_c is None or (chord_c[1] - chord_c[0]) < 1e-10:
                all_points.append(corner)
                all_values.append(0.0)
                continue
            t_in_c, t_out_c = chord_c
            ts_c   = np.linspace(t_in_c, t_out_c, n_chord)
            dtau_c = (t_out_c - t_in_c) / max(n_chord - 1, 1)
            cs_c   = s + ts_c[:, None] * d_corner
            cs_aug_c = np.column_stack([cs_c, np.ones(len(cs_c))]).T
            alphas_c = (K_aug_inv @ cs_aug_c).T
            Lambda_c = 0.0
            for cc, alpha in zip(cs_c, alphas_c):
                a = np.where(np.abs(alpha) < floor, floor, alpha)
                a = a / a.sum()
                mass_alpha = self._convert_phase_fractions_to_mass_fractions(sample.knowns, a)
                try:
                    lj = float(dirichlet.pdf(mass_obs, mass_alpha / sigma))
                except (ValueError, ZeroDivisionError):
                    lj = 0.0
                if not np.isfinite(lj):
                    lj = 0.0
                Lambda_c += lj * dtau_c
            all_points.append(corner)
            all_values.append(float(Lambda_c / len_corner))

        if not all_points:
            raise ZeroPException()

        all_points = np.array(all_points)
        all_values = np.array(all_values)

        sample._simplex_points   = np.array(chord_pts_viz) if chord_pts_viz else None
        sample._simplex_pdf      = np.array(chord_pdf_viz) if chord_pdf_viz else None
        sample._n_p              = n_p
        sample._n_rays           = len(all_points) // n_p if n_p else 0
        sample._n_rays_emitted   = sample._n_rays
        sample._projected_points = all_points
        sample._projected_p      = all_values
        self.projected_simplex   = list(all_points)
        self.projected_simplex_p = list(all_values)

        try:
            p = griddata(all_points, all_values, self.phase_field.omega,
                         method='linear', fill_value=0)
        except Exception:
            p = griddata(all_points, all_values, self.phase_field.omega,
                         method='nearest', fill_value=0)
        if np.isnan(p).any():
            p = np.nan_to_num(p)
        if np.any(p < 0):
            p[p < 0] = 0
        if p.sum() == 0:
            raise ZeroPException()
        return p / p.sum()

    def _map_to_ab_line(self,point, A, B, ret_t=False):
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

    def _project_known_simplex(self, known_simplex, known_pdf, sample, n_p,
                               tol=1e-4, extra_points=None):
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
        points=[]
        values=[]
        n_rays_used=0
        for c, pdf in zip(known_simplex, known_pdf):
            direction = sample - c
            if np.linalg.norm(direction) < 1e-10:
                continue   # simplex point coincides with sample — skip
            end_point = self._find_intersection(polygon_constraints, c, sample)
            if end_point is None:
                # _find_intersection discards crossings closer than its epsilon
                # to c, so it returns None when c already sits on the phase-field
                # boundary — which happens whenever a known phase is a corner,
                # edge or face phase and the simplex touches the hull.  Such a
                # ray has no interior length to carry density along, so skip it.
                continue
            # Place n_p points strictly *past* the sample along the forward ray.
            # Skipping the i=0 case avoids n_l duplicate input points at the
            # sample location — duplicates make scipy's Delaunay triangulation
            # pick one ray's value arbitrarily for the shared vertex, which
            # propagates as radial oscillation in the interpolated density.
            l_ps = [sample + (i / n_p) * (end_point - sample)
                    for i in range(1, n_p + 1)]
            points += l_ps
            values += [pdf / np.linalg.norm(sample - end_point)] * n_p
            n_rays_used += 1

        if n_rays_used == 0:
            raise ZeroPException()

        # Caller-supplied (point, value) pairs — used to add the phase-field
        # corners so the interpolation hull covers the diagram.
        if extra_points is not None:
            for pt, v in extra_points:
                points.append(np.asarray(pt))
                values.append(float(v))

        #plotting
        self.projected_simplex=points
        self.projected_simplex_p=values
        # Rays actually emitted, which is <= len(known_simplex) once degenerate
        # rays are skipped.  The first n_rays_used * n_p entries of
        # projected_simplex are the ray block, laid out ray-major.
        self.projected_n_rays=n_rays_used

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
            p=np.nan_to_num(p)
        return(p/np.sum(p))

    def _convert_phase_fractions_to_mass_fractions(self, phases, weights):
        """Convert phase-fraction weights as used in the `Sample` class
        to mass-fraction weights for the given phases."""
        molar_masses=np.array([phase.weight for phase in phases])
        mass_weights=molar_masses*weights
        weights=np.array(mass_weights)/sum(mass_weights)
        return weights

    def _find_intersection(self, polygon_constraints, x, sample, epsilon=1e-2):
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

        # Iterate through each edge of the polygon,
        # given by the inequalities Ax <= b
        for i, (A_row, b_val) in enumerate(
            zip(polygon_constraints["A"], polygon_constraints["b"])
        ):
            A_dot_dir = np.dot(A_row, direction)

            if abs(A_dot_dir) > 1e-6:
                t_intersect = (b_val - np.dot(A_row, x)) / A_dot_dir

                if t_intersect > 1e-6:
                    intersection_point = self._line_through_point(
                        x, sample, t_intersect
                    )
                    if np.linalg.norm(intersection_point - x) > epsilon:
                        intersections.append(intersection_point)

        if len(intersections) > 0:
            distances = [
                np.linalg.norm(intersection - x)
                for intersection in intersections
            ]
            closest_intersection = intersections[np.argmin(distances)]
            return closest_intersection
        else:
            return None

    def _line_through_point(self, x, sample, t):
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

    def _top_probability_mass(self, omega, p, x):
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
        """
        omega = np.asarray(omega)
        p = np.asarray(p, dtype=float)
        p = p / p.sum()

        sort_idx = np.argsort(p)[::-1]
        omega_sorted = omega[sort_idx]
        p_sorted = p[sort_idx]

        cumsum = np.cumsum(p_sorted)

        cutoff = x / 100.0
        keep_mask = cumsum <= cutoff
        if not np.all(keep_mask):
            first_above = np.argmax(~keep_mask)
            keep_mask[first_above] = True

        return omega_sorted[keep_mask]


