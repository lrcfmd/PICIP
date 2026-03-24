"""
Composition spreading via Lloyd's algorithm (Voronoi relaxation).

Given a phase field, finds a set of *n* compositions that cover the
accessible composition space as uniformly as possible.  The objective
minimised is the **worst-case coverage distance**: the maximum distance
from any grid point in ``omega`` to its nearest suggested composition.

The algorithm runs ``num_repeats`` independent trials in parallel, each
starting from a random selection of grid points and running Lloyd
(Voronoi centroid) iterations until convergence.  After each trial a
big-jump escape step can teleport the most redundant site to the most
poorly covered region, helping avoid local optima.  The trial with the
smallest worst-case coverage distance is returned.

Phase-field corners and any compositions registered via
``add_known_phases()`` are treated as fixed sites: they attract nearby
omega points but are never moved, so suggested compositions stay away
from already-known regions.

Typical usage
-------------
pf = Phase_Field()
pf.setup_uncharged(["Fe", "Mn", "Ti"])

spread = Spread(pf)
spread.add_known_phases(["FeMn", "FeTi"])   # optional: treat as fixed points

result = spread.run(n=10)
result.save("suggestions")
"""

import math
import multiprocessing as mp
import os
import time

import numpy as np
import pandas as pd
from scipy.optimize import nnls


DEFAULT_NUM_REPEATS  = 100
DEFAULT_MAX_ITER     = 200
DEFAULT_TOL          = 1e-3
DEFAULT_BIG_JUMP     = 2


# ─────────────────────────────────────────────────────────────────────────────
class SpreadResult:
    """Result of a Spread run.

    Each point is a Voronoi centroid on the ``omega`` grid — the centre of
    mass of the grid points assigned to that site after Lloyd relaxation
    converges.

    Attributes
    ----------
    points : ndarray, shape (n, constrained_dim)
        Suggested compositions in the constrained basis.
    points_standard : ndarray, shape (n, nelements)
        Suggested compositions in the standard (element-fraction) basis;
        rows sum to 1.
    elements : list of str
        Element labels, in the same order as the columns of ``points_standard``.
    """

    def __init__(self, points, phase_field):
        self.points           = points
        self.points_standard  = phase_field.convert_to_standard_basis(points)
        self.elements         = phase_field.elements

    def save(self, path, prec=3):
        """Write suggested compositions to a CSV file.

        One row per suggested composition, columns named by element.
        The index starts at 1.

        Parameters
        ----------
        path : str
            File path.  ``.csv`` is appended if not already present.
        prec : int, optional
            Decimal places for composition values.  Default 3.

        Returns
        -------
        str
            Path of the file written.
        """
        rows = []
        for p in self.points_standard:
            row = {el: round(float(v), prec) for el, v in zip(self.elements, p)}
            rows.append(row)
        df = pd.DataFrame(rows)
        df.index += 1
        csv_path = path if path.endswith(".csv") else path + ".csv"
        df.to_csv(csv_path)
        return csv_path


# ─────────────────────────────────────────────────────────────────────────────
class SpreadResultWithPrecursors(SpreadResult):
    """SpreadResult that also holds precursor formula-unit amounts.

    Returned by ``Spread.simplify_to_precursors()``.  ``points`` and
    ``points_standard`` reflect the rounded compositions — i.e. what the
    precursor amounts actually produce — rather than the idealised spread
    positions.

    Attributes
    ----------
    precursor_amounts : ndarray, shape (n, n_precursors)
        Formula-unit amounts for each precursor at each suggested point.
        Unrounded values sum to the LCM of the precursor formula sizes;
        rounding introduces a small deviation from that sum.
    precursor_labels : list of str
        Composition string for each precursor, matching the columns of
        ``precursor_amounts``.
    """

    def __init__(self, points, phase_field, precursor_amounts, precursor_labels):
        super().__init__(points, phase_field)
        self.precursor_amounts = precursor_amounts
        self.precursor_labels  = precursor_labels

    def save(self, path, prec=3):
        """Write suggested compositions and precursor amounts to CSV.

        Columns are the element fractions followed by one column per
        precursor labelled by its composition string.  The index starts at 1.

        Parameters
        ----------
        path : str
            File path.  ``.csv`` is appended if not already present.
        prec : int, optional
            Decimal places for all numeric values.  Default 3.

        Returns
        -------
        str
            Path of the file written.
        """
        rows = []
        for p, amounts in zip(self.points_standard, self.precursor_amounts):
            row = {el: round(float(v), prec) for el, v in zip(self.elements, p)}
            for label, a in zip(self.precursor_labels, amounts):
                row[label] = round(float(a), prec)
            rows.append(row)
        df = pd.DataFrame(rows)
        df.index += 1
        csv_path = path if path.endswith(".csv") else path + ".csv"
        df.to_csv(csv_path)
        return csv_path


# ─────────────────────────────────────────────────────────────────────────────
class Spread:
    """Spread compositions across a phase field using Lloyd's algorithm.

    Parameters
    ----------
    phase_field : Phase_Field
        The phase field to spread compositions over.

    Attributes
    ----------
    phase_field : Phase_Field
    known_phases : ndarray or None
        Known phase compositions in the constrained basis, treated as
        fixed points during spreading.  Set via ``add_known_phases()``.

    Notes
    -----
    **Precursor rounding** — if your phase field was set up with
    ``precursors=[...]``, call ``simplify_to_precursors(result)`` after
    ``run()`` to snap each suggested composition to the closest one
    achievable as a ratio of the given precursor powders.  The result is
    a :class:`SpreadResultWithPrecursors` whose ``precursor_amounts``
    give the formula-unit amounts to weigh out.
    """

    def __init__(self, phase_field):
        self.phase_field   = phase_field
        self.known_phases  = None

    # ── User-facing ──────────────────────────────────────────────────────────

    def add_known_phases(self, compositions):
        """Register known co-existing phases as fixed points.

        These phases are treated as immovable sites during the Voronoi
        relaxation so that suggested compositions stay away from them.

        Parameters
        ----------
        compositions : list of str
            Composition strings, e.g. ``["FeMn", "FeTi"]``.
        """
        self.phase_field.add_knowns(compositions)
        self.known_phases = self.phase_field.knowns_constrained

    def run(
        self,
        n,
        num_repeats=DEFAULT_NUM_REPEATS,
        max_iter=DEFAULT_MAX_ITER,
        tol=DEFAULT_TOL,
        big_jump_steps=DEFAULT_BIG_JUMP,
    ):
        """Find *n* maximally spread-out compositions.

        Runs ``num_repeats`` independent Lloyd-relaxation trials in parallel
        and returns the trial with the smallest worst-case coverage distance
        (the maximum distance from any omega grid point to its nearest site).
        Phase-field corners and any ``add_known_phases()`` compositions are
        treated as immovable fixed sites throughout.

        Parameters
        ----------
        n : int
            Number of compositions to suggest.
        num_repeats : int, optional
            Number of independent random restarts.  More restarts explore
            more of the space and tend to give better solutions at the cost
            of runtime.  Default 100.
        max_iter : int, optional
            Maximum Lloyd iterations per trial.  Most trials converge well
            before this limit.  Default 200.
        tol : float, optional
            Convergence tolerance: a trial is considered converged when the
            largest site displacement in a single iteration falls below this
            value (in constrained-basis units).  Default 1e-3.
        big_jump_steps : int, optional
            After each converged trial, the most redundant site (the one in
            the closest pair with the smaller Voronoi cell) is teleported to
            the most poorly covered omega point, and Lloyd relaxation restarts.
            This parameter controls how many such escape attempts are allowed
            per trial.  Default 2.

        Returns
        -------
        SpreadResult

        """
        fixed = self._build_fixed_sites()

        tasks = [
            (i, n, fixed, max_iter, tol, big_jump_steps)
            for i in range(num_repeats)
        ]

        with mp.Pool(initializer=self._worker_init) as pool:
            results = pool.map(self._run_voronoi_parallel, tasks)

        best_points, best_cover = min(results, key=lambda x: x[1])
        print(f"Best solution: worst coverage distance = {round(best_cover, 4)}")

        return SpreadResult(best_points, self.phase_field)

    def simplify_to_precursors(self, result, accuracy=2):
        """Snap each suggested composition to the closest achievable precursor ratio.

        Uses non-negative least squares (NNLS) to find the precursor weights
        that best reproduce each suggested composition, converts the weights to
        formula-unit amounts scaled to the LCM of the precursor formula sizes,
        rounds to ``accuracy`` decimal places, and recomputes the actual
        composition the rounded amounts produce.

        Requires the phase field to have been set up with
        ``precursors=[...]``.

        Parameters
        ----------
        result : SpreadResult
            Output from ``run()``.
        accuracy : int, optional
            Decimal places to round precursor formula-unit amounts to.
            Use ``accuracy=0`` for integer amounts (e.g. when precursors are
            oxide compounds such as Fe2O3, MnO).  Default 2.

        Returns
        -------
        SpreadResultWithPrecursors
        """
        pf    = self.phase_field
        A     = pf.convert_to_constrained_basis(pf.precursors_standard)
        A     = np.hstack((np.ones((A.shape[0], 1)), A)).T

        pre_s = np.array([int(round(x)) for x in pf.precursors_size], dtype=float)
        lcm   = math.lcm(*pre_s.astype(int).tolist())

        adjusted_points  = []
        precursor_amounts = []

        for p in result.points:
            p_ext = np.insert(p, 0, 1)
            x, _  = nnls(A, p_ext, maxiter=60000)

            # Convert NNLS atom-fraction weights to formula-unit amounts.
            # x[i] is proportional to the number of atoms from precursor i,
            # so formula units ∝ x[i] / precursors_size[i].
            q  = x / pre_s
            fu = lcm * q / q.sum()          # scale so amounts sum to lcm
            fu = np.round(fu, accuracy)

            # Reconstruct the composition that the rounded amounts produce.
            # Each formula unit of precursor i contributes pre_s[i] atoms
            # with composition precursors_standard[i].
            v = np.zeros(pf.nelements)
            for ps, fi, si in zip(pf.precursors_standard, fu, pre_s):
                v += fi * si * ps
            if v.sum() > 0:
                v /= v.sum()

            adjusted_points.append(pf.convert_to_constrained_basis([v])[0])
            precursor_amounts.append(fu)

        adjusted_points  = np.array(adjusted_points)
        precursor_amounts = np.array(precursor_amounts)
        labels = pf.get_composition_strings(pf.precursors_standard, out="txt")

        return SpreadResultWithPrecursors(
            adjusted_points, pf, precursor_amounts, labels
        )

    def evaluate_spread(self, result):
        """Plot per-site coverage and separation diagnostics for a result.

        Produces a matplotlib figure with two overlaid histograms:

        * **Worst coverage distance (dwcd, blue)** — for each suggested site,
          the distance to the furthest omega point in its Voronoi cell.  A
          smaller value means that site covers its region well.
        * **Minimum pairwise distance (dmpd, red)** — for each suggested site,
          the distance to its nearest neighbour (movable or fixed).  A larger
          value means sites are well separated.

        A good solution has small dwcd and large dmpd.  Both distributions are
        annotated with a composition label derived by walking that distance
        along a corner-to-corner axis, giving an intuitive sense of scale.

        Parameters
        ----------
        result : SpreadResult
            Output from ``run()``.
        """
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        pf      = self.phase_field
        sites_m = result.points
        fixed   = self._build_fixed_sites()
        omega   = pf.omega

        all_sites   = np.vstack([sites_m, fixed])
        distances   = np.linalg.norm(omega[:, None, :] - all_sites[None, :, :], axis=2)
        assignments = np.argmin(distances, axis=1)

        dmpd, dwcd = [], []
        for i in range(len(sites_m)):
            dists    = np.linalg.norm(sites_m[i] - all_sites, axis=1)
            dists[i] = np.inf
            dmpd.append(float(np.min(dists)))

            mask = assignments == i
            dwcd.append(
                float(np.max(np.linalg.norm(omega[mask] - sites_m[i], axis=1)))
                if np.any(mask) else 0.0
            )

        dmpd = np.array(dmpd)
        dwcd = np.array(dwcd)

        # Use constrained basis for corners so distances are in the same space as
        # dwcd/dmpd (which are computed from omega and sites_m, both constrained).
        corners_c   = pf.convert_to_constrained_basis(pf.corners)
        corner_strs = pf.get_composition_strings(pf.corners, out="txt_short", simple_ratio_lim=1)
        corner_dict = dict(zip(corner_strs, corners_c))
        c0_name     = corner_strs[0]
        c0          = corner_dict[c0_name]

        corner_dists = {
            name: float(np.linalg.norm(c0 - coord))
            for name, coord in corner_dict.items()
            if name != c0_name
        }
        max_val  = max(dwcd.max(), dmpd.max())
        valid    = {n: d for n, d in corner_dists.items() if d > max_val}
        if valid:
            ref_name, ref_dist = min(valid.items(), key=lambda x: x[1])
        else:
            ref_dist = max_val * 1.1
            ref_name = "N/A"

        bins        = np.linspace(0, ref_dist, 50)
        c1, _       = np.histogram(dwcd, bins=bins)
        c2, _       = np.histogram(dmpd, bins=bins)
        max_count   = max(c1.max(), c2.max())
        bw          = np.diff(bins)[0]
        bcenters    = bins[:-1]

        mean_dwcd = dwcd.mean()
        mean_dmpd = dmpd.mean()

        # Walk along the corner-to-corner line in constrained space — the distance
        # is meaningful there. get_composition_strings auto-detects constrained basis.
        def _comp_along_line(start, end, dist):
            v = end - start
            v = v / np.linalg.norm(v)
            return start + dist * v

        ref_corner = corner_dict[ref_name] if ref_name in corner_dict else None

        plt.figure(figsize=(10, 6))
        plt.bar(bcenters, c1 / max_count, width=bw, alpha=0.5,
                color='blue', label='Worst coverage distance', align='edge')
        plt.bar(bcenters, c2 / max_count, width=bw, alpha=0.5,
                color='red',  label='Minimum pairwise distance', align='edge')
        plt.xlabel('Distance')
        plt.ylabel('Normalised count')
        plt.title('Histograms for dwcd and dmpd with relative distances')
        plt.legend()
        plt.axhline(y=0, color='black', linewidth=1)
        plt.axvline(x=0,         color='black', linestyle='--')
        plt.axvline(x=ref_dist,  color='black', linestyle='--')

        text_offset = 0.05

        # Main arrow: corner-to-corner distance
        arrow_y_main = -0.1
        plt.annotate('', xy=(ref_dist, arrow_y_main), xytext=(0, arrow_y_main),
                     arrowprops=dict(arrowstyle='<->', color='black'))
        plt.text(ref_dist / 2, arrow_y_main + text_offset,
                 f"d({c0_name}, {ref_name})",
                 color='black', ha='center', va='center', fontsize=10)

        # Arrow for mean dwcd
        arrow_y_a = -0.2
        plt.annotate('', xy=(mean_dwcd, arrow_y_a), xytext=(0, arrow_y_a),
                     arrowprops=dict(arrowstyle='<->', color='blue'))
        if ref_corner is not None:
            pt_a    = _comp_along_line(c0, ref_corner, mean_dwcd)
            label_a = pf.get_composition_strings(pt_a.reshape(1, -1), out='txt_short')
        else:
            label_a = f"{mean_dwcd:.3f}"
        text_x_a = mean_dwcd / 2
        ha_a = 'center'
        if text_x_a < 0.1 * ref_dist:
            text_x_a = mean_dwcd + 0.01 * ref_dist
            ha_a = 'left'
        plt.text(text_x_a, arrow_y_a + text_offset,
                 f"d({c0_name}, {label_a})",
                 color='blue', ha=ha_a, va='center', fontsize=10)

        # Arrow for mean dmpd
        arrow_y_b = -0.3
        plt.annotate('', xy=(mean_dmpd, arrow_y_b), xytext=(0, arrow_y_b),
                     arrowprops=dict(arrowstyle='<->', color='red'))
        if ref_corner is not None:
            pt_b    = _comp_along_line(c0, ref_corner, mean_dmpd)
            label_b = pf.get_composition_strings(pt_b.reshape(1, -1), out='txt_short')
        else:
            label_b = f"{mean_dmpd:.3f}"
        text_x_b = mean_dmpd / 2
        ha_b = 'center'
        if text_x_b < 0.1 * ref_dist:
            text_x_b = mean_dmpd + 0.01 * ref_dist
            ha_b = 'left'
        plt.text(text_x_b, arrow_y_b + text_offset,
                 f"d({c0_name}, {label_b})",
                 color='red', ha=ha_b, va='center', fontsize=10)

        plt.ylim(bottom=-0.4)
        plt.xlim(left=0, right=ref_dist)
        plt.tight_layout()
        import subprocess, tempfile
        tmp = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
        plt.savefig(tmp.name, dpi=150, bbox_inches='tight')
        plt.close()
        subprocess.Popen(['xdg-open', tmp.name])

    # ── Internal ─────────────────────────────────────────────────────────────

    def _build_fixed_sites(self):
        """Return corners + any known phases as a single fixed-sites array."""
        pf      = self.phase_field
        corners = (
            pf.convert_to_constrained_basis(pf.corners)
            if pf.corners is not None
            else np.empty((0, pf.constrained_dim))
        )

        if self.known_phases is not None and len(self.known_phases):
            return np.vstack([corners, self.known_phases])
        return corners

    @staticmethod
    def _worker_init():
        """Seed each worker process independently."""
        np.random.seed(int(time.time()) + os.getpid())

    def _run_voronoi_parallel(self, task):
        """Unpack a task tuple and run one trial; print progress every 20 trials."""
        i, n_sites, fixed_sites, max_iter, tol, big_jump_steps = task
        if i % 20 == 1:
            print(f"  Completed {i} runs")
        return self._run_voronoi(n_sites, fixed_sites, max_iter, tol, big_jump_steps)

    def _run_voronoi(self, n_sites, fixed_sites, max_iter, tol, big_jump_steps):
        """One independent Lloyd-relaxation trial.

        Randomly initialises ``n_sites`` movable sites from ``omega``, then
        alternates between Lloyd convergence and big-jump escape steps, keeping
        the positions and coverage score of the best converged state seen.

        Returns
        -------
        tuple : (best_positions ndarray, best_cover float)
            ``best_positions`` has shape ``(n_sites, constrained_dim)``;
            ``best_cover`` is the worst-case coverage distance achieved.
        """
        omega   = self.phase_field.omega
        N       = len(omega)
        m_fixed = len(fixed_sites)

        if m_fixed == 0:
            fixed_sites = np.empty((0, self.phase_field.constrained_dim))

        chosen       = np.random.choice(N, size=n_sites, replace=False)
        sites_m      = omega[chosen].copy()
        best_pos     = sites_m.copy()
        best_cover   = 9999.0

        for bj in range(big_jump_steps + 1):
            sites_m, assignments, converged = self._lloyd_iteration_run(
                sites_m, fixed_sites, max_iter, tol
            )

            if converged:
                all_sites = np.vstack([sites_m, fixed_sites]) if m_fixed else sites_m
                cover     = self._get_max_cover(all_sites)
                if cover < best_cover:
                    best_cover = cover
                    best_pos   = sites_m.copy()

                if bj < big_jump_steps:
                    sites_m, jumped = self._voronoi_escape_local_opt(
                        all_sites, sites_m, assignments
                    )
                    if not jumped:
                        break
                else:
                    break
            else:
                break

        return (best_pos, best_cover)

    def _lloyd_iteration_run(self, sites_m, fixed_sites, max_iter, tol):
        """Run Lloyd iterations until convergence or ``max_iter`` is reached.

        Each iteration assigns every omega point to its nearest site (Voronoi
        partition), then moves each *movable* site to the centroid of its cell.
        Fixed sites participate in the assignment step but are never moved.
        Converges when the largest site displacement in an iteration is below
        ``tol``.

        Returns
        -------
        tuple : (sites_m, assignments, converged)
            ``assignments[j]`` is the index into ``all_sites`` of the nearest
            site to ``omega[j]``.  ``converged`` is ``True`` if the tolerance
            criterion was met before ``max_iter``.
        """
        omega    = self.phase_field.omega
        n_sites  = len(sites_m)
        m_fixed  = len(fixed_sites)

        if m_fixed:
            all_sites = np.vstack([sites_m, fixed_sites])
        else:
            all_sites = sites_m.copy()

        for _ in range(max_iter):
            old_m     = sites_m.copy()
            distances = np.linalg.norm(omega[:, None, :] - all_sites[None, :, :], axis=2)
            assignments = np.argmin(distances, axis=1)

            for i in range(n_sites):
                mask = assignments == i
                if np.any(mask):
                    sites_m[i] = omega[mask].mean(axis=0)

            all_sites[:n_sites] = sites_m
            max_shift = np.max(np.linalg.norm(sites_m - old_m, axis=1))
            if max_shift < tol:
                return (sites_m, assignments, True)

        return (sites_m, assignments, False)

    def _get_max_cover(self, all_sites):
        """Maximum distance from any omega point to its nearest site."""
        omega  = self.phase_field.omega
        d      = np.linalg.norm(omega[:, None, :] - all_sites[None, :, :], axis=2)
        return float(np.max(np.min(d, axis=1)))

    def _voronoi_escape_local_opt(self, all_sites, sites_m, assignments):
        """Escape a local optimum by teleporting the most redundant movable site.

        Finds the closest pair of sites (movable or fixed).  Of the two, picks
        the movable one with the smaller Voronoi cell (fewest omega points
        assigned to it) as the redundant site.  Moves it to the omega point
        that is furthest from any site — the most poorly covered location —
        then returns for another round of Lloyd relaxation.

        Returns
        -------
        tuple : (sites_m, did_jump)
            ``did_jump`` is ``False`` if no valid movable site could be
            identified (e.g. the closest pair are both fixed).
        """
        omega   = self.phase_field.omega
        n_sites = len(sites_m)

        pairwise = np.linalg.norm(
            all_sites[:, None, :] - all_sites[None, :, :], axis=2
        )
        np.fill_diagonal(pairwise, np.inf)
        min_dist   = np.min(pairwise)
        if not np.isfinite(min_dist):
            return (sites_m, False)
        candidates = np.argwhere(pairwise == min_dist)
        if len(candidates) == 0:
            return (sites_m, False)
        r, c       = candidates[0]
        if r > c:
            r, c = c, r

        if r < n_sites and c < n_sites:
            sizes = [np.sum(assignments == r), np.sum(assignments == c)]
            site_to_jump = r if sizes[0] <= sizes[1] else c
        elif r < n_sites:
            site_to_jump = r
        elif c < n_sites:
            site_to_jump = c
        else:
            return (sites_m, False)

        dists_to_sites = np.linalg.norm(omega[:, None, :] - all_sites[None, :, :], axis=2)
        min_dists      = np.min(dists_to_sites, axis=1)
        furthest       = omega[np.argmax(min_dists)]
        sites_m[site_to_jump] = furthest

        return (sites_m, True)
