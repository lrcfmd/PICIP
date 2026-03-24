"""
Tests for Spread, SpreadResult, and SpreadResultWithPrecursors.

Covers all Phase_Field geometries:
  - Uncharged 2D / 3D  (simplex triangle / tetrahedron)
  - Charged 2D / 3D    (non-simplex polytope — charge-neutrality hyperplane
                         cuts the simplex, producing a different shape)
  - Charge-ranges 2D   (further relaxed polytope — Mn can be 3+ or 4+)
  - Custom constraint  (spinel stoichiometry — polytope with fixed O fraction)
  - With minimum-amount constraint (truncated field)
  - With precursors (reduced convex-hull field)

The key distinction: uncharged fields are simplices; every other setup type
produces a polytope whose geometry differs from a simplex, so the corners,
edges, and omega distribution are structurally different.  Tests that pass
for a simplex must also pass for these non-simplex cases.
"""

import os
import sys
import tempfile

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from phase_field import Phase_Field
from spread import Spread, SpreadResult, SpreadResultWithPrecursors


# ─── helpers ─────────────────────────────────────────────────────────────────

FAST = dict(num_repeats=3, max_iter=50)   # keep unit tests quick

def _uncharged_2d():
    pf = Phase_Field()
    pf.setup_uncharged(["Fe", "Mn", "Ti"], n_points=200)
    return pf

def _uncharged_3d():
    pf = Phase_Field()
    pf.setup_uncharged(["Fe", "Mn", "Ti", "Cu"], n_points=300)
    return pf

def _charged_2d():
    """Li-Ni-Mn-Co-O  (5 elements + charge → 3D constrained → 2D after basis)
    This is NOT a simplex: the charge-neutral polytope is a quadrilateral,
    not a triangle.  Its corners are not pure elements.
    """
    pf = Phase_Field()
    pf.setup_charged({"Li": 1, "Ni": 3, "Mn": 4, "O": -2}, n_points=200)
    return pf

def _charged_3d():
    """Li-Ni-Mn-Co-O  (5 elements + charge → 3D polytope)"""
    pf = Phase_Field()
    pf.setup_charged({"Li": 1, "Ni": 3, "Mn": 4, "Co": 3, "O": -2}, n_points=300)
    return pf

def _charge_ranges_3d():
    """Li-Mn(3+/4+)-Co-O  (4 elements + charge ranges → 3D)
    The valid region is wider than a fixed-charge field because Mn can vary
    between 3+ and 4+.  Not a simplex.
    """
    pf = Phase_Field()
    pf.setup_charge_ranges({"Li": 1, "Mn": [3, 4], "Co": 3, "O": -2}, n_points=300)
    return pf

def _custom_spinel():
    """Li-Fe-Mn-O spinel (4 elements + sum=1 + 3:4 cation:anion constraint → 2D)
    O is fixed at 4/7; corners are end-member spinels, not pure elements.
    """
    pf = Phase_Field()
    pf.setup_custom(
        ["Li", "Fe", "Mn", "O"],
        custom_con=np.array([[4, 4, 4, -3]]),
        n_points=200,
    )
    return pf

def _uncharged_2d_min_amount():
    pf = Phase_Field()
    pf.setup_uncharged(["Fe", "Mn", "Ti"], n_points=300)
    pf.constrain_minimum_amount(0.05)
    return pf

def _uncharged_2d_precursors():
    pf = Phase_Field()
    pf.setup_uncharged(
        ["Fe", "Mn", "Ti"],
        precursors=["Fe3Mn", "FeTi2", "MnTi"],
        n_points=200,
    )
    return pf


ALL_PFS = [
    ("uncharged_2d",         _uncharged_2d),
    ("uncharged_3d",         _uncharged_3d),
    ("charged_2d",           _charged_2d),
    ("charged_3d",           _charged_3d),
    ("charge_ranges_3d",     _charge_ranges_3d),
    ("custom_spinel",        _custom_spinel),
    ("uncharged_2d_min",     _uncharged_2d_min_amount),
    ("uncharged_2d_precurs", _uncharged_2d_precursors),
]


# ─── SpreadResult ─────────────────────────────────────────────────────────────

class TestSpreadResult:

    def test_attributes(self):
        pf = _uncharged_2d()
        pts = pf.omega[:5]
        r = SpreadResult(pts, pf)
        assert r.points is pts
        assert r.points_standard.shape == (5, pf.nelements)
        assert r.elements == pf.elements

    def test_save_csv(self, tmp_path):
        pf = _uncharged_2d()
        r = SpreadResult(pf.omega[:3], pf)
        path = str(tmp_path / "out")
        saved = r.save(path)
        assert saved.endswith(".csv")
        assert os.path.exists(saved)
        import pandas as pd
        df = pd.read_csv(saved, index_col=0)
        assert list(df.columns) == pf.elements
        assert len(df) == 3

    def test_save_with_extension(self, tmp_path):
        pf = _uncharged_2d()
        r = SpreadResult(pf.omega[:2], pf)
        path = str(tmp_path / "out.csv")
        saved = r.save(path)
        assert saved == path

    def test_points_standard_sum_to_one(self):
        pf = _uncharged_2d()
        r = SpreadResult(pf.omega[:10], pf)
        sums = r.points_standard.sum(axis=1)
        np.testing.assert_allclose(sums, 1.0, atol=1e-6)


# ─── Spread construction and basic run ───────────────────────────────────────

class TestSpreadInit:

    def test_init(self):
        pf = _uncharged_2d()
        s = Spread(pf)
        assert s.phase_field is pf
        assert s.known_phases is None

    def test_add_known_phases(self):
        pf = _uncharged_2d()
        s = Spread(pf)
        s.add_known_phases(["FeMn", "FeTi"])
        assert s.known_phases is not None
        assert s.known_phases.shape == (2, pf.constrained_dim)

    def test_add_known_phases_sets_knowns_constrained(self):
        pf = _uncharged_2d()
        s = Spread(pf)
        s.add_known_phases(["FeMn"])
        assert hasattr(pf, "knowns_constrained")


# ─── run() on all phase field types ──────────────────────────────────────────

@pytest.mark.parametrize("label,pf_fn", ALL_PFS)
class TestSpreadRun:

    def test_returns_spread_result(self, label, pf_fn):
        pf = pf_fn()
        result = Spread(pf).run(n=3, **FAST)
        assert isinstance(result, SpreadResult)

    def test_result_shape(self, label, pf_fn):
        pf = pf_fn()
        n = 5
        result = Spread(pf).run(n=n, **FAST)
        assert result.points.shape == (n, pf.constrained_dim)
        assert result.points_standard.shape == (n, pf.nelements)

    def test_result_elements_match(self, label, pf_fn):
        pf = pf_fn()
        result = Spread(pf).run(n=3, **FAST)
        assert result.elements == pf.elements

    def test_standard_fractions_sum_to_one(self, label, pf_fn):
        """All suggested compositions must be valid (fractions sum to 1)."""
        pf = pf_fn()
        result = Spread(pf).run(n=5, **FAST)
        sums = result.points_standard.sum(axis=1)
        np.testing.assert_allclose(sums, 1.0, atol=1e-4,
            err_msg=f"{label}: standard fractions do not sum to 1")

    def test_points_near_omega(self, label, pf_fn):
        """Each suggested point should lie on or very near the omega grid.
        Tolerance scales with dimension: 3D grids are coarser so points
        can be up to ~0.2 from the nearest grid point."""
        pf = pf_fn()
        tol = 0.05 if pf.constrained_dim == 2 else 0.2
        result = Spread(pf).run(n=5, **FAST)
        for pt in result.points:
            dists = np.linalg.norm(pf.omega - pt, axis=1)
            assert dists.min() < tol, (
                f"{label}: suggested point {pt} too far from omega "
                f"(min dist={dists.min():.4f}, tol={tol})")

    def test_with_known_phases_still_runs(self, label, pf_fn):
        pf = pf_fn()
        # Use the first corner as a known phase
        corner_str = pf.get_composition_strings(pf.corners[:1], out='txt')
        s = Spread(pf)
        s.add_known_phases([corner_str])
        result = s.run(n=3, **FAST)
        assert result.points.shape[0] == 3

    def test_n_equals_one(self, label, pf_fn):
        pf = pf_fn()
        result = Spread(pf).run(n=1, **FAST)
        assert result.points.shape == (1, pf.constrained_dim)


# ─── Charged / non-simplex specific ──────────────────────────────────────────

class TestNonSimplexGeometry:
    """
    Charged and custom-constraint phase fields are NOT simplices.
    Their corners are charge-balanced compositions, not pure elements,
    and the omega grid is a strict subset of the full compositional simplex.
    The Spread algorithm must work correctly on these non-simplex polytopes.
    """

    def test_charged_corners_are_not_pure_elements(self):
        """In a charged field, corners should not be identity vectors."""
        pf = _charged_2d()
        for corner in pf.corners:
            assert not np.any(np.isclose(corner, 1.0) & np.all(
                np.isclose(np.delete(corner, np.argmax(corner)), 0.0)
            )), "Corner is a pure element — charged field should have mixed corners"

    def test_charged_2d_omega_not_full_simplex(self):
        """Omega of a charged field is smaller than the full simplex."""
        pf_charged = _charged_2d()
        pf_uncharged = Phase_Field()
        pf_uncharged.setup_uncharged(
            pf_charged.elements, n_points=pf_charged.omega.shape[0]
        )
        assert pf_charged.omega.shape[0] < pf_uncharged.omega.shape[0], \
            "Charged field omega should be smaller than uncharged simplex"

    def test_spread_on_charged_2d(self):
        pf = _charged_2d()
        result = Spread(pf).run(n=5, **FAST)
        assert result.points.shape == (5, 2)
        np.testing.assert_allclose(
            result.points_standard.sum(axis=1), 1.0, atol=1e-4)

    def test_spread_on_charge_ranges(self):
        """Charge-ranges polytope is wider than fixed-charge — must still work."""
        pf = _charge_ranges_3d()
        result = Spread(pf).run(n=5, **FAST)
        assert result.points.shape[0] == 5

    def test_spread_on_spinel(self):
        """Spinel custom constraint fixes O at 4/7 — not a simplex."""
        pf = _custom_spinel()
        result = Spread(pf).run(n=4, **FAST)
        assert result.points.shape[0] == 4
        # O fraction should be fixed at ~4/7 for all suggested points
        o_idx = pf.elements.index("O")
        o_fracs = result.points_standard[:, o_idx]
        np.testing.assert_allclose(o_fracs, 4/7, atol=0.02,
            err_msg="Spinel constraint violated: O fraction should be ~4/7")

    def test_charged_3d(self):
        pf = _charged_3d()
        result = Spread(pf).run(n=5, **FAST)
        assert result.points.shape == (5, 3)


# ─── Fixed sites / known phases ──────────────────────────────────────────────

class TestFixedSites:

    def test_fixed_sites_include_corners(self):
        pf = _uncharged_2d()
        s = Spread(pf)
        fixed = s._build_fixed_sites()
        corners_c = pf.convert_to_constrained_basis(pf.corners)
        for corner in corners_c:
            dists = np.linalg.norm(fixed - corner, axis=1)
            assert dists.min() < 1e-6, "Corner not found in fixed sites"

    def test_known_phases_added_to_fixed_sites(self):
        pf = _uncharged_2d()
        s = Spread(pf)
        s.add_known_phases(["FeMn", "FeTi"])
        fixed = s._build_fixed_sites()
        corners_c = pf.convert_to_constrained_basis(pf.corners)
        assert len(fixed) == len(corners_c) + 2

    def test_spread_avoids_known_phases(self):
        """Suggested points should stay away from known phases."""
        pf = _uncharged_2d()
        s = Spread(pf)
        s.add_known_phases(["FeMn", "FeTi"])
        result = s.run(n=8, num_repeats=10, max_iter=100)
        for pt in result.points:
            for known in s.known_phases:
                dist = np.linalg.norm(pt - known)
                assert dist > 0.01, \
                    f"Suggested point too close to known phase (dist={dist:.4f})"


# ─── simplify_to_precursors ───────────────────────────────────────────────────

class TestSimplifyToPrecursors:

    def test_returns_spread_result_with_precursors(self):
        pf = Phase_Field()
        pf.setup_uncharged(["Fe", "Mn", "Ti"], precursors=["Fe3Mn", "FeTi2", "MnTi"])
        s = Spread(pf)
        result = s.run(n=5, **FAST)
        r2 = s.simplify_to_precursors(result, accuracy=0)
        assert isinstance(r2, SpreadResultWithPrecursors)

    def test_precursor_amounts_shape(self):
        pf = Phase_Field()
        pf.setup_uncharged(["Fe", "Mn", "Ti"], precursors=["Fe3Mn", "FeTi2", "MnTi"])
        s = Spread(pf)
        result = s.run(n=5, **FAST)
        r2 = s.simplify_to_precursors(result)
        assert r2.precursor_amounts.shape == (5, 3)

    def test_precursor_labels_set(self):
        pf = Phase_Field()
        pf.setup_uncharged(["Fe", "Mn", "Ti"], precursors=["Fe3Mn", "FeTi2", "MnTi"])
        s = Spread(pf)
        result = s.run(n=3, **FAST)
        r2 = s.simplify_to_precursors(result)
        assert len(r2.precursor_labels) == 3

    def test_rounded_fractions_sum_to_one(self):
        pf = Phase_Field()
        pf.setup_uncharged(["Fe", "Mn", "Ti"], precursors=["Fe3Mn", "FeTi2", "MnTi"])
        s = Spread(pf)
        result = s.run(n=5, **FAST)
        r2 = s.simplify_to_precursors(result, accuracy=2)
        sums = r2.points_standard.sum(axis=1)
        np.testing.assert_allclose(sums, 1.0, atol=1e-3)

    def test_precursor_csv_includes_amounts(self, tmp_path):
        pf = Phase_Field()
        pf.setup_uncharged(["Fe", "Mn", "Ti"], precursors=["Fe3Mn", "FeTi2", "MnTi"])
        s = Spread(pf)
        result = s.run(n=3, **FAST)
        r2 = s.simplify_to_precursors(result)
        saved = r2.save(str(tmp_path / "pre"))
        import pandas as pd
        df = pd.read_csv(saved, index_col=0)
        # Should have element columns + precursor columns
        assert len(df.columns) == len(pf.elements) + len(r2.precursor_labels)


# ─── evaluate_spread (non-display) ───────────────────────────────────────────

class TestEvaluateSpread:

    def test_does_not_raise_uncharged(self, monkeypatch):
        """evaluate_spread should run without error (block display)."""
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        monkeypatch.setattr(plt, 'show', lambda: None)

        pf = _uncharged_2d()
        s = Spread(pf)
        result = s.run(n=5, **FAST)
        s.evaluate_spread(result)   # should not raise

    def test_does_not_raise_charged(self, monkeypatch):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        monkeypatch.setattr(plt, 'show', lambda: None)

        pf = _charged_2d()
        s = Spread(pf)
        result = s.run(n=5, **FAST)
        s.evaluate_spread(result)

    def test_does_not_raise_spinel(self, monkeypatch):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        monkeypatch.setattr(plt, 'show', lambda: None)

        pf = _custom_spinel()
        s = Spread(pf)
        result = s.run(n=4, **FAST)
        s.evaluate_spread(result)
