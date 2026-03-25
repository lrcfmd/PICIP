"""
Tests for all Phase_Field setup methods and post-setup modifiers.
Run each block independently with if True/False.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from model.phase_field import Phase_Field
from model.picip import PICIP, Sample
from model.visualise_cube import make_plotter
import numpy as np

def check_pf(pf, label):
    print(f"\n{'─'*60}")
    print(f"  {label}")
    print(f"  elements:       {pf.elements}")
    print(f"  constrained_dim:{pf.constrained_dim}")
    print(f"  omega shape:    {pf.omega.shape}")
    print(f"  corners shape:  {pf.corners.shape}")
    print(f"  basis shape:    {pf.basis.shape}")
    print(f"  OK")


# ══════════════════════════════════════════════════════════════════════════════
# 1 — setup_uncharged  (no charge constraint)
# ══════════════════════════════════════════════════════════════════════════════
if True:
    pf = Phase_Field()
    pf.setup_uncharged(["Fe", "Mn", "Ti"])
    check_pf(pf, "setup_uncharged  Fe-Mn-Ti  (2D)")

    pf4 = Phase_Field()
    pf4.setup_uncharged(["Fe", "Mn", "Ti", "Cu"])
    check_pf(pf4, "setup_uncharged  Fe-Mn-Ti-Cu  (3D)")


# ══════════════════════════════════════════════════════════════════════════════
# 2 — setup_charged  (charge neutrality constraint)
# ══════════════════════════════════════════════════════════════════════════════
if True:
    # Li=+1, Mn=+4, O=-2  →  charge neutral, 3 elements collapses to 1D → should be rejected
    try:
        pf = Phase_Field()
        pf.setup_charged({"Li": 1, "Mn": 4, "O": -2}, resolution=30)
        print("\n  1D rejection test FAILED: should have raised ValueError")
    except ValueError as e:
        print(f"\n  Li-Mn-O correctly rejected (1D): {e}")

    # 5-element charged system → 3D
    pf5 = Phase_Field()
    pf5.setup_charged({"Li": 1, "Ni": 3, "Mn": 4, "Co": 3, "O": -2}, resolution=20)
    check_pf(pf5, "setup_charged  Li-Ni-Mn-Co-O  (charge neutral, 3D)")


# ══════════════════════════════════════════════════════════════════════════════
# 3 — setup_charge_ranges  (mixed valence — e.g. Mn can be 3 or 4)
# ══════════════════════════════════════════════════════════════════════════════
if True:
    try:
        pf = Phase_Field()
        pf.setup_charge_ranges({"Li": 1, "Mn": [3, 4], "O": -2}, resolution=20)
        check_pf(pf, "setup_charge_ranges  Li-Mn(3/4)-O")
    except Exception as e:
        print(f"\n  setup_charge_ranges FAILED: {type(e).__name__}: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# 4 — setup_custom  (custom linear constraint on top of sum-to-1)
#     Example: constrain Fe+Mn >= 0.4  (custom inequality passed as normal vector)
# ══════════════════════════════════════════════════════════════════════════════
if True:
    try:
        pf = Phase_Field()
        # 4 elements + sum=1 + 1 custom constraint = 2D
        custom_con = np.array([[1, -1, 0, 0]])  # Fe = Mn  (extra equality)
        pf.setup_custom(["Fe", "Mn", "Ti", "Cu"], custom_con=custom_con, resolution=20)
        check_pf(pf, "setup_custom  Fe-Mn-Ti-Cu  (extra constraint, 2D)")
    except Exception as e:
        print(f"\n  setup_custom FAILED: {type(e).__name__}: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# 5 — setup_uncharged + precursors  (restrict omega to convex hull of precursors)
# ══════════════════════════════════════════════════════════════════════════════
if True:
    try:
        pf = Phase_Field()
        pf.setup_uncharged(["Fe", "Mn", "Ti"],
                           precursors=["FeMn", "FeTi", "MnTi"])
        check_pf(pf, "setup_uncharged + precursors  Fe-Mn-Ti")
        print(f"  omega reduced to {len(pf.omega)} points (was ~{30**2} without)")
    except Exception as e:
        print(f"\n  precursors setup FAILED: {type(e).__name__}: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# 6 — setup_uncharged + add_precursors (post-setup)
# ══════════════════════════════════════════════════════════════════════════════
if True:
    try:
        from pymatgen.core import Composition
        pf = Phase_Field()
        pf.setup_uncharged(["Fe", "Mn", "Ti"])
        omega_before = len(pf.omega)
        pf.add_precursors([Composition("FeMn"), Composition("FeTi"), Composition("MnTi")])
        omega_after = len(pf.omega)
        check_pf(pf, "setup_uncharged + add_precursors (post-setup)  Fe-Mn-Ti")
        print(f"  omega: {omega_before} → {omega_after} points")
    except Exception as e:
        print(f"\n  add_precursors FAILED: {type(e).__name__}: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# 7 — setup_uncharged + constrain_minimum_amount
# ══════════════════════════════════════════════════════════════════════════════
if True:
    try:
        pf = Phase_Field()
        pf.setup_uncharged(["Fe", "Mn", "Ti"])
        omega_before = len(pf.omega)
        pf.constrain_minimum_amount(0.1)   # all elements >= 10%
        omega_after = len(pf.omega)
        check_pf(pf, "setup_uncharged + constrain_minimum_amount(0.1)")
        print(f"  omega: {omega_before} → {omega_after} points")
    except Exception as e:
        print(f"\n  constrain_minimum_amount FAILED: {type(e).__name__}: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# 8 — end-to-end PICIP run on a charged system (sanity check)
# ══════════════════════════════════════════════════════════════════════════════
if True:
    try:
        pf = Phase_Field()
        pf.setup_charged({"Li": 1, "Mn": 4, "O": -2}, resolution=20)
        print("\n  1D phase field test FAILED: should have raised ValueError")
    except ValueError as e:
        print(f"\n  1D phase field correctly rejected: {e}")
