"""
PICIP Test Suite
================
Automated tests (no graphics) and visual tests (opens browser figures).

Usage
-----
  python test_picip.py             # automated only
  python test_picip.py --visual    # automated + visual inspection figures
"""

import numpy as np
from pymatgen.core import Composition

from picip import Phase_Field, PICIP, Sample, make_plotter


# ── shared fixtures ────────────────────────────────────────────────────────────

def _pf_2d():
    pf = Phase_Field()
    pf.setup_uncharged(["Fe", "Mn", "Ti"])
    return pf

def _pf_3d():
    pf = Phase_Field()
    pf.setup_uncharged(["Fe", "Mn", "Ti", "Cu"])
    return pf

def _sample_2d(name, err=0.3):
    s = Sample(name, "Fe2Mn4Ti4")
    s.add_knowns(["FeMn", "FeTi"])
    s.add_molar_weights([0.3, 0.7])
    s.set_predicted_error(err)
    return s

def _sample_3d(name, err=0.3):
    s = Sample(name, "Fe2Mn2Ti2Cu4")
    s.add_knowns(["FeMn", "FeTi", "FeCu"])
    s.add_molar_weights([0.25, 0.35, 0.40])
    s.set_predicted_error(err)
    return s


# ── 2-D phase field fixtures (setup variants) ──────────────────────────────────
# Each returns (pf, sample_a, sample_b) with crossing-ray samples.

def _fixture_charged():
    """Charged Fe-Mn-Cu-O (O=0.5). Four elements, charge neutrality → 2-D."""
    pf = Phase_Field()
    pf.setup_charged({"Fe": 2, "Mn": 2, "Cu": 2, "O": -2})
    comp = Composition({"Fe": 0.15, "Mn": 0.10, "Cu": 0.25, "O": 0.5})
    s_a = Sample("a", comp); s_a.add_knowns(["FeO", "CuO"]); s_a.add_molar_weights([0.3, 0.7]); s_a.set_predicted_error(0.3)
    s_b = Sample("b", comp); s_b.add_knowns(["FeO", "CuO"]); s_b.add_molar_weights([0.7, 0.3]); s_b.set_predicted_error(0.3)
    return pf, s_a, s_b

def _fixture_custom():
    """Spinel Li-Fe-Mn-O with 3:4 cation:anion constraint (O=4/7) → 2-D."""
    pf = Phase_Field()
    pf.setup_custom(["Li", "Fe", "Mn", "O"], custom_con=np.array([[4, 4, 4, -3]]))
    # K1=LiFe2O4, K2=LiMn2O4; unknown U interior; both samples mix (K1, K2, U)
    # with opposite K1:K2 ratios so their rays cross near U
    K1 = np.array([1/7, 2/7,   0,   4/7])
    K2 = np.array([1/7,   0, 2/7,   4/7])
    U  = np.array([0.05, 0.05, 3/7 - 0.10, 4/7])
    ca = 0.24*K1 + 0.56*K2 + 0.20*U    # molar split K1:K2 = 3:7
    cb = 0.56*K1 + 0.24*K2 + 0.20*U    # molar split K1:K2 = 7:3
    s_a = Sample("a", Composition({"Li": ca[0], "Fe": ca[1], "Mn": ca[2], "O": ca[3]}))
    s_a.add_knowns(["LiFe2O4", "LiMn2O4"]); s_a.add_molar_weights([0.3, 0.7]); s_a.set_predicted_error(0.3)
    s_b = Sample("b", Composition({"Li": cb[0], "Fe": cb[1], "Mn": cb[2], "O": cb[3]}))
    s_b.add_knowns(["LiFe2O4", "LiMn2O4"]); s_b.add_molar_weights([0.7, 0.3]); s_b.set_predicted_error(0.3)
    return pf, s_a, s_b

def _fixture_uncharged_cut():
    """Uncharged Fe-Mn-Ti with precursors cutting to a sub-triangle."""
    pf = Phase_Field()
    pf.setup_uncharged(["Fe", "Mn", "Ti"], precursors=["Fe2Mn", "FeTi2", "Mn2Ti"])
    comp = Composition({"Fe": 1/3, "Mn": 1/3, "Ti": 1/3})
    s_a = Sample("a", comp); s_a.add_knowns(["Fe2Mn", "FeTi2"]); s_a.add_molar_weights([0.3, 0.7]); s_a.set_predicted_error(0.3)
    s_b = Sample("b", comp); s_b.add_knowns(["Fe2Mn", "FeTi2"]); s_b.add_molar_weights([0.7, 0.3]); s_b.set_predicted_error(0.3)
    return pf, s_a, s_b

def _fixture_charged_cut():
    """Charged Fe-Mn-Cu-O with precursors cutting to a sub-triangle."""
    pf = Phase_Field()
    pf.setup_charged({"Fe": 2, "Mn": 2, "Cu": 2, "O": -2},
                     precursors=["FeMnO2", "FeCuO2", "MnCuO2"])
    comp = Composition({"Fe": 1/6, "Mn": 1/6, "Cu": 1/6, "O": 0.5})
    s_a = Sample("a", comp); s_a.add_knowns(["FeMnO2", "FeCuO2"]); s_a.add_molar_weights([0.3, 0.7]); s_a.set_predicted_error(0.3)
    s_b = Sample("b", comp); s_b.add_knowns(["FeMnO2", "FeCuO2"]); s_b.add_molar_weights([0.7, 0.3]); s_b.set_predicted_error(0.3)
    return pf, s_a, s_b

def _fixture_custom_cut():
    """Spinel Li-Fe-Mn-O with precursors cutting to a sub-triangle."""
    pf = Phase_Field()
    pf.setup_custom(["Li", "Fe", "Mn", "O"],
                    custom_con=np.array([[4, 4, 4, -3]]),
                    precursors=["LiFe2O4", "Fe3O4", "LiFeMnO4"])
    # Centroid of the three precursors
    comp = Composition({"Li": 2/21, "Fe": 2/7, "Mn": 1/21, "O": 4/7})
    s_a = Sample("a", comp); s_a.add_knowns(["LiFe2O4", "LiFeMnO4"]); s_a.add_molar_weights([0.3, 0.7]); s_a.set_predicted_error(0.3)
    s_b = Sample("b", comp); s_b.add_knowns(["LiFe2O4", "LiFeMnO4"]); s_b.add_molar_weights([0.7, 0.3]); s_b.set_predicted_error(0.3)
    return pf, s_a, s_b

def _fixture_uncharged_reduce():
    """Uncharged 4-element Fe-Mn-Ti-Cu with Cu=0 precursors reducing 3-D → 2-D."""
    pf = Phase_Field()
    pf.setup_uncharged(["Fe", "Mn", "Ti", "Cu"], precursors=["FeMn", "FeTi", "MnTi"])
    comp = Composition({"Fe": 0.30, "Mn": 0.35, "Ti": 0.35, "Cu": 0.0})
    s_a = Sample("a", comp); s_a.add_knowns(["FeMn", "FeTi"]); s_a.add_molar_weights([0.3, 0.7]); s_a.set_predicted_error(0.3)
    s_b = Sample("b", comp); s_b.add_knowns(["FeMn", "FeTi"]); s_b.add_molar_weights([0.7, 0.3]); s_b.set_predicted_error(0.3)
    return pf, s_a, s_b

def _fixture_charged_reduce():
    """Charged 5-element Fe-Mn-Ni-Cu-O with Cu=0 precursors reducing 3-D → 2-D."""
    pf = Phase_Field()
    pf.setup_charged({"Fe": 2, "Mn": 2, "Ni": 2, "Cu": 2, "O": -2},
                     precursors=["FeO", "MnO", "NiO"])
    comp = Composition({"Fe": 0.10, "Mn": 0.15, "Ni": 0.25, "Cu": 0.0, "O": 0.5})
    s_a = Sample("a", comp); s_a.add_knowns(["FeO", "NiO"]); s_a.add_molar_weights([0.3, 0.7]); s_a.set_predicted_error(0.3)
    s_b = Sample("b", comp); s_b.add_knowns(["FeO", "NiO"]); s_b.add_molar_weights([0.7, 0.3]); s_b.set_predicted_error(0.3)
    return pf, s_a, s_b

def _fixture_custom_reduce():
    """Custom 5-element spinel Li-Fe-Mn-Co-O with Co=0 precursors reducing 3-D → 2-D."""
    pf = Phase_Field()
    pf.setup_custom(["Li", "Fe", "Mn", "Co", "O"],
                    custom_con=np.array([[4, 4, 4, 4, -3]]),
                    precursors=["LiFe2O4", "LiMn2O4", "Fe3O4"])
    # Same crossing-ray design as _fixture_custom (Co=0 throughout)
    K1 = np.array([1/7, 2/7, 0, 0, 4/7])
    K2 = np.array([1/7, 0, 2/7, 0, 4/7])
    U  = np.array([0.05, 0.05, 3/7 - 0.10, 0, 4/7])
    ca = 0.24*K1 + 0.56*K2 + 0.20*U
    cb = 0.56*K1 + 0.24*K2 + 0.20*U
    s_a = Sample("a", Composition({"Li": ca[0], "Fe": ca[1], "Mn": ca[2], "O": ca[4]}))
    s_a.add_knowns(["LiFe2O4", "LiMn2O4"]); s_a.add_molar_weights([0.3, 0.7]); s_a.set_predicted_error(0.3)
    s_b = Sample("b", Composition({"Li": cb[0], "Fe": cb[1], "Mn": cb[2], "O": cb[4]}))
    s_b.add_knowns(["LiFe2O4", "LiMn2O4"]); s_b.add_molar_weights([0.7, 0.3]); s_b.set_predicted_error(0.3)
    return pf, s_a, s_b


# ── 3-D phase field fixtures (setup variants) ──────────────────────────────────
# Each returns (pf, sample_a, sample_b) with crossing-ray 3-known samples.
# For 3D (3 knowns), the unknown U must be off the K1-K2-K3 plane in constrained space.
# Knowns K1,K2,K3 all lie in the Cu=0 face, so U must have Cu>0 (or Ni>0 for reduce variants).

def _compose_3d(K1, K2, K3, U, ra, rb):
    """Compute two crossing-ray sample compositions from 3 knowns + unknown U.
    ra/rb: relative molar weights [w1,w2,w3] for samples a/b (normalized to 0.8; wU=0.2).
    """
    wa = np.array(ra, float); wa = wa / wa.sum() * 0.8
    wb = np.array(rb, float); wb = wb / wb.sum() * 0.8
    return wa[0]*K1 + wa[1]*K2 + wa[2]*K3 + 0.2*U, wb[0]*K1 + wb[1]*K2 + wb[2]*K3 + 0.2*U


def _fixture_3d_charged():
    """Charged Fe(3)-Mn(2)-Ti(4)-Cu(2)-O(-2): 5 elements → 3-D."""
    pf = Phase_Field()
    pf.setup_charged({"Fe": 3, "Mn": 2, "Ti": 4, "Cu": 2, "O": -2})
    # K1=Fe2O3, K2=MnO, K3=TiO2 (all Cu=0); U has Cu>0 so sample is off the K1-K2-K3 plane
    # Constraint: 5Fe+4Mn+6Ti+4Cu=2 → Fe=0.1,Mn=0.1,Ti=0.05 → Cu=(2-0.5-0.4-0.3)/4=0.2
    K1 = np.array([0.4,  0.0, 0.0,  0.0,  0.6 ])   # Fe2O3
    K2 = np.array([0.0,  0.5, 0.0,  0.0,  0.5 ])   # MnO
    K3 = np.array([0.0,  0.0, 1/3,  0.0,  2/3 ])   # TiO2
    U  = np.array([0.1,  0.1, 0.05, 0.2,  0.55])   # Cu=0.2>0
    ca, cb = _compose_3d(K1, K2, K3, U, [3,3,4], [4,4,2])
    els = ["Fe","Mn","Ti","Cu","O"]
    s_a = Sample("a", Composition(dict(zip(els, ca)))); s_a.add_knowns(["Fe2O3","MnO","TiO2"]); s_a.add_molar_weights([0.3,0.3,0.4]); s_a.set_predicted_error(0.3)
    s_b = Sample("b", Composition(dict(zip(els, cb)))); s_b.add_knowns(["Fe2O3","MnO","TiO2"]); s_b.add_molar_weights([0.4,0.4,0.2]); s_b.set_predicted_error(0.3)
    return pf, s_a, s_b


def _fixture_3d_charge_ranges():
    """Charge-ranges Fe([2,3])-Mn([2,3])-Ti([3,4])-O(-2): 4 elements, uncharged geometry → 3-D."""
    pf = Phase_Field()
    pf.setup_charge_ranges({"Fe": [2, 3], "Mn": [2, 3], "Ti": [3, 4], "O": -2})
    K1 = np.array([0.4, 0.0, 0.0, 0.6])   # Fe2O3
    K2 = np.array([0.0, 0.5, 0.0, 0.5])   # MnO
    K3 = np.array([0.0, 0.0, 1/3, 2/3])   # TiO2
    U  = np.array([0.2, 0.2, 0.1, 0.5])   # interior point
    ca, cb = _compose_3d(K1, K2, K3, U, [3,3,4], [4,4,2])
    els = ["Fe","Mn","Ti","O"]
    s_a = Sample("a", Composition(dict(zip(els, ca)))); s_a.add_knowns(["Fe2O3","MnO","TiO2"]); s_a.add_molar_weights([0.3,0.3,0.4]); s_a.set_predicted_error(0.3)
    s_b = Sample("b", Composition(dict(zip(els, cb)))); s_b.add_knowns(["Fe2O3","MnO","TiO2"]); s_b.add_molar_weights([0.4,0.4,0.2]); s_b.set_predicted_error(0.3)
    return pf, s_a, s_b


def _fixture_3d_custom():
    """Custom spinel Fe-Mn-Ti-Cu-O with 4(Fe+Mn+Ti+Cu)=3O: 5 elements → 3-D."""
    pf = Phase_Field()
    pf.setup_custom(["Fe","Mn","Ti","Cu","O"], custom_con=np.array([[4,4,4,4,-3]]))
    # M3O4 compounds (K1-K3 have Cu=0); U has equal cations including Cu>0
    K1 = np.array([3/7, 0,   0,    0,    4/7])   # Fe3O4
    K2 = np.array([0,   3/7, 0,    0,    4/7])   # Mn3O4
    K3 = np.array([0,   0,   3/7,  0,    4/7])   # Ti3O4
    U  = np.array([3/28,3/28,3/28, 3/28, 4/7])   # Cu=3/28>0
    ca, cb = _compose_3d(K1, K2, K3, U, [3,3,4], [4,4,2])
    els = ["Fe","Mn","Ti","Cu","O"]
    s_a = Sample("a", Composition(dict(zip(els, ca)))); s_a.add_knowns(["Fe3O4","Mn3O4","Ti3O4"]); s_a.add_molar_weights([0.3,0.3,0.4]); s_a.set_predicted_error(0.3)
    s_b = Sample("b", Composition(dict(zip(els, cb)))); s_b.add_knowns(["Fe3O4","Mn3O4","Ti3O4"]); s_b.add_molar_weights([0.4,0.4,0.2]); s_b.set_predicted_error(0.3)
    return pf, s_a, s_b


def _fixture_3d_uncharged_cut():
    """Uncharged Fe-Mn-Ti-Cu with sub-tetrahedron precursors."""
    pf = Phase_Field()
    pf.setup_uncharged(["Fe","Mn","Ti","Cu"], precursors=["Fe3Mn","FeTi3","MnCu","FeMnTiCu"])
    K1 = np.array([0.75, 0.25, 0.0,  0.0 ])   # Fe3Mn
    K2 = np.array([0.25, 0.0,  0.75, 0.0 ])   # FeTi3
    K3 = np.array([0.0,  0.5,  0.0,  0.5 ])   # MnCu
    U  = np.array([0.25, 0.25, 0.25, 0.25])   # FeMnTiCu — inside sub-tetrahedron
    ca, cb = _compose_3d(K1, K2, K3, U, [3,3,4], [4,4,2])
    els = ["Fe","Mn","Ti","Cu"]
    s_a = Sample("a", Composition(dict(zip(els, ca)))); s_a.add_knowns(["Fe3Mn","FeTi3","MnCu"]); s_a.add_molar_weights([0.3,0.3,0.4]); s_a.set_predicted_error(0.3)
    s_b = Sample("b", Composition(dict(zip(els, cb)))); s_b.add_knowns(["Fe3Mn","FeTi3","MnCu"]); s_b.add_molar_weights([0.4,0.4,0.2]); s_b.set_predicted_error(0.3)
    return pf, s_a, s_b


def _fixture_3d_charged_cut():
    """Charged Fe(3)-Mn(2)-Ti(4)-Cu(2)-O(-2) with precursors restricting region."""
    pf = Phase_Field()
    pf.setup_charged({"Fe": 3, "Mn": 2, "Ti": 4, "Cu": 2, "O": -2},
                     precursors=["Fe2O3","MnO","TiO2","Fe2MnTiCuO7"])
    K1 = np.array([0.4,  0.0, 0.0,  0.0,  0.6 ])
    K2 = np.array([0.0,  0.5, 0.0,  0.0,  0.5 ])
    K3 = np.array([0.0,  0.0, 1/3,  0.0,  2/3 ])
    U  = np.array([0.1,  0.1, 0.05, 0.2,  0.55])
    ca, cb = _compose_3d(K1, K2, K3, U, [3,3,4], [4,4,2])
    els = ["Fe","Mn","Ti","Cu","O"]
    s_a = Sample("a", Composition(dict(zip(els, ca)))); s_a.add_knowns(["Fe2O3","MnO","TiO2"]); s_a.add_molar_weights([0.3,0.3,0.4]); s_a.set_predicted_error(0.3)
    s_b = Sample("b", Composition(dict(zip(els, cb)))); s_b.add_knowns(["Fe2O3","MnO","TiO2"]); s_b.add_molar_weights([0.4,0.4,0.2]); s_b.set_predicted_error(0.3)
    return pf, s_a, s_b


def _fixture_3d_custom_cut():
    """Custom spinel Fe-Mn-Ti-Cu-O with sub-region precursors."""
    pf = Phase_Field()
    pf.setup_custom(["Fe","Mn","Ti","Cu","O"], custom_con=np.array([[4,4,4,4,-3]]),
                    precursors=["Fe3O4","Mn3O4","Ti3O4","FeMnTiCuO4"])
    K1 = np.array([3/7, 0,    0,    0,    4/7])
    K2 = np.array([0,   3/7,  0,    0,    4/7])
    K3 = np.array([0,   0,    3/7,  0,    4/7])
    U  = np.array([3/28,3/28, 3/28, 3/28, 4/7])   # Cu=3/28>0
    ca, cb = _compose_3d(K1, K2, K3, U, [3,3,4], [4,4,2])
    els = ["Fe","Mn","Ti","Cu","O"]
    s_a = Sample("a", Composition(dict(zip(els, ca)))); s_a.add_knowns(["Fe3O4","Mn3O4","Ti3O4"]); s_a.add_molar_weights([0.3,0.3,0.4]); s_a.set_predicted_error(0.3)
    s_b = Sample("b", Composition(dict(zip(els, cb)))); s_b.add_knowns(["Fe3O4","Mn3O4","Ti3O4"]); s_b.add_molar_weights([0.4,0.4,0.2]); s_b.set_predicted_error(0.3)
    return pf, s_a, s_b


def _fixture_3d_uncharged_reduce():
    """Uncharged 5-element Fe-Mn-Ti-Cu-Zn (4-D) with Zn=0 precursors reducing to 3-D."""
    pf = Phase_Field()
    pf.setup_uncharged(["Fe","Mn","Ti","Cu","Zn"], precursors=["Fe3Mn","FeTi3","MnCu","FeMnTiCu"])
    K1 = np.array([0.75, 0.25, 0.0,  0.0,  0.0])
    K2 = np.array([0.25, 0.0,  0.75, 0.0,  0.0])
    K3 = np.array([0.0,  0.5,  0.0,  0.5,  0.0])
    U  = np.array([0.25, 0.25, 0.25, 0.25, 0.0])   # Zn=0 throughout
    ca, cb = _compose_3d(K1, K2, K3, U, [3,3,4], [4,4,2])
    els = ["Fe","Mn","Ti","Cu","Zn"]
    s_a = Sample("a", Composition(dict(zip(els, ca)))); s_a.add_knowns(["Fe3Mn","FeTi3","MnCu"]); s_a.add_molar_weights([0.3,0.3,0.4]); s_a.set_predicted_error(0.3)
    s_b = Sample("b", Composition(dict(zip(els, cb)))); s_b.add_knowns(["Fe3Mn","FeTi3","MnCu"]); s_b.add_molar_weights([0.4,0.4,0.2]); s_b.set_predicted_error(0.3)
    return pf, s_a, s_b


def _fixture_3d_charged_reduce():
    """Charged Fe(3)-Mn(2)-Ti(4)-Ni(2)-Cu(2)-O(-2) (6 elements → 4-D) with Cu=0 precursors → 3-D."""
    pf = Phase_Field()
    pf.setup_charged({"Fe": 3, "Mn": 2, "Ti": 4, "Ni": 2, "Cu": 2, "O": -2},
                     precursors=["Fe2O3","MnO","TiO2","FeNiO2"])
    # All precursors have Cu=0; U has Ni>0 to be off the K1-K2-K3 plane
    # Constraint: 5Fe+4Mn+6Ti+4Ni+4Cu=2 → Fe=0.1,Mn=0.1,Ti=0.05,Cu=0 → Ni=(2-0.5-0.4-0.3)/4=0.2
    K1 = np.array([0.4,  0.0, 0.0,  0.0,  0.0,  0.6 ])
    K2 = np.array([0.0,  0.5, 0.0,  0.0,  0.0,  0.5 ])
    K3 = np.array([0.0,  0.0, 1/3,  0.0,  0.0,  2/3 ])
    U  = np.array([0.1,  0.1, 0.05, 0.2,  0.0,  0.55])
    ca, cb = _compose_3d(K1, K2, K3, U, [3,3,4], [4,4,2])
    s_a = Sample("a", Composition({"Fe":ca[0],"Mn":ca[1],"Ti":ca[2],"Ni":ca[3],"O":ca[5]}))
    s_a.add_knowns(["Fe2O3","MnO","TiO2"]); s_a.add_molar_weights([0.3,0.3,0.4]); s_a.set_predicted_error(0.3)
    s_b = Sample("b", Composition({"Fe":cb[0],"Mn":cb[1],"Ti":cb[2],"Ni":cb[3],"O":cb[5]}))
    s_b.add_knowns(["Fe2O3","MnO","TiO2"]); s_b.add_molar_weights([0.4,0.4,0.2]); s_b.set_predicted_error(0.3)
    return pf, s_a, s_b


def _fixture_3d_custom_reduce():
    """Custom spinel Fe-Mn-Ti-Ni-Cu-O (6 elements → 4-D) with Cu=0 precursors → 3-D."""
    pf = Phase_Field()
    pf.setup_custom(["Fe","Mn","Ti","Ni","Cu","O"], custom_con=np.array([[4,4,4,4,4,-3]]),
                    precursors=["Fe3O4","Mn3O4","Ti3O4","FeNiTiO4"])
    K1 = np.array([3/7,  0,    0,    0,    0,    4/7])
    K2 = np.array([0,    3/7,  0,    0,    0,    4/7])
    K3 = np.array([0,    0,    3/7,  0,    0,    4/7])
    U  = np.array([3/28, 3/28, 3/28, 3/28, 0,    4/7])   # Ni=3/28>0, Cu=0
    ca, cb = _compose_3d(K1, K2, K3, U, [3,3,4], [4,4,2])
    s_a = Sample("a", Composition({"Fe":ca[0],"Mn":ca[1],"Ti":ca[2],"Ni":ca[3],"O":ca[5]}))
    s_a.add_knowns(["Fe3O4","Mn3O4","Ti3O4"]); s_a.add_molar_weights([0.3,0.3,0.4]); s_a.set_predicted_error(0.3)
    s_b = Sample("b", Composition({"Fe":cb[0],"Mn":cb[1],"Ti":cb[2],"Ni":cb[3],"O":cb[5]}))
    s_b.add_knowns(["Fe3O4","Mn3O4","Ti3O4"]); s_b.add_molar_weights([0.4,0.4,0.2]); s_b.set_predicted_error(0.3)
    return pf, s_a, s_b


def _run_2sample(pf, s_a, s_b, label):
    """Run PICIP with two samples; assert non-zero combined density."""
    picip = PICIP(pf)
    picip.add_sample(s_a)
    picip.add_sample(s_b)
    pred = picip.run()
    assert pred.prob_density.sum() > 0, f"{label}: combined density is zero"
    return picip, pred


def _run_2sample_3d(pf, s_a, s_b, label, n_l=30, n_p=10):
    """Run PICIP with two 3-D samples; assert non-zero combined density."""
    picip = PICIP(pf)
    picip.add_sample(s_a)
    picip.add_sample(s_b)
    pred = picip.run(n_l=n_l, n_p=n_p)
    assert pred.prob_density.sum() > 0, f"{label}: combined density is zero"
    return picip, pred


# ── automated tests ────────────────────────────────────────────────────────────

def test_single_sample_2d():
    """Single 2-D sample runs and produces a normalised probability map."""
    pf = _pf_2d()
    s = _sample_2d("s1")
    picip = PICIP(pf)
    picip.add_sample(s)
    pred = picip.run()
    assert pred.prob_density.sum() > 0, "Probability map is all zero"
    assert abs(pred.prob_density.sum() - 1.0) < 1e-6, "Probability map not normalised"
    print("PASS  test_single_sample_2d")


def test_single_sample_3d():
    """Single 3-D sample runs and produces a normalised probability map."""
    pf = _pf_3d()
    s = _sample_3d("s1")
    picip = PICIP(pf)
    picip.add_sample(s)
    pred = picip.run(n_l=30, n_p=10)
    assert pred.prob_density.sum() > 0
    assert abs(pred.prob_density.sum() - 1.0) < 1e-6
    print("PASS  test_single_sample_3d")


def test_multi_sample_nonzero_intersection():
    """Two samples with crossing rays produce a non-zero combined density."""
    pf = _pf_2d()
    # Constructed so rays cross near Mn2Ti
    from pymatgen.core import Composition
    s_a = Sample("a", Composition({"Fe": 0.300, "Mn": 0.357, "Ti": 0.343}))
    s_a.add_knowns(["FeMn", "FeTi"])
    s_a.add_molar_weights([0.3, 0.7])
    s_a.set_predicted_error(0.3)

    s_b = Sample("b", Composition({"Fe": 0.300, "Mn": 0.477, "Ti": 0.223}))
    s_b.add_knowns(["FeMn", "FeTi"])
    s_b.add_molar_weights([0.7, 0.3])
    s_b.set_predicted_error(0.3)

    picip = PICIP(pf)
    picip.add_sample(s_a)
    picip.add_sample(s_b)
    pred = picip.run()

    assert pred.prob_density.sum() > 0, "Combined density is zero despite crossing rays"
    # Combined should be narrower than either individual prediction
    pred_a = picip.run(0)
    pred_b = picip.run(1)
    n_nonzero_a    = (pred_a.prob_density > 0).sum()
    n_nonzero_b    = (pred_b.prob_density > 0).sum()
    n_nonzero_both = (pred.prob_density > 0).sum()
    assert n_nonzero_both < n_nonzero_a, "Combined is not narrower than sample a"
    assert n_nonzero_both < n_nonzero_b, "Combined is not narrower than sample b"
    print("PASS  test_multi_sample_nonzero_intersection")
    return picip, pred, pred_a, pred_b


def test_multi_sample_zero_intersection_autoretry():
    """Two incompatible samples trigger the auto-retry and succeed or fail gracefully."""
    pf = _pf_2d()
    # Deliberately opposite compositions to create a zero intersection with tight error
    from pymatgen.core import Composition
    s_a = Sample("a", Composition({"Fe": 0.1, "Mn": 0.8, "Ti": 0.1}))
    s_a.add_knowns(["FeMn", "FeTi"])
    s_a.add_molar_weights([0.9, 0.1])
    s_a.set_predicted_error(0.01)   # very tight

    s_b = Sample("b", Composition({"Fe": 0.1, "Mn": 0.1, "Ti": 0.8}))
    s_b.add_knowns(["FeMn", "FeTi"])
    s_b.add_molar_weights([0.1, 0.9])
    s_b.set_predicted_error(0.01)   # very tight, opposite direction

    picip = PICIP(pf)
    picip.add_sample(s_a)
    picip.add_sample(s_b)

    try:
        pred = picip.run()
        # If it succeeded, it must have non-zero density
        assert pred.prob_density.sum() > 0
        print("PASS  test_multi_sample_zero_intersection_autoretry  (auto-retry succeeded)")
    except ValueError as e:
        # Acceptable: truly incompatible samples exhaust all retries
        assert "still zero" in str(e), f"Unexpected error: {e}"
        print("PASS  test_multi_sample_zero_intersection_autoretry  (exhausted retries as expected)")


def test_detail_view_multi_sample_raises():
    """plot_detail_view on a multi-sample prediction without sample_index raises ValueError."""
    pf = _pf_2d()
    from pymatgen.core import Composition
    s_a = Sample("a", Composition({"Fe": 0.300, "Mn": 0.357, "Ti": 0.343}))
    s_a.add_knowns(["FeMn", "FeTi"])
    s_a.add_molar_weights([0.3, 0.7])
    s_a.set_predicted_error(0.3)

    s_b = Sample("b", Composition({"Fe": 0.300, "Mn": 0.477, "Ti": 0.223}))
    s_b.add_knowns(["FeMn", "FeTi"])
    s_b.add_molar_weights([0.7, 0.3])
    s_b.set_predicted_error(0.3)

    picip = PICIP(pf)
    picip.add_sample(s_a)
    picip.add_sample(s_b)
    pred = picip.run()

    pl = make_plotter(pf)
    try:
        pl.plot_detail_view(pred)   # should raise — no sample_index specified
        raise AssertionError("Expected ValueError but none raised")
    except ValueError as e:
        assert "single-sample" in str(e)
        print("PASS  test_detail_view_multi_sample_raises")


def test_detail_view_multi_sample_always_raises():
    """plot_detail_view on a multi-sample prediction always raises ValueError."""
    pf = _pf_2d()
    from pymatgen.core import Composition
    s_a = Sample("a", Composition({"Fe": 0.300, "Mn": 0.357, "Ti": 0.343}))
    s_a.add_knowns(["FeMn", "FeTi"])
    s_a.add_molar_weights([0.3, 0.7])
    s_a.set_predicted_error(0.3)

    s_b = Sample("b", Composition({"Fe": 0.300, "Mn": 0.477, "Ti": 0.223}))
    s_b.add_knowns(["FeMn", "FeTi"])
    s_b.add_molar_weights([0.7, 0.3])
    s_b.set_predicted_error(0.3)

    picip = PICIP(pf)
    picip.add_sample(s_a)
    picip.add_sample(s_b)
    pred = picip.run()

    pl = make_plotter(pf)
    try:
        pl.plot_detail_view(pred)
        raise AssertionError("Expected ValueError but none raised")
    except ValueError:
        pass
    print("PASS  test_detail_view_multi_sample_always_raises")


def test_2sample_charged():
    pf, s_a, s_b = _fixture_charged()
    _run_2sample(pf, s_a, s_b, "charged")
    print("PASS  test_2sample_charged")

def test_2sample_custom():
    pf, s_a, s_b = _fixture_custom()
    _run_2sample(pf, s_a, s_b, "custom")
    print("PASS  test_2sample_custom")

def test_2sample_uncharged_cut():
    pf, s_a, s_b = _fixture_uncharged_cut()
    _run_2sample(pf, s_a, s_b, "uncharged_cut")
    print("PASS  test_2sample_uncharged_cut")

def test_2sample_charged_cut():
    pf, s_a, s_b = _fixture_charged_cut()
    _run_2sample(pf, s_a, s_b, "charged_cut")
    print("PASS  test_2sample_charged_cut")

def test_2sample_custom_cut():
    pf, s_a, s_b = _fixture_custom_cut()
    _run_2sample(pf, s_a, s_b, "custom_cut")
    print("PASS  test_2sample_custom_cut")

def test_2sample_uncharged_reduce():
    pf, s_a, s_b = _fixture_uncharged_reduce()
    _run_2sample(pf, s_a, s_b, "uncharged_reduce")
    print("PASS  test_2sample_uncharged_reduce")

def test_2sample_charged_reduce():
    pf, s_a, s_b = _fixture_charged_reduce()
    _run_2sample(pf, s_a, s_b, "charged_reduce")
    print("PASS  test_2sample_charged_reduce")

def test_2sample_custom_reduce():
    pf, s_a, s_b = _fixture_custom_reduce()
    _run_2sample(pf, s_a, s_b, "custom_reduce")
    print("PASS  test_2sample_custom_reduce")


def test_3d_charged():
    pf, s_a, s_b = _fixture_3d_charged()
    _run_2sample_3d(pf, s_a, s_b, "3d_charged")
    print("PASS  test_3d_charged")

def test_3d_charge_ranges():
    pf, s_a, s_b = _fixture_3d_charge_ranges()
    _run_2sample_3d(pf, s_a, s_b, "3d_charge_ranges")
    print("PASS  test_3d_charge_ranges")

def test_3d_custom():
    pf, s_a, s_b = _fixture_3d_custom()
    _run_2sample_3d(pf, s_a, s_b, "3d_custom")
    print("PASS  test_3d_custom")

def test_3d_uncharged_cut():
    pf, s_a, s_b = _fixture_3d_uncharged_cut()
    _run_2sample_3d(pf, s_a, s_b, "3d_uncharged_cut")
    print("PASS  test_3d_uncharged_cut")

def test_3d_charged_cut():
    pf, s_a, s_b = _fixture_3d_charged_cut()
    _run_2sample_3d(pf, s_a, s_b, "3d_charged_cut")
    print("PASS  test_3d_charged_cut")

def test_3d_custom_cut():
    pf, s_a, s_b = _fixture_3d_custom_cut()
    _run_2sample_3d(pf, s_a, s_b, "3d_custom_cut")
    print("PASS  test_3d_custom_cut")

def test_3d_uncharged_reduce():
    pf, s_a, s_b = _fixture_3d_uncharged_reduce()
    _run_2sample_3d(pf, s_a, s_b, "3d_uncharged_reduce")
    print("PASS  test_3d_uncharged_reduce")

def test_3d_charged_reduce():
    pf, s_a, s_b = _fixture_3d_charged_reduce()
    _run_2sample_3d(pf, s_a, s_b, "3d_charged_reduce")
    print("PASS  test_3d_charged_reduce")

def test_3d_custom_reduce():
    pf, s_a, s_b = _fixture_3d_custom_reduce()
    _run_2sample_3d(pf, s_a, s_b, "3d_custom_reduce")
    print("PASS  test_3d_custom_reduce")


def run_automated():
    print("\n── Automated tests ───────────────────────────────────────")
    test_single_sample_2d()
    test_single_sample_3d()
    test_multi_sample_nonzero_intersection()
    test_multi_sample_zero_intersection_autoretry()
    test_detail_view_multi_sample_raises()
    test_detail_view_multi_sample_always_raises()
    test_2sample_charged()
    test_2sample_custom()
    test_2sample_uncharged_cut()
    test_2sample_charged_cut()
    test_2sample_custom_cut()
    test_2sample_uncharged_reduce()
    test_2sample_charged_reduce()
    test_2sample_custom_reduce()
    test_3d_charged()
    test_3d_charge_ranges()
    test_3d_custom()
    test_3d_uncharged_cut()
    test_3d_charged_cut()
    test_3d_custom_cut()
    test_3d_uncharged_reduce()
    test_3d_charged_reduce()
    test_3d_custom_reduce()
    print("── All automated tests passed ────────────────────────────\n")


# ── visual tests ───────────────────────────────────────────────────────────────

def visual_multi_sample_nonzero():
    """Visual: combined prediction is narrower than either individual sample."""
    picip, pred, pred_a, pred_b = test_multi_sample_nonzero_intersection()
    pf = picip.phase_field

    pl = make_plotter(pf)
    pl.plot_prediction_results(pred_a, plot_knowns=True, plot_average_known=True)
    pl.show(title="Visual — multi-sample: sample a only")

    pl = make_plotter(pf)
    pl.plot_prediction_results(pred_b, plot_knowns=True, plot_average_known=True)
    pl.show(title="Visual — multi-sample: sample b only")

    pl = make_plotter(pf)
    pl.plot_prediction_results(pred, plot_knowns=True, plot_average_known=True)
    pl.show(title="Visual — multi-sample: combined (narrower than either)")

    # Detail view: single-sample predictions only (combined raises ValueError)
    pl = make_plotter(pf)
    pl.plot_detail_view(pred_a)
    pl.show(title="Visual — detail view: sample a")

    pl = make_plotter(pf)
    pl.plot_detail_view(pred_b)
    pl.show(title="Visual — detail view: sample b")


def visual_multi_sample_zero_then_retry():
    """Visual: zero intersection triggers auto-retry messages (check terminal output)."""
    pf = _pf_2d()
    from pymatgen.core import Composition
    s_a = Sample("a", Composition({"Fe": 0.1, "Mn": 0.8, "Ti": 0.1}))
    s_a.add_knowns(["FeMn", "FeTi"])
    s_a.add_molar_weights([0.9, 0.1])
    s_a.set_predicted_error(0.01)

    s_b = Sample("b", Composition({"Fe": 0.1, "Mn": 0.1, "Ti": 0.8}))
    s_b.add_knowns(["FeMn", "FeTi"])
    s_b.add_molar_weights([0.1, 0.9])
    s_b.set_predicted_error(0.01)

    picip = PICIP(pf)
    picip.add_sample(s_a)
    picip.add_sample(s_b)
    print("\nExpect auto-retry messages below:")
    try:
        pred = picip.run()
        pl = make_plotter(pf)
        pl.plot_prediction_results(pred, plot_knowns=True, plot_average_known=True)
        pl.show(title="Visual — zero intersection: auto-retry result")
    except ValueError as e:
        print(f"  Retries exhausted: {e}")


def visual_3d_multi_sample():
    """Visual: 3-D two-sample — single prediction, combined prediction, and detail views."""
    from pymatgen.core import Composition
    pf = _pf_3d()

    # Constructed so rays cross near TiCu3
    s_c = Sample("c", Composition({"Fe": 0.180, "Mn": 0.165, "Ti": 0.235, "Cu": 0.420}))
    s_c.add_knowns(["Fe3Mn", "FeTi3", "MnCu"])
    s_c.add_molar_weights([0.30, 0.30, 0.40])
    s_c.set_predicted_error(0.3)

    s_d = Sample("d", Composition({"Fe": 0.150, "Mn": 0.112, "Ti": 0.348, "Cu": 0.390}))
    s_d.add_knowns(["Fe3Mn", "FeTi3", "MnCu"])
    s_d.add_molar_weights([0.15, 0.55, 0.30])
    s_d.set_predicted_error(0.3)

    picip = PICIP(pf)
    picip.add_sample(s_c)
    picip.add_sample(s_d)

    pred_c  = picip.run(0, n_l=50, n_p=10)
    pred_cd = picip.run(n_l=50, n_p=10)

    # Default view — single sample
    pl = make_plotter(pf)
    pl.plot_prediction_results(pred_c, plot_average_known=True)
    pl.show()

    # Default view — combined
    pl = make_plotter(pf)
    pl.plot_prediction_results(pred_cd, plot_average_known=True)
    pl.show()

    # Detail view — single sample
    pl = make_plotter(pf)
    pl.plot_detail_view(pred_c, top_mode='surface', top=0.5)
    pl.show()

    # Detail view — combined raises (as expected)
    try:
        pl = make_plotter(pf)
        pl.plot_detail_view(pred_cd)
        print('FAIL  detail view on combined should have raised ValueError')
    except ValueError as e:
        print(f'PASS  detail view on combined raised: {e}')


def _visual_2sample(fixture_fn, title):
    """Run a 2-sample fixture and show single, combined, and detail plots."""
    pf, s_a, s_b = fixture_fn()
    picip, pred = _run_2sample(pf, s_a, s_b, title)
    pred_a = picip.run(0)

    pl = make_plotter(pf)
    pl.plot_prediction_results(pred_a, plot_average_known=True)
    pl.show(title=f"{title} — single (a)")

    pl = make_plotter(pf)
    pl.plot_prediction_results(pred, plot_average_known=True)
    pl.show(title=f"{title} — combined")

    pl = make_plotter(pf)
    pl.plot_detail_view(pred_a)
    pl.show(title=f"{title} — detail (a)")


def visual_2sample_charged():
    _visual_2sample(_fixture_charged, "charged 2-D")

def visual_2sample_custom():
    _visual_2sample(_fixture_custom, "custom 2-D")

def visual_2sample_uncharged_cut():
    _visual_2sample(_fixture_uncharged_cut, "uncharged + precursors (region cut)")

def visual_2sample_charged_cut():
    _visual_2sample(_fixture_charged_cut, "charged + precursors (region cut)")

def visual_2sample_custom_cut():
    _visual_2sample(_fixture_custom_cut, "custom + precursors (region cut)")

def visual_2sample_uncharged_reduce():
    _visual_2sample(_fixture_uncharged_reduce, "uncharged + precursors (dim reduce)")

def visual_2sample_charged_reduce():
    _visual_2sample(_fixture_charged_reduce, "charged + precursors (dim reduce)")

def visual_2sample_custom_reduce():
    _visual_2sample(_fixture_custom_reduce, "custom + precursors (dim reduce)")


def _visual_3d_fixture(fixture_fn, title):
    """Run a 3-D 2-sample fixture and show single (a), combined, and detail (a) plots."""
    pf, s_a, s_b = fixture_fn()
    picip, pred = _run_2sample_3d(pf, s_a, s_b, title)
    pred_a = picip.run(0, n_l=30, n_p=10)

    pl = make_plotter(pf)
    pl.plot_prediction_results(pred_a, plot_average_known=True)
    pl.show(title=f"{title} — single (a)")

    pl = make_plotter(pf)
    pl.plot_prediction_results(pred, plot_average_known=True)
    pl.show(title=f"{title} — combined")

    pl = make_plotter(pf)
    pl.plot_detail_view(pred_a, top_mode='surface', top=0.5)
    pl.show(title=f"{title} — detail (a)")


def visual_3d_charged():
    _visual_3d_fixture(_fixture_3d_charged, "charged 3-D")

def visual_3d_charge_ranges():
    _visual_3d_fixture(_fixture_3d_charge_ranges, "charge ranges 3-D")

def visual_3d_custom():
    _visual_3d_fixture(_fixture_3d_custom, "custom 3-D")

def visual_3d_uncharged_cut():
    _visual_3d_fixture(_fixture_3d_uncharged_cut, "uncharged + precursors cut 3-D")

def visual_3d_charged_cut():
    _visual_3d_fixture(_fixture_3d_charged_cut, "charged + precursors cut 3-D")

def visual_3d_custom_cut():
    _visual_3d_fixture(_fixture_3d_custom_cut, "custom + precursors cut 3-D")

def visual_3d_uncharged_reduce():
    _visual_3d_fixture(_fixture_3d_uncharged_reduce, "uncharged + dim reduce 3-D")

def visual_3d_charged_reduce():
    _visual_3d_fixture(_fixture_3d_charged_reduce, "charged + dim reduce 3-D")

def visual_3d_custom_reduce():
    _visual_3d_fixture(_fixture_3d_custom_reduce, "custom + dim reduce 3-D")


def run_visual():
    print("\n── Visual tests ──────────────────────────────────────────")
    visual_multi_sample_nonzero()
    visual_multi_sample_zero_then_retry()
    visual_3d_multi_sample()
    visual_2sample_charged()
    visual_2sample_custom()
    visual_2sample_uncharged_cut()
    visual_2sample_charged_cut()
    visual_2sample_custom_cut()
    visual_2sample_uncharged_reduce()
    visual_2sample_charged_reduce()
    visual_2sample_custom_reduce()
    visual_3d_charged()
    visual_3d_charge_ranges()
    visual_3d_custom()
    visual_3d_uncharged_cut()
    visual_3d_charged_cut()
    visual_3d_custom_cut()
    visual_3d_uncharged_reduce()
    visual_3d_charged_reduce()
    visual_3d_custom_reduce()
    print("── Visual tests complete ─────────────────────────────────\n")


# ── entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run_automated()
    if "--visual" in sys.argv:
        run_visual()
