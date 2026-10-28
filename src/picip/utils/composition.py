from collections.abc import Collection
from typing import Iterable
from picip._typing import KnownsLike

import numpy as np
import numpy.typing as npt
from pymatgen.core import Composition


def atom_count(compositions: Iterable[Composition]) -> npt.NDArray[np.float64]:
    raise NotImplementedError


def molar_mass(compositions: Collection[Composition]) -> npt.NDArray[np.float64]:
    return np.fromiter(
        (composition.weight for composition in compositions),
        dtype=np.float64,
        count=len(compositions)
    )


def composition_to_standard_basis(composition: KnownsLike, elements: Collection[str]) -> npt.NDArray[np.float64]:
    """
    Converts one or more compositions to their standard basis representations
    given a `Collection` of element symbols.

    Parameters
    ----------
    composition : KnownsLike
        The composition(s) to convert.
    elements : Collection[str]
        The chemical elements whose concentrations in each phase are to be found.

    Returns
    -------
    ndarray, shape (n_compositions, n_elements) or (n_elements,)
        (`ndarray` of) the compositions, each converted to standard basis
    """

    raise NotImplementedError


def formula_unit_fraction_to_mass_fraction(
        phase_fractions: npt.NDArray[np.float64],
        molar_masses: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    raise NotImplementedError


def formula_unit_fraction_to_phase_fraction(
        formula_unit_fractions: npt.NDArray[np.float64],
        atom_counts: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    raise NotImplementedError
