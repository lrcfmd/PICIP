from collections.abc import Collection
from typing import Iterable

import numpy as np
import numpy.typing as npt
from pymatgen.core import Composition


def atom_counts(compositions: Iterable[Composition]) -> npt.NDArray[np.float64]:
    raise NotImplementedError


def molar_masses(compositions: Collection[Composition]) -> npt.NDArray[np.float64]:
    return np.fromiter(
        (composition.weight for composition in compositions),
        dtype=np.float64,
        count=len(compositions)
    )


def composition_to_standard_basis(composition: Composition, elements: Iterable[str]) -> npt.NDArray[np.float64]:
    raise NotImplementedError


def compositions_to_standard_basis(composition: Collection[Composition], elements: Iterable[str]) \
        -> npt.NDArray[np.float64]:
    raise NotImplementedError
