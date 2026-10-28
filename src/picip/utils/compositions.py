from typing import Iterable

import numpy as np
import numpy.typing as npt
from pymatgen.core import Composition


def atom_counts(compositions: Iterable[Composition]) -> npt.NDArray[np.float64]:
    raise NotImplementedError
