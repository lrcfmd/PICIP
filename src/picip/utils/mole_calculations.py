import numpy as np
import numpy.typing as npt


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


def mass_fraction_to_phase_fraction(
        mass_fractions: npt.NDArray[np.float64],
        molar_masses: npt.NDArray[np.float64],
        atom_counts: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    raise NotImplementedError


def phase_fraction_to_mass_fraction(
        phase_fractions: npt.NDArray[np.float64],
        molar_masses: npt.NDArray[np.float64],
        atom_counts: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    raise NotImplementedError
