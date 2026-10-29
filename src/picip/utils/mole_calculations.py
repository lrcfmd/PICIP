import numpy as np
import numpy.typing as npt


def formula_unit_fraction_to_mass_fraction(
        phase_fractions: npt.NDArray[np.float64],
        molar_masses: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    return (rel_masses := phase_fractions * molar_masses) / rel_masses.sum()


def formula_unit_fraction_to_phase_fraction(
        formula_unit_fractions: npt.NDArray[np.float64],
        atom_counts: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    return (rel_atom_counts := formula_unit_fractions * atom_counts) / rel_atom_counts.sum()


def mass_fraction_to_phase_fraction(
        mass_fractions: npt.NDArray[np.float64],
        molar_masses: npt.NDArray[np.float64],
        atom_counts: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    return (rel_atom_counts := mass_fractions / molar_masses * atom_counts) / rel_atom_counts.sum()


def phase_fraction_to_mass_fraction(
        phase_fractions: npt.NDArray[np.float64],
        molar_masses: npt.NDArray[np.float64],
        atom_counts: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    return (mass := phase_fractions * molar_masses / atom_counts) / mass.sum()
