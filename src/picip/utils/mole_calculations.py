import numpy as np
import numpy.typing as npt


def rel_n_formula_units_to_mass_fraction(
        rel_n_formula_units: npt.NDArray[np.float64],
        molar_masses: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """Converts relative numbers of formula units (e.g. normalised formula-unit fractions) to mass fractions."""
    return (rel_masses := rel_n_formula_units * molar_masses) / rel_masses.sum()


def rel_n_formula_units_to_phase_fraction(
        rel_n_formula_units: npt.NDArray[np.float64],
        atom_counts: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """Converts relative numbers of formula units (e.g. normalised formula-unit fractions) to phase fractions."""
    return (rel_atom_counts := rel_n_formula_units * atom_counts) / rel_atom_counts.sum()


def rel_mass_to_phase_fraction(
        rel_masses: npt.NDArray[np.float64],
        molar_masses: npt.NDArray[np.float64],
        atom_counts: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """Converts relative phase masses (e.g. normalised mass fractions) to phase fractions."""
    return (rel_atom_counts := rel_masses / molar_masses * atom_counts) / rel_atom_counts.sum()


def rel_total_phase_atom_count_to_mass_fraction(
        rel_total_phase_atom_counts: npt.NDArray[np.float64],
        molar_masses: npt.NDArray[np.float64],
        atom_counts: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """Converts relative total atom counts of each phase in whatever `phase_fractions` as a whole
     represents (e.g. normalised phase fractions) to mass fractions."""
    return (mass := rel_total_phase_atom_counts * molar_masses / atom_counts) / mass.sum()
