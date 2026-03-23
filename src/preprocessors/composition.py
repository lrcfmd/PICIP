from pymatgen.core import Composition


def parse_sample(raw):
    """
    Validate and normalise a raw sample dict from the input JSON.

    Parameters
    ----------
    raw : dict
        Keys: name (str), composition (str), knowns (list of str),
        mass_weights (list of float), predicted_error (float, optional).

    Returns
    -------
    dict
        Same structure with composition and knowns parsed as
        pymatgen Composition objects and weights normalised to sum to 1.

    Raises
    ------
    ValueError
        If required keys are missing or compositions cannot be parsed.
    """
    required = {"name", "composition", "knowns", "mass_weights"}
    missing = required - set(raw)
    if missing:
        raise ValueError(f"Sample missing required fields: {missing}")

    try:
        composition = Composition(raw["composition"])
    except Exception as e:
        raise ValueError(f"Could not parse composition '{raw['composition']}': {e}")

    knowns = []
    for k in raw["knowns"]:
        try:
            knowns.append(Composition(k))
        except Exception as e:
            raise ValueError(f"Could not parse known phase '{k}': {e}")

    weights = list(raw["mass_weights"])
    if len(weights) != len(knowns):
        raise ValueError(
            f"mass_weights length ({len(weights)}) does not match "
            f"knowns length ({len(knowns)})"
        )
    total = sum(weights)
    if total <= 0:
        raise ValueError("mass_weights must sum to a positive number")

    return {
        "name": raw["name"],
        "composition": composition,
        "knowns": knowns,
        "mass_weights": [w / total for w in weights],
        "predicted_error": float(raw.get("predicted_error", 0.3)),
    }
