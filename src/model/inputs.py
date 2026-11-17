"""Checks on the user's inputs, run before any inference.

Every problem the user can fix raises ``InputError`` with a plain-English
message; the runner reports it as the run's error without a traceback.
"""

import json
import math
from pathlib import Path

import numpy as np
from pymatgen.core import Composition, Element

from infrastructure.handler import InputError

PHASE_FIELD_TYPES = ("uncharged", "charged", "charge_ranges", "custom")
P_MODES = ("nonzero", "all", "top", "surface")
VERSIONS = ("b", "gaussian")
MAX_KNOWNS = 3  # PICIP implements 1, 2 or 3 known phases per sample


def _is_number(value) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def _whole_number(value, name: str, minimum: int) -> int:
    if _is_number(value) and float(value).is_integer() and value >= minimum:
        return int(value)
    raise InputError(f"{name} must be a whole number of at least {minimum}, got {value!r}.")


def _check_symbols(symbols, where: str) -> list[str]:
    if not isinstance(symbols, (list, tuple)) or not symbols:
        raise InputError(f"{where}: give the elements as a list of symbols, e.g. [\"Fe\", \"Mn\", \"Ti\"].")
    out = []
    for s in symbols:
        sym = str(s).strip()
        if not Element.is_valid_symbol(sym):
            hint = sym[:1].upper() + sym[1:].lower()
            extra = f" - did you mean '{hint}'?" if Element.is_valid_symbol(hint) else "."
            raise InputError(f"{where}: '{sym}' is not an element symbol{extra}")
        if sym in out:
            raise InputError(f"{where}: element '{sym}' is listed twice.")
        out.append(sym)
    return out


def _check_charge(value, element: str, allow_range: bool) -> None:
    if _is_number(value):
        return
    if allow_range and isinstance(value, list):
        if len(value) == 2 and all(_is_number(v) for v in value) and value[0] <= value[1]:
            return
        raise InputError(
            f"Charge range for {element} must be [lowest, highest], e.g. [2, 3], got {value!r}."
        )
    kind = "a number or a [lowest, highest] range" if allow_range else "a number"
    raise InputError(f"Charge for {element} must be {kind}, got {value!r}.")


def _charge_bounds(species: dict) -> tuple[list[float], list[float]]:
    lows = [v[0] if isinstance(v, list) else v for v in species.values()]
    highs = [v[1] if isinstance(v, list) else v for v in species.values()]
    return lows, highs


def check_phase_field(cfg) -> tuple[dict, list[str]]:
    """Check a phase_field block; return it (normalised) and its element list."""
    if not isinstance(cfg, dict):
        raise InputError("phase_field must be an object with a \"type\" and its \"elements\".")
    kind = cfg.get("type", "uncharged")
    if kind not in PHASE_FIELD_TYPES:
        raise InputError(
            f"Unknown phase_field type '{kind}'. Use one of: {', '.join(PHASE_FIELD_TYPES)}."
        )
    cfg = dict(cfg, type=kind)
    key = "species" if kind == "custom" else "elements"
    if key not in cfg:
        raise InputError(f"A '{kind}' phase_field needs \"{key}\".")
    spec = cfg[key]
    n_constraints = 1  # element fractions sum to one

    if kind == "uncharged" or (kind == "custom" and isinstance(spec, list)):
        elements = _check_symbols(spec, f"phase_field {key}")
        cfg[key] = elements
    elif isinstance(spec, dict) and spec:
        elements = _check_symbols(list(spec), f"phase_field {key}")
        allow_range = kind in ("charge_ranges", "custom")
        for el, value in spec.items():
            _check_charge(value, el, allow_range)
        lows, highs = _charge_bounds(spec)
        fixed = not any(isinstance(v, list) for v in spec.values())
        balances = min(lows) < 0 < max(highs) if fixed else min(lows) <= 0 <= max(highs)
        if not balances:
            raise InputError(
                "The charges in the phase_field cannot balance: give at least one element a "
                "positive charge and one a negative charge, e.g. {\"Cu\": 2, \"S\": -2}."
            )
        if kind == "charge_ranges" and fixed:
            raise InputError(
                "A 'charge_ranges' phase_field needs at least one element with a range, e.g. "
                "\"Fe\": [2, 3]. If every charge is fixed, use type 'charged'."
            )
        if fixed:
            n_constraints += 1  # charge neutrality
    else:
        example = '{"Cu": 2, "S": -2}' if kind == "charged" else '{"Fe": [2, 3], "O": -2}'
        raise InputError(
            f"A '{kind}' phase_field needs \"{key}\" as an object of element: charge, e.g. {example}."
        )

    if kind == "custom":
        if "custom_con" not in cfg:
            raise InputError("A 'custom' phase_field needs \"custom_con\": the extra constraint rows.")
        try:
            con = np.array(cfg["custom_con"], dtype=float)
        except (TypeError, ValueError):
            con = None
        if con is None or con.ndim not in (1, 2) or con.shape[-1] != len(elements) or con.size == 0:
            raise InputError(
                f"custom_con must be a list of numbers with one entry per element "
                f"({len(elements)}), or a list of such rows."
            )
        n_constraints += 1 if con.ndim == 1 else con.shape[0]

    dim = len(elements) - n_constraints
    if dim not in (2, 3):
        raise InputError(
            f"The phase field {'-'.join(elements)} ({kind}) has {dim} free dimension(s); PICIP "
            "works with 2 or 3. For an uncharged field give 3 or 4 elements; a charge or "
            "custom constraint removes one dimension each."
        )
    return cfg, elements


def _composition(text, where: str) -> Composition:
    if not isinstance(text, str) or not text.strip():
        raise InputError(f"{where}: give a formula, e.g. \"Fe2Mn4Ti4\".")
    try:
        comp = Composition(text.strip())
    except Exception:
        comp = None
    if comp is None or comp.num_atoms <= 0:
        raise InputError(f"{where}: could not read the formula '{text}'. Use e.g. \"Fe2Mn4Ti4\".")
    unknown = [el.symbol for el in comp.elements if not Element.is_valid_symbol(el.symbol)]
    if unknown:
        raise InputError(
            f"{where}: could not read the formula '{text}': '{unknown[0]}' is not an element symbol."
        )
    return comp


def check_sample(raw, elements: list[str], label: str) -> dict:
    """Check one sample and return it in the shape preprocessors.parse_sample expects."""
    if not isinstance(raw, dict):
        raise InputError(f"{label} must be an object with composition, knowns and mass_weights.")
    name = str(raw.get("name") or raw.get("composition") or label)
    where = f"{label} ({name})" if name != label else label
    comp = _composition(raw.get("composition"), f"{where} composition")
    _within_field(comp, elements, f"{where} composition '{raw['composition']}'")

    knowns = raw.get("knowns") or []
    if not isinstance(knowns, list) or not knowns:
        raise InputError(f"{where}: give at least one known phase (seen in the XRD pattern).")
    if len(knowns) > MAX_KNOWNS:
        raise InputError(
            f"{where}: {len(knowns)} known phases given; PICIP handles 1 to {MAX_KNOWNS} per sample."
        )
    known_comps = []
    for k in knowns:
        kc = _composition(k, f"{where} known phase")
        _within_field(kc, elements, f"{where} known phase '{k}'")
        if _same(kc, comp, elements):
            raise InputError(
                f"{where}: the sample composition is the same as known phase '{k}', so there "
                "is no unknown phase to infer. Check the composition and the known phases."
            )
        known_comps.append(kc)

    weights = raw.get("mass_weights")
    if weights in (None, []) and len(knowns) == 1:
        weights = [1.0]
    if not isinstance(weights, list) or len(weights) != len(knowns):
        got = len(weights) if isinstance(weights, list) else 0
        raise InputError(
            f"{where}: give one mass weight per known phase ({len(knowns)} known, {got} weights), "
            "in the same order."
        )
    for w, k in zip(weights, knowns):
        if not _is_number(w) or w <= 0:
            raise InputError(
                f"{where}: the mass weight of '{k}' must be a number greater than 0, got {w!r}. "
                "Leave out known phases that are not present."
            )

    error = raw.get("predicted_error", 0.3)
    if not _is_number(error) or error <= 0:
        raise InputError(f"{where}: predicted_error must be a number greater than 0, got {error!r}.")

    return {
        "name": name,
        "composition": raw["composition"],
        "knowns": list(knowns),
        "mass_weights": [float(w) for w in weights],
        "predicted_error": float(error),
    }


def _within_field(comp: Composition, elements: list[str], what: str) -> None:
    extra = sorted({el.symbol for el in comp.elements} - set(elements))
    if extra:
        raise InputError(
            f"{what} contains {', '.join(extra)}, which is not in the phase field "
            f"({', '.join(elements)})."
        )


def _same(a: Composition, b: Composition, elements: list[str]) -> bool:
    return all(
        abs(a.get_atomic_fraction(el) - b.get_atomic_fraction(el)) < 1e-6 for el in elements
    )


def check_run_options(data: dict) -> None:
    """Check the optional run settings of a dataset file (in place)."""
    for key, minimum in (("n_l", 1), ("n_p", 1), ("n_suggestions", 1)):
        if key in data:
            data[key] = _whole_number(data[key], key, minimum)
    if data.get("min_dist") is not None:
        if not _is_number(data["min_dist"]) or data["min_dist"] < 0:
            raise InputError(f"min_dist must be a number of at least 0, got {data['min_dist']!r}.")
    if data.get("version", "b") not in VERSIONS:
        raise InputError(f"version must be \"b\" or \"gaussian\", got {data['version']!r}.")
    plot = data.get("plot", {})
    if plot is None:
        data["plot"] = plot = {}
    if not isinstance(plot, dict):
        raise InputError("plot must be an object of plot settings, e.g. {\"p_mode\": \"surface\"}.")
    if "p_mode" in plot and plot["p_mode"] not in P_MODES:
        raise InputError(f"plot p_mode must be one of {', '.join(P_MODES)}, got {plot['p_mode']!r}.")
    if "top" in plot and not (_is_number(plot["top"]) and 0 < plot["top"] <= 1):
        raise InputError(f"plot top must be a fraction between 0 and 1, got {plot['top']!r}.")
    if "surface_opacity" in plot and not (
        _is_number(plot["surface_opacity"]) and 0 <= plot["surface_opacity"] <= 1
    ):
        raise InputError(
            f"plot surface_opacity must be between 0 and 1, got {plot['surface_opacity']!r}."
        )
    if "comp_precision" in plot:
        plot["comp_precision"] = _whole_number(plot["comp_precision"], "plot comp_precision", 0)


def check_dataset(data) -> dict:
    """Check a whole dataset file's contents; return it normalised."""
    if not isinstance(data, dict):
        raise InputError("The input file must hold one JSON object with \"phase_field\" and \"samples\".")
    if "phase_field" not in data:
        raise InputError("The input file has no \"phase_field\" (the elements and their type).")
    data = dict(data)
    data["phase_field"], elements = check_phase_field(data["phase_field"])
    samples = data.get("samples")
    if not isinstance(samples, list) or not samples:
        raise InputError("The input file has no \"samples\": give a list of at least one sample.")
    data["samples"] = [
        check_sample(raw, elements, f"Sample {i}") for i, raw in enumerate(samples, start=1)
    ]
    check_run_options(data)
    return data


def read_dataset_file(path: Path) -> dict:
    try:
        text = Path(path).read_text(encoding="utf-8-sig")
    except UnicodeDecodeError:
        raise InputError("The input file is not a text JSON file.") from None
    if not text.strip():
        raise InputError("The input file is empty.")
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        raise InputError(
            f"The input file is not valid JSON (line {e.lineno}, column {e.colno}: {e.msg})."
        ) from None
