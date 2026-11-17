"""Input checks of the LMDS handler: bad input fails early with an InputError."""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from infrastructure.handler import InputError  # noqa: E402
from infrastructure.logging import newLogger  # noqa: E402
from model.inputs import check_dataset, check_phase_field, read_dataset_file  # noqa: E402
from model.picip_model import PicipModel  # noqa: E402

LOG = newLogger("test")
GOOD_SAMPLE = {"name": "s1", "composition": "Fe2Mn4Ti4", "knowns": ["FeMn", "FeTi"],
               "mass_weights": [0.5, 0.5]}


def _instance(tmp_path, **values):
    base = {"elements": ["Fe", "Mn", "Ti"], "composition": "Fe2Mn4Ti4",
            "knowns": ["FeMn"], "mass_weights": [1.0]}
    return PicipModel().process("instance", {**base, **values}, {}, tmp_path,
                                {"predicted_error": 0.3, "n_suggestions": 3}, LOG)


@pytest.mark.parametrize("values, match", [
    ({"elements": ["Fe", "mn", "Ti"]}, "'mn' is not an element symbol - did you mean 'Mn'"),
    ({"elements": ["Fe", "Fe", "Ti"]}, "listed twice"),
    ({"elements": ["Fe", "Mn"]}, "1 free dimension"),
    ({"elements": ["Fe", "Mn", "Ti", "Cu", "Ni"]}, "4 free dimension"),
    ({"composition": "Fe2Mn4Xx"}, "'Xx' is not an element symbol"),
    ({"composition": "fe2mn4ti4"}, "could not read the formula"),
    ({"composition": "Fe2Mn4Cu"}, "contains Cu, which is not in the phase field"),
    ({"knowns": ["FeCu"]}, "known phase 'FeCu' contains Cu"),
    ({"knowns": []}, "at least one known phase"),
    ({"knowns": ["FeMn", "FeTi", "MnTi", "Fe2Ti"], "mass_weights": [1, 1, 1, 1]}, "1 to 3"),
    ({"knowns": ["FeMn", "FeTi"], "mass_weights": [1.0]}, "one mass weight per known phase"),
    ({"knowns": ["FeMn", "FeTi"], "mass_weights": [1.0, 0.0]}, "greater than 0"),
    ({"composition": "FeMn"}, "same as known phase 'FeMn'"),
])
def test_instance_rejects(tmp_path, values, match):
    with pytest.raises(InputError, match=match):
        _instance(tmp_path, **values)


def test_instance_bad_parameters(tmp_path):
    model = PicipModel()
    values = {"elements": ["Fe", "Mn", "Ti"], "composition": "Fe2Mn4Ti4", "knowns": ["FeMn"]}
    with pytest.raises(InputError, match="predicted_error must be a number greater than 0"):
        model.process("instance", values, {}, tmp_path, {"predicted_error": 0}, LOG)
    with pytest.raises(InputError, match="n_suggestions must be a whole number of at least 1"):
        model.process("instance", values, {}, tmp_path, {"n_suggestions": 0}, LOG)


def test_single_known_needs_no_weight(tmp_path):
    out = _instance(tmp_path, mass_weights=[])
    assert out["samples_processed"] == 1
    assert (tmp_path / "suggestions.csv").exists()


@pytest.mark.parametrize("phase_field, match", [
    ({"type": "magic", "elements": ["Fe", "Mn", "Ti"]}, "Unknown phase_field type"),
    ({"type": "charged", "elements": ["Cu", "Zn", "S"]}, "object of element: charge"),
    ({"type": "charged", "elements": {"Cu": 2, "Zn": 2, "Fe": 3, "Mn": 2}}, "cannot balance"),
    ({"type": "charged", "elements": {"Cu": 2, "S": -2}}, "0 free dimension"),
    ({"type": "charged", "elements": {"Cu": "two", "Zn": 2, "S": -2}}, "Charge for Cu must be a number"),
    ({"type": "charge_ranges", "elements": {"Fe": [3, 2], "Mn": 2, "O": -2}}, r"\[lowest, highest\]"),
    ({"type": "charge_ranges", "elements": {"Fe": 3, "Mn": 2, "O": -2}}, "use type 'charged'"),
    ({"type": "custom", "species": ["Fe", "Mn", "Ti", "Cu"]}, "custom_con"),
    ({"type": "custom", "species": ["Fe", "Mn", "Ti", "Cu"], "custom_con": [1, -1]}, "one entry per element"),
])
def test_phase_field_rejects(phase_field, match):
    with pytest.raises(InputError, match=match):
        check_phase_field(phase_field)


def test_phase_field_accepts_examples():
    for path in sorted((ROOT / "examples").glob("*.json")):
        data = json.loads(path.read_text())
        if path.name == "charged.json":
            # Cu-Zn-S with fixed charges is 1-D, which PICIP cannot handle
            # (Phase_Field raises too); the check now says so up front.
            with pytest.raises(InputError, match="1 free dimension"):
                check_dataset(data)
            continue
        check_dataset(data)  # every other bundled example passes the checks


@pytest.mark.parametrize("data, match", [
    ([], "one JSON object"),
    ({"samples": [GOOD_SAMPLE]}, "no \"phase_field\""),
    ({"phase_field": {"elements": ["Fe", "Mn", "Ti"]}}, "no \"samples\""),
    ({"phase_field": {"elements": ["Fe", "Mn", "Ti"]}, "samples": [GOOD_SAMPLE, {"knowns": ["FeMn"]}]},
     "Sample 2 composition: give a formula"),
    ({"phase_field": {"elements": ["Fe", "Mn", "Ti"]}, "samples": [GOOD_SAMPLE], "n_l": 0}, "n_l must be"),
    ({"phase_field": {"elements": ["Fe", "Mn", "Ti"]}, "samples": [GOOD_SAMPLE], "version": "c"}, "version must be"),
    ({"phase_field": {"elements": ["Fe", "Mn", "Ti"]}, "samples": [GOOD_SAMPLE], "plot": {"p_mode": "x"}}, "p_mode"),
    ({"phase_field": {"elements": ["Fe", "Mn", "Ti"]}, "samples": [GOOD_SAMPLE], "min_dist": -1}, "min_dist"),
])
def test_dataset_rejects(data, match):
    with pytest.raises(InputError, match=match):
        check_dataset(data)


def test_dataset_file_errors(tmp_path):
    model = PicipModel()
    bad = tmp_path / "bad.json"
    bad.write_text('{"phase_field": {"elements": ["Fe", "Mn", "Ti"]},\n "samples": [}')
    with pytest.raises(InputError, match="not valid JSON \\(line 2"):
        model.process("dataset", {}, {"file": (bad, "application/json")}, tmp_path, {}, LOG)
    empty = tmp_path / "empty.json"
    empty.write_text("")
    with pytest.raises(InputError, match="empty"):
        read_dataset_file(empty)
    with pytest.raises(InputError, match="No input file"):
        model.process("dataset", {}, {}, tmp_path, {}, LOG)


def test_schema_example_runs(tmp_path):
    """The example file shown in the LMDS form is a valid, runnable dataset."""
    schema = json.loads((ROOT / "schema" / "model.json").read_text())
    example = schema["execution"]["modes"]["dataset"]["inputs"]["file"]["example"]
    path = tmp_path / "example.json"
    path.write_text(example)
    out_dir = tmp_path / "out"
    PicipModel().process("dataset", {}, {"file": (path, "application/json")}, out_dir, {}, LOG)
    assert (out_dir / "suggestions.csv").exists()
    assert (out_dir / "results.zip").exists()
