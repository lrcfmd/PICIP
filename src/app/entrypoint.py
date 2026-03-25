import json
import logging
import os
import sys
from pathlib import Path

import numpy as np

from model.phase_field import Phase_Field
from model.picip import PICIP, Sample
from model.visualise_cube import make_plotter
from preprocessors import parse_sample

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

DATA_DIR    = Path(os.environ.get("DATA_DIR", "/data"))
INPUT_PATH  = DATA_DIR / "input.json"
OUTPUT_DIR  = DATA_DIR / "output"


def _build_phase_field(cfg):
    """Construct a Phase_Field from the phase_field config block."""
    pf = Phase_Field()
    kind = cfg.get("type", "uncharged")
    if kind == "uncharged":
        pf.setup_uncharged(cfg["elements"])
    elif kind == "charged":
        pf.setup_charged(cfg["elements"])
    elif kind == "charge_ranges":
        pf.setup_charge_ranges(cfg["elements"])
    elif kind == "custom":
        pf.setup_custom(cfg["species"], np.array(cfg["custom_con"]))
    else:
        raise ValueError(f"Unknown phase_field type: '{kind}'")
    return pf


def main():
    logging.info("Starting PICIP")

    if not INPUT_PATH.exists():
        logging.error("Input file not found: %s", INPUT_PATH)
        sys.exit(1)

    with INPUT_PATH.open() as f:
        data = json.load(f)

    # Build phase field
    pf = _build_phase_field(data["phase_field"])
    logging.info("Phase field: %s (constrained_dim=%d)", pf.elements, pf.constrained_dim)

    # Parse and validate samples
    raw_samples = data.get("samples", [])
    if not raw_samples:
        logging.error("No samples provided in input")
        sys.exit(1)

    picip = PICIP(pf)
    for raw in raw_samples:
        parsed = parse_sample(raw)
        s = Sample(parsed["name"], parsed["composition"])
        s.add_knowns([str(k.reduced_formula) for k in parsed["knowns"]])
        s.add_mass_weights(parsed["mass_weights"])
        s.set_predicted_error(parsed["predicted_error"])
        picip.add_sample(s)
        logging.info("Added sample '%s'", s.name)

    # Run inference
    n_l     = int(data.get("n_l", 50))
    n_p     = int(data.get("n_p", 50))
    version = data.get("version", "b")
    pred    = picip.run(n_l=n_l, n_p=n_p, version=version)
    logging.info("Inference complete")

    # Suggestions
    n_suggestions = int(data.get("n_suggestions", 5))
    min_dist      = data.get("min_dist", None)
    suggestions   = picip.suggest(pred, n=n_suggestions, min_dist=min_dist)

    # Write outputs
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    suggestions_path = OUTPUT_DIR / "suggestions.csv"
    suggestions.save(str(suggestions_path))

    # Generate HTML figure
    plot_cfg = data.get("plot", {})
    pl = make_plotter(pf)
    common = dict(
        show_comps_hover = bool(plot_cfg.get("show_comps_hover", True)),
        comp_precision   = int(plot_cfg.get("comp_precision", 2)),
    )
    if pf.constrained_dim == 3:
        common.update(
            p_mode          = plot_cfg.get("p_mode", "nonzero"),
            top             = float(plot_cfg.get("top", 0.9)),
            surface_opacity = float(plot_cfg.get("surface_opacity", 0.35)),
        )
    pl.plot_prediction_results(pred, **common)
    figure_path = OUTPUT_DIR / "prediction.html"
    pl.show(show=False, save=str(OUTPUT_DIR / "prediction"))
    logging.info("Figure saved to %s", figure_path)

    result = {
        "samples_processed": len(picip.samples),
        "suggestions": suggestions_path.name,
        "figure": figure_path.name,
    }
    print(json.dumps(result))
    logging.info("Done")


if __name__ == "__main__":
    main()
