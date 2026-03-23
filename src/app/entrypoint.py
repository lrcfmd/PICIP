import json
import logging
import sys
from pathlib import Path

from model.phase_field import Phase_Field
from model.picip import PICIP, Sample
from preprocessors import parse_sample

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

DATA_DIR    = Path("/data")
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
    n_l = int(data.get("n_l", 50))
    n_p = int(data.get("n_p", 50))
    pred = picip.run(n_l=n_l, n_p=n_p)
    logging.info("Inference complete")

    # Suggestions
    n_suggestions = int(data.get("n_suggestions", 5))
    min_dist      = data.get("min_dist", None)
    suggestions   = picip.suggest(pred, n=n_suggestions, min_dist=min_dist)

    # Write outputs
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    suggestions_path = OUTPUT_DIR / "suggestions.csv"
    suggestions.save(str(suggestions_path))

    result = {
        "samples_processed": len(picip.samples),
        "suggestions": suggestions_path.name,
    }
    print(json.dumps(result))
    logging.info("Done")


if __name__ == "__main__":
    main()
