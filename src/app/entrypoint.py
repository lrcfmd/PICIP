import argparse
import json
import logging
import os
import sys
from pathlib import Path
from urllib.parse import urlparse

import boto3
from botocore.config import Config
import numpy as np

from model.phase_field import Phase_Field
from model.picip import PICIP, Sample
from model.visualise_cube import make_plotter
from preprocessors import parse_sample
from infrastructure.s3 import S3Client, S3Config
from infrastructure.storage import StorageService
from infrastructure.logging import newLogger

DATA_DIR = Path(os.environ.get("DATA_DIR", "data"))
INPUT_DIR = DATA_DIR / "input"
OUTPUT_DIR = DATA_DIR / "output"
OUTPUT_ZIP = DATA_DIR / "output.zip"


def _build_phase_field(cfg):
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


def _resolve_local_file(uri: str) -> Path:
    """Convert file:// URI or plain path to local Path."""
    if uri.startswith("file://"):
        return Path(uri[7:])
    return Path(uri)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parameters", required=True, help="JSON exec config from API")
    parser.add_argument("--local", action="store_true", help="Run locally without S3")
    args = parser.parse_args()

    exec_config = json.loads(args.parameters)

    run_id = exec_config.get("run_id", "unknown")
    mode = exec_config.get("mode", "instance")
    inputs = exec_config.get("inputs", {})
    parameters = exec_config.get("parameters", {})
    output_path = exec_config.get("output", {}).get("path", "")
    local_mode = args.local

    # Unwrap API-style inputs
    inputs = {
        k: v["value"] if isinstance(v, dict) and "value" in v else v
        for k, v in inputs.items()
    }

    logger = newLogger("model")
    logger.info("Starting PICIP model processing")
    logger.info("Run ID: %s, Mode: %s", run_id, mode)
    logger.info("Inputs: %s", inputs)
    logger.info("Parameters: %s", parameters)

    if not local_mode:
        s3_config = S3Config.from_env()
        s3_client = S3Client(s3_config)
        storage = StorageService(s3_client, logger)
    else:
        s3_config = None
        storage = None

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    default_elements = ["Fe", "Mn", "Ti"]
    default_composition = "Fe2Mn4Ti4"

    raw_elements = inputs.get("elements")
    raw_composition = inputs.get("composition")
    raw_knowns = inputs.get("knowns")
    raw_mass_weights = inputs.get("mass_weights")

    def _or_default(value, default):
        if value is None or value == "" or value == []:
            return default
        return value

    def _or_defaultParam(value, default):
        if value is None or value == "":
            return default
        if isinstance(default, int) and isinstance(value, str):
            try:
                return int(value)
            except (ValueError, TypeError):
                return default
        if isinstance(default, float) and isinstance(value, str):
            try:
                return float(value)
            except (ValueError, TypeError):
                return default
        return value

    def _parse_elements(value):
        if value is None or value == "":
            return default_elements
        if isinstance(value, list):
            return value
        if isinstance(value, str):
            if "," in value:
                return [e.strip() for e in value.split(",") if e.strip()]
            return [value]
        return default_elements

    elements = _parse_elements(raw_elements)
    composition = _or_default(raw_composition, default_composition)
    knowns = _or_default(raw_knowns, None)
    mass_weights = _or_default(raw_mass_weights, None)

    if not knowns and not mass_weights:
        knowns = ["FeMn"]
        mass_weights = [1.0]
        logger.info("Using default knowns: %s, mass_weights: %s", knowns, mass_weights)

    predicted_error = _or_defaultParam(parameters.get("predicted_error"), 0.3)
    n_suggestions = _or_defaultParam(parameters.get("n_suggestions"), 5)

    sample_data = {
        "name": composition,
        "composition": composition,
        "predicted_error": predicted_error,
    }
    if knowns:
        sample_data["knowns"] = knowns
    if mass_weights:
        sample_data["mass_weights"] = mass_weights

    data = {
        "phase_field": {
            "type": "uncharged",
            "elements": elements,
        },
        "samples": [sample_data],
        "n_suggestions": n_suggestions,
        "n_l": 50,
        "n_p": 50,
        "version": "b",
        "plot": {},
    }

    logger.info("Created input data")

    pf = _build_phase_field(data["phase_field"])
    logger.info(
        "Phase field: %s (constrained_dim=%d)", pf.elements, pf.constrained_dim
    )

    raw_samples = data.get("samples", [])
    if not raw_samples:
        logger.error("No samples provided in input")
        sys.exit(1)

    picip = PICIP(pf)
    for raw in raw_samples:
        parsed = parse_sample(raw)
        s = Sample(parsed["name"], parsed["composition"])
        s.add_knowns([str(k.reduced_formula) for k in parsed["knowns"]])
        s.add_mass_weights(parsed["mass_weights"])
        s.set_predicted_error(parsed["predicted_error"])
        picip.add_sample(s)
        logger.info("Added sample '%s'", s.name)

    n_l = int(data.get("n_l", 50))
    n_p = int(data.get("n_p", 50))
    version = data.get("version", "b")
    pred = picip.run(n_l=n_l, n_p=n_p, version=version)
    logger.info("Inference complete")

    n_suggestions = int(data.get("n_suggestions", 5))
    min_dist = data.get("min_dist", None)
    suggestions = picip.suggest(pred, n=n_suggestions, min_dist=min_dist)

    suggestions_path = OUTPUT_DIR / "suggestions.csv"
    suggestions.save(str(suggestions_path))
    logger.info("Saved suggestions to %s", suggestions_path)

    plot_cfg = data.get("plot", {})
    pl = make_plotter(pf)
    common = dict(
        show_comps_hover=bool(plot_cfg.get("show_comps_hover", True)),
        comp_precision=int(plot_cfg.get("comp_precision", 2)),
    )
    if pf.constrained_dim == 3:
        common.update(
            p_mode=plot_cfg.get("p_mode", "nonzero"),
            top=float(plot_cfg.get("top", 0.9)),
            surface_opacity=float(plot_cfg.get("surface_opacity", 0.35)),
        )
    pl.plot_prediction_results(pred, **common)
    figure_path = OUTPUT_DIR / "prediction.html"
    pl.show(show=False, save=str(OUTPUT_DIR / "prediction"))
    logger.info("Figure saved to %s", figure_path)

    # Write result.json
    result = {
        "samples_processed": len(picip.samples),
        "suggestions": n_suggestions,
    }
    result_path = OUTPUT_DIR / "result.json"
    with result_path.open("w") as f:
        json.dump(result, f, indent=2)
    logger.info("Saved result to %s", result_path)

    if not local_mode:
        output_prefix = output_path.replace(f"s3://{s3_config.bucket}/", "").rstrip("/")
        if not output_prefix.endswith("/"):
            output_prefix += "/"

        logger.info(
            "Uploading output files to S3: bucket=%s, prefix=%s", s3_config.bucket, output_prefix
        )

        storage.upload_outputs(OUTPUT_DIR, output_prefix)
        logger.info("Upload complete")

    print(json.dumps(result))


if __name__ == "__main__":
    main()
