import json
import zipfile
from pathlib import Path

import numpy as np

from model.phase_field import Phase_Field
from model.picip import PICIP, Sample
from model.visualise_cube import make_plotter
from preprocessors import parse_sample


class PicipModel:
    """PICIP handler: infer unknown phase composition from XRD data.

    Two Modes:
    - ``instance`` (deferred): inline value inputs -> assets + value outputs.
    - ``dataset`` (deferred): JSON file input -> assets (including results.zip).
    """

    # ── instance mode ────────────────────────────────────────────────────

    def _process_instance(self, values, output_dir, parameters, logger):
        default_elements = ["Fe", "Mn", "Ti"]
        default_composition = "Fe2Mn4Ti4"

        elements = values.get("elements") or default_elements
        if isinstance(elements, str):
            elements = [e.strip() for e in elements.split(",") if e.strip()] if "," in elements else [elements]

        composition = values.get("composition") or default_composition
        knowns = values.get("knowns") or ["FeMn"]
        mass_weights = values.get("mass_weights") or [1.0]

        predicted_error = parameters.get("predicted_error", 0.3)
        n_suggestions = int(parameters.get("n_suggestions", 5))

        sample_data = {
            "name": composition,
            "composition": composition,
            "predicted_error": predicted_error,
            "knowns": knowns,
            "mass_weights": mass_weights,
        }

        data = {
            "phase_field": {"type": "uncharged", "elements": elements},
            "samples": [sample_data],
            "n_suggestions": n_suggestions,
            "n_l": 50,
            "n_p": 50,
            "version": "b",
            "plot": {},
        }

        n_processed = self._run_from_data(data, output_dir, logger)
        return {"samples_processed": n_processed, "suggestions": n_suggestions}

    # ── dataset mode ─────────────────────────────────────────────────────

    def _process_dataset(self, files, output_dir, parameters, logger):
        if "file" not in files:
            raise ValueError("No input file provided")
        input_path, _ = files["file"]

        data = json.loads(Path(input_path).read_text())
        logger.info("loaded dataset", path=str(input_path))

        # dataset mode merges parameters into data (parameters are top-level
        # model config; the JSON file may also carry n_suggestions etc. — the
        # file's values take precedence, but parameters fill gaps)
        if "n_suggestions" not in data and "n_suggestions" in parameters:
            data["n_suggestions"] = parameters["n_suggestions"]

        n_processed = self._run_from_data(data, output_dir, logger)

        # zip all output files into results.zip
        zip_path = output_dir / "results.zip"
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for f in output_dir.iterdir():
                if f.is_file() and f.name != "results.zip":
                    zf.write(f, f.name)
        logger.info("wrote asset", path=str(zip_path))

        return {}

    # ── shared inference + asset writing ────────────────────────────────

    def _build_phase_field(self, cfg):
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

    def _run_from_data(self, data, output_dir, logger) -> int:
        """Run PICIP inference from a parsed data dict, write assets, return sample count."""
        pf = self._build_phase_field(data["phase_field"])
        logger.info("phase field", elements=pf.elements, constrained_dim=pf.constrained_dim)

        raw_samples = data.get("samples", [])
        if not raw_samples:
            raise ValueError("No samples provided in input")

        picip = PICIP(pf)
        for raw in raw_samples:
            parsed = parse_sample(raw)
            s = Sample(parsed["name"], parsed["composition"])
            s.add_knowns([str(k.reduced_formula) for k in parsed["knowns"]])
            s.add_mass_weights(parsed["mass_weights"])
            s.set_predicted_error(parsed["predicted_error"])
            picip.add_sample(s)
            logger.info("added sample", name=s.name)

        n_l = int(data.get("n_l", 50))
        n_p = int(data.get("n_p", 50))
        version = data.get("version", "b")
        pred = picip.run(n_l=n_l, n_p=n_p, version=version)
        logger.info("inference complete")

        n_suggestions = int(data.get("n_suggestions", 5))
        min_dist = data.get("min_dist")
        suggestions = picip.suggest(pred, n=n_suggestions, min_dist=min_dist)

        output_dir.mkdir(parents=True, exist_ok=True)

        # asset: suggestions.csv
        suggestions_path = output_dir / "suggestions.csv"
        suggestions.save(str(suggestions_path))
        logger.info("wrote asset", path=str(suggestions_path))

        # asset: prediction.html
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
        pl.show(show=False, save=str(output_dir / "prediction"))
        logger.info("wrote asset", path=str(output_dir / "prediction.html"))

        return len(picip.samples)

    # ── Protocol entry point ─────────────────────────────────────────────

    def process(self, mode, values, files, output_dir, parameters, logger):
        if mode == "instance":
            return self._process_instance(values, output_dir, parameters, logger)
        elif mode == "dataset":
            return self._process_dataset(files, output_dir, parameters, logger)
        else:
            raise ValueError(f"Unknown mode: {mode!r}")