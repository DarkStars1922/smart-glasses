from __future__ import annotations

import argparse
from pathlib import Path
import sys


if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis.degradation.fitting import fit_manifest
from analysis.degradation.reporting import write_results
from analysis.degradation.schema import load_manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fit the path-specific smart-glasses JPEG degradation model"
    )
    parser.add_argument(
        "--manifest", type=Path, default=Path("analysis/calibration_b_v1.json")
    )
    parser.add_argument("--output", type=Path, default=Path("analysis/results"))
    parser.add_argument("--seed", type=int, default=20260721)
    arguments = parser.parse_args()

    manifest = load_manifest(arguments.manifest)
    result = fit_manifest(manifest, seed=arguments.seed)
    write_results(result, manifest, arguments.output, seed=arguments.seed)
    print(f"wrote {arguments.output / 'v1_parameters.json'}")
    print(f"wrote {arguments.output / 'v1_report.md'}")
    for domain_name, domain in result["domains"].items():
        parameters = domain["effective_parameters"]
        print(
            f"{domain_name}: scale={parameters['scale_camera_per_source']} "
            f"blur_fwhm={parameters['blur_fwhm_camera_px']}"
        )


if __name__ == "__main__":
    main()
