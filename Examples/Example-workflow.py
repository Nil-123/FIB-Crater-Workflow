# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 18:17:35 2026
"Improved with LLMs"
@author: Nilesh Dalla
"""

# For easy run through IDE like Spyder
import sys
import json
from pathlib import Path
from dataclasses import dataclass, asdict
import numpy as np

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

# Importing functionalities
from FIB_generator.Generator import generate_streamfile
from AFM_analysis.Fit_craters import analyze_crater


# =========================
# Dataclasses
# =========================

@dataclass
class CraterDesign:
    size_um: float
    roc_um: float
    depth_um: float
    current: float
    correction_factor_height: float
    correction_factor_sigma: float
    pixel_order: int          # 0 = serpentine, 1 = random
    show_plot: bool
    comment: str

    @property
    def sigma_um(self) -> float:
        """Target Gaussian sigma derived from depth and radius of curvature."""
        return np.sqrt(self.depth_um * self.roc_um)

    def validate(self) -> None:
        if self.size_um <= 0:
            raise ValueError("size_um must be > 0")
        if self.roc_um <= 0:
            raise ValueError("roc_um must be > 0")
        if self.depth_um <= 0:
            raise ValueError("depth_um must be > 0")
        if self.current <= 0:
            raise ValueError("current must be > 0")
        if self.pixel_order not in (0, 1):
            raise ValueError("pixel_order must be 0 (serpentine) or 1 (random)")
        if self.correction_factor_height <= 0:
            raise ValueError("correction_factor_height must be > 0")
        if self.correction_factor_sigma <= 0:
            raise ValueError("correction_factor_sigma must be > 0")


@dataclass
class AFMFitConfig:
    region_radius: float
    save_data: bool
    show_plot: bool
    save_data_directory: Path

    def validate(self) -> None:
        if self.region_radius <= 0:
            raise ValueError("region_radius must be > 0")


@dataclass
class WorkflowSummary:
    target_depth_um: float
    measured_depth_um: float
    old_height_correction_factor: float
    suggested_height_correction_factor: float
    target_sigma_um: float
    measured_sigma_um: float
    old_sigma_correction_factor: float
    suggested_sigma_correction_factor: float
    streamfile_result: dict | str | None
    afm_result: dict

    def to_dict(self) -> dict:
        data = asdict(self)
        return data


# =========================
# Helper functions
# =========================

def get_project_paths() -> dict[str, Path]:
    root = Path(__file__).resolve().parent.parent
    return {
        "project_root": root,
        "database_dir": root / "Databases",
        "example_dir": root / "Examples",
        "results_dir": root / "Results",
    }


def ensure_directories(paths: dict[str, Path]) -> None:
    paths["database_dir"].mkdir(parents=True, exist_ok=True)
    paths["results_dir"].mkdir(parents=True, exist_ok=True)


def run_streamfile_generation(design: CraterDesign, output_folder: Path):
    """
    Run the FIB streamfile generator.
    """
    return generate_streamfile(
        Output_folder=output_folder,
        **asdict(design)
    )


def run_afm_analysis(filepath: Path, afm_config: AFMFitConfig) -> dict:
    """
    Run AFM crater analysis.
    """
    if not filepath.exists():
        raise FileNotFoundError(f"AFM input file not found: {filepath}")

    return analyze_crater(
        filepath=filepath,
        **asdict(afm_config)
    )


def build_workflow_summary(
    design: CraterDesign,
    streamfile_result,
    afm_result: dict
    ) -> WorkflowSummary:
    """
    Compare target crater design with measured AFM crater fit
    and suggest updated correction factors.
    """
    desired_depth = design.depth_um
    obtained_depth = abs(afm_result["A"])
    suggested_height_correction_factor = abs(desired_depth / obtained_depth)

    desired_sigma = design.sigma_um
    measured_sigma = (afm_result["sigma_x"] + afm_result["sigma_y"]) / 2
    suggested_sigma_correction_factor = abs(desired_sigma / measured_sigma)

    return WorkflowSummary(
        target_depth_um=desired_depth,
        measured_depth_um=obtained_depth,
        old_height_correction_factor=design.correction_factor_height,
        suggested_height_correction_factor=suggested_height_correction_factor,
        target_sigma_um=desired_sigma,
        measured_sigma_um=measured_sigma,
        old_sigma_correction_factor=design.correction_factor_sigma,
        suggested_sigma_correction_factor=suggested_sigma_correction_factor,
        streamfile_result=streamfile_result,
        afm_result=afm_result,
    )


def print_summary(summary: WorkflowSummary) -> None:
    print("\n===== FIB WORKFLOW SUMMARY =====")
    print(f" Target depth                  : {summary.target_depth_um:.3f} µm")
    print(f" Measured depth                : {summary.measured_depth_um:.3f} µm")
    print(f" Old height correction factor  : {summary.old_height_correction_factor:.3f}")
    print(f" Suggested height factor        : {summary.suggested_height_correction_factor:.3f}")
    print(f" Target sigma                  : {summary.target_sigma_um:.3f} µm")
    print(f" Measured sigma                : {summary.measured_sigma_um:.3f} µm")
    print(f" Old sigma correction factor   : {summary.old_sigma_correction_factor:.3f}")
    print(f" Suggested sigma factor         : {summary.suggested_sigma_correction_factor:.3f}")


def save_summary(summary: WorkflowSummary, output_path: Path) -> None:
    """
    Save workflow summary as JSON.
    """
    serializable = summary.to_dict()

    # Convert Path or unsupported objects if needed
    def convert(obj):
        if isinstance(obj, Path):
            return str(obj)
        if isinstance(obj, np.generic):
            return obj.item()
        return obj

    cleaned = json.loads(json.dumps(serializable, default=convert))

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(cleaned, f, indent=4)


# =========================
# Main workflow
# =========================

def main():
    paths = get_project_paths()
    ensure_directories(paths)

    # -------------------------
    # Streamfile generation parameters
    # -------------------------
    design = CraterDesign(
        size_um=15,                    # µm size of milling square
        roc_um=18,                     # µm target ROC
        depth_um=0.3,                  # µm target crater depth
        current=26e-12,                # beam current
        correction_factor_height=1.0,  # from previous AFM calibration
        correction_factor_sigma=1.0,   # from previous AFM calibration
        pixel_order=1,                 # 0 = serpentine, 1 = random
        show_plot=True,                # Comments for streamfile database
        comment="Testing"
    )
    design.validate()

    streamfile_result = run_streamfile_generation(
        design=design,
        output_folder=paths["results_dir"]
    )

    # -------------------------
    # AFM fitting parameters  --This assumes the AFM data is properly levelled.
    # -------------------------
    afm_config = AFMFitConfig(
        region_radius=5,
        save_data=True,
        show_plot=True,
        save_data_directory=paths["results_dir"]
    )
    afm_config.validate()

    afm_file = paths["example_dir"] / "AFM_test_data.txt"

    afm_result = run_afm_analysis(
        filepath=afm_file,
        afm_config=afm_config
    )

    # -------------------------
    # Build summary
    # -------------------------
    summary = build_workflow_summary(
        design=design,
        streamfile_result=streamfile_result,
        afm_result=afm_result
    )

    print_summary(summary)

    # -------------------------
    # Save summary
    # -------------------------
    summary_file = paths["results_dir"] / "last_workflow_summary.json"
    save_summary(summary, summary_file)

    print(f"\n💾 Workflow summary saved to: {summary_file}")


if __name__ == "__main__":
    main()