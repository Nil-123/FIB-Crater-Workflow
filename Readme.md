# FIB Crater Workflow

Python workflow for automated design, fabrication parameter generation, and AFM-based analysis of Gaussian-shaped focused ion beam (FIB) milled craters.

The project was developed during research on optical microcavities and provides a framework for:

- Streamfile generation
- Gaussian crater design
- AFM crater fitting
- Radius-of-curvature extraction
- Process correction factor estimation
- Closed-loop fabrication optimization

---

## Workflow

```text
Target crater parameters
        ↓
Generate streamfile
        ↓
FIB fabrication
        ↓
AFM measurement
        ↓
Crater fitting
        ↓
Correction factor estimation
        ↓
Generate improved streamfile
```

---

## Features

### Streamfile Generation

Generate FIB streamfiles from desired crater parameters:

- Crater depth
- Radius of curvature
- Beam current
- Pixel ordering strategy
- Height and sigma correction factors
- Each crater is given a unique identifier that is milled along the crater.
- Updates the streamfile database for every streamfile generation run.

### AFM Analysis

Extract crater properties from AFM measurements:

- Depth
- Sigma X
- Sigma Y
- Radius of curvature
- Fit quality metrics

### Closed-Loop Optimization

Automatically compares:

- Target crater geometry
- Measured crater geometry

and suggests updated fabrication correction factors.

### Workflow Summary

Automatically exports:

```text
last_workflow_summary.json
```

containing:

- Design parameters
- AFM fitting results
- Suggested correction factors

---


## Example Usage

```python
design = CraterDesign(
    size_um=15,
    roc_um=18,
    depth_um=0.3,
    current=26e-12,
    correction_factor_height=1,
    correction_factor_sigma=1,
    pixel_order=1,
    show_plot=True,
    comment="Testing"
)
```

Run workflow:

```bash
python Example-workflow.py
```

---

## Future Development

Planned features:

- Automatic workflow database
- Experiment tracking
- Multi-current calibration datasets
- Predictive correction models
- Machine-learning assisted optimization
- Automatic report generation

---

## Author

Nilesh Dalla

PhD Researcher

University of Warsaw

Research areas:

- FIB fabrication
- Optical microcavities
- AFM metrology
