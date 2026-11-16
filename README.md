# Data and Code for *When Do Online Calibration Gains Carry through to Deadline Outcomes in Heterogeneous GPU Pools?*

This repository contains the experimental records, reference implementations, analysis
scripts, and derived data for the manuscript above (Pang, Guo, Hao, Guo, Huang, and Lü,
North Automatic Control Technology Institute). It is the public location of the two
archives supplied with the submission, unpacked so that every file can be read and
downloaded individually.

Earlier versions of this work were circulated under the titles *From Prediction Error to
Deadline Outcomes: Online Execution-Time Calibration for Deadline-Aware Scheduling in
Heterogeneous GPU Pools* and *Closed-Loop Online Performance Calibration for Deadline-Aware
Scheduling in Heterogeneous GPU Pools*. The materials are the same delivery (20261109);
the internal package READMEs still carry the earlier title and describe the archives as
review-only attachments, which predates this public release.

## Contents

| Path | Size | Files | What it holds |
|---|---|---|---|
| `reproducibility/` | 158 MB | 1459 | Main package: three-host deployment records, main replay and point-estimate results, four CPU experiment sets, supplemental analyses, figure plotting data, H-event extracts, checksum manifests |
| `tf32_diagnostic/` | 2.8 MB | 247 | TF32 software-path diagnosis: per-round timings, configurations, environments, statistical summaries, figure |
| `docs/` | 0.4 MB | 2 | Reproducibility guide (PDF and Markdown): figure/table-to-file map, entry points, scope of what can and cannot be recomputed |
| `CITATION.cff` | — | — | Machine-readable citation metadata |

The two packages are deliberately separate: the main package predates the TF32 diagnosis
and does not contain it.

Each package keeps its own README, license notices, provenance tables, and SHA-256
manifests. Start with [`docs/Reproducibility_Guide.md`](docs/Reproducibility_Guide.md);
it is the authoritative map from manuscript figures and tables to the files that support
them.

## Quick start

Requirements: Python 3.10 or newer (3.11+ for the NumPy/pandas recomputation). The two
standard-library entry points below were re-run for this repository on Windows with
Python 3.12; the NumPy/pandas and Matplotlib steps follow the versions pinned in each
package and were verified by the authors (Section 7 of the reproducibility guide).

The archives are already extracted here, so begin from the directory itself rather than
unpacking a zip:

```bash
git clone https://github.com/muyi010630-jpg/hetero-gpu-deadline-calibration.git
cd hetero-gpu-deadline-calibration
```

Main package — standard library only, no GPU and no simulator needed:

```bash
cd reproducibility
python recompute_revision.py
python rebuild_deployment_summaries.py
```

`recompute_revision.py` rechecks the task sets, success states, and violation flags of the
60 runs against 6000 task records and 6000 planned tasks, recomputes within-run nearest-rank
P99, the two-batch absolute means and within-seed paired differences, and verifies the
45 profile units. It prints `PASS` and writes to `recomputed/revision/`.

Supplemental analyses — needs the pinned NumPy and pandas, Python 3.11+:

```bash
python -m pip install -r requirements_recompute.txt
python recompute_supplemental_analyses.py
python recompute_h_event_diagnostics.py
```

These recompute the 25 RQ2 supplementary comparisons, the duration-resolved probability
diagnosis, and the H-event summaries and quantiles from 1,800,000 retained events, writing
to `recomputed/supplemental_analyses/`.

TF32 package:

```bash
cd ../tf32_diagnostic
python -m pip install -r requirements_analysis.txt
python scripts/analyze_tf32.py --input-root raw --output-dir recomputed_tf32
```

This validates the complete design, recomputes the 36 round comparisons and the 12-row
device-context summary (2160 timing windows), and regenerates the figure. Figure drawing
requires Matplotlib; the numerical aggregation uses the standard library.

`reproducibility/recomputed/` ships the authors' verified recomputation output and is
tracked here; rerunning the scripts regenerates it in place, which is expected to leave the
working tree unchanged. Point the TF32 script at `recomputed_tf32/` rather than at the
bundled `analysis/`, so that the shipped results and checksum files there are not
overwritten.

## Scope

**Supported.** Rechecking the reported task counts, denominators, violation flags,
within-run P99, batched and paired metrics, the 45-unit transfer decomposition, the 25
supplementary comparisons, the H-event summaries, and the TF32 timing summaries from the
retained records.

**Not supported.** Rerunning the GPU experiments from this delivery, recovering raw traces
that were never included, reconstructing complete TF32 output matrices, or reconstructing
the complete controller state of each historical run. The two deployment batches are five
seeds × two configurations × three strategies each; within-batch consistency does not
establish that historical parameter selection was independent of evaluation, and the two
batches are not ten identically distributed replications in one environment.

These limits are stated in full in Section 4 of the reproducibility guide and are also
reflected in the manuscript's limitations. They are properties of the evidence that was
retained, not of this repository.

## Hardware and software environment

The three-host measurements used an RTX 5070 (PyTorch 2.13.0+cu132 / CUDA 13.2), a T4 and
an A100-PCIE-40GB (PyTorch 2.5.1+cu124 / CUDA 12.4). The TF32 diagnosis used
PyTorch 2.5.1+cu124 / CUDA 12.4 on A100 and T4, and PyTorch 2.11.0+cu128 / CUDA 12.8 on
the RTX 5070. The RTX 5070 driver was 616.64 for all reported experiments; recorded
metadata that conflicts with this is explained in `DRIVER_METADATA_CORRECTION.md` inside
each package. The authoritative environment record for each TF32 round is the per-round
`environment.json`.

## Licenses

- Author-created source code: MIT License — [`LICENSE-CODE.txt`](LICENSE-CODE.txt), also
  included in each package.
- Author-created data, figures, and explanatory documentation: CC BY 4.0 —
  [`LICENSE-DATA.txt`](LICENSE-DATA.txt), also included in each package.
- Third-party material and external dependencies retain their original terms.

## Citation

If you use these materials, please cite the article; see [`CITATION.cff`](CITATION.cff).
The article DOI will be added here once it is assigned.

## Contact

Corresponding author: Jiangyu Guo, North Automatic Control Technology Institute, Taiyuan
030006, Shanxi, China — tjupyj0914@163.com

## Integrity

The SHA-256 of each archive as supplied with the submission is recorded in
[`PACKAGE_SHA256.txt`](PACKAGE_SHA256.txt). Per-file checksums for the extracted trees are
`reproducibility/package_manifest_sha256.csv` and `tf32_diagnostic/delivery_manifest.json`.
