# Reproducibility Guide

**Manuscript:** Closed-Loop Online Performance Calibration for Deadline-Aware Scheduling in Heterogeneous GPU Pools
**Companion files:** the main manuscript identified by the title above and `Supplementary_Materials.pdf` (supplement)
**Index version:** 20261110

This index tells a reader which archive holds which evidence, how to reach the entry point for each figure and table, and what can and cannot be reproduced from the supplied files. It does not add results and does not replace the package READMEs.

## 1. The two archives

| Archive (actual file name) | Size | Entries | Role |
|---|---|---|---|
| `Reproducibility_Data_and_Code.zip` | 73,604,511 bytes | 1459 | Profiles, per-run and task-level records, calibrator reference implementation, summaries, plotting data |
| `TF32_Data_and_Code.zip` | 929,623 bytes | 247 | TF32 timing, configurations, environments, statistical summaries |

The two archives are deliberately separate. The 2026-10-15 package predates the TF32 diagnosis and does not contain it; this is not an omission. Archive entries now use English names and package-relative paths. PATH_RENAMING_MAP.csv maps historical identifiers to the delivery names; PROVENANCE_TRANSFORMATIONS.csv retains original supplied-file hashes.

The current manifests cover 1458 main-package files and 246 TF32 files, excluding each manifest itself. Both archives are submitted for review and will be published as Supplementary Materials with the article upon acceptance. Historical source hashes and current-delivery hashes identify their respective records. This delivery includes the 1000 H-event extracts documented below.

## 2. Where each figure and table comes from

| Item in the main manuscript | Evidence entry |
|---|---|
| Figure 1: calibration/scheduling loop | Conceptual schematic; no independent numerical dataset |
| Figure 2: profile heat map | Main archive: profiles/canonical.csv; profiles/profile_samples_and_migration.csv |
| Figure 3: early prediction convergence | Main archive: cpu/pooled_ema_cpu_20261009/01_independent_calibration/analysis/summary_first_k.csv and summary_first_k_per_seed.csv |
| Figure 4: shared-fraction comparison | Main archive: figures/figure4_plot_data.csv; figure4_paired_image_differences.csv |
| Figure 5; Supplementary Figure S4: risk coverage | Main archive: figures/reference_figure5/figure5_plot_data.csv |
| Figure 6: three-host paired differences | Main archive: figures/figure6_statistics.csv; figure6_points.csv |
| Figure 7; Supplementary Figure S11: TF32 ratios | TF32 archive: analysis/round_ratios.csv; analysis/all_timing_windows.csv |
| Redraw entry for Figures 4 and 6 | Main archive: figures/redraw_figures4_6.py; historical file name retained |

Main-archive paths are relative to `reproducibility/`; TF32 paths are relative to `tf32_diagnostic/`. Supplementary Figures S4 and S11 repeat main-text Figures 5 and 7.

| Main-text table | Entry point |
|---|---|
| Table 1: related work | Literature synthesis; no experimental dataset |
| Table 2: notation | S12; definitions only |
| Table 3: point estimators | S3/S4 and S8; reference implementations under cpu/ |
| Table 4: version scope | S3/S4; version-specific interfaces |
| Table 5: deployment switches | S3; deployment/verified_parameters.csv and reference sources |
| Table 6: deployment batches | Table S3a; deployment/runs/ configuration and readiness records |
| Table 7: diagnostic interfaces | Table S9b; frozen development records under cpu/ |
| Table 8: paired units | Table S8; S6/S8 and saved analysis plans |
| Table 9: common-feedback errors | cpu/pooled_ema_cpu_20261009/01_independent_calibration/analysis/summary_overall.csv; summary_by_phase.csv |
| Table 10: shared-structure closed loop | Table S1b; supplemental_analyses/RQ2/rq2_supplementary_comparisons_25.csv |
| Table 11(a): main replay | main_experiments/main_replay/overall_by_strategy_fixed.csv; per_seed_results.csv |
| Table 11(b): common-risk comparison | cpu/pooled_ema_cpu_20261009/02_scheduling_replay/full/analysis/summary_by_variant_mode.csv; paired_seed_inference.csv |
| Table 12: probability/decision diagnostics | Table S9c; cpu/probability_calibration_cpu_20261010/ and cpu/decision_feedback_cpu_20261010/ |
| Table 13: deployment results | Tables S3/S5; deployment/task_records.csv, batch_absolute_and_paired.csv, paired_comparison_10seed.csv |

| Supplementary item | Entry point |
|---|---|
| Tables S1/S1b | `supplemental_analyses/RQ2/rq2_supplementary_comparisons_25.csv` |
| Table S2 | `supplemental_analyses/diagnostics/probability_horizon_summary.csv`, `probability_horizon_paired.csv` |
| Table S3: batch results; Table S3a: configurations; Table S5: combined metrics | `deployment/`; S3/S5 of the supplement |
| Tables S9a-S9c | `cpu/probability_calibration_cpu_20261010/`, `cpu/decision_feedback_cpu_20261010/` |
| Tables S11a/S11b | TF32 `analysis/round_ratios.csv`, `device_context_summary.csv` |

## 3. Recomputation entry points

Unpack `Reproducibility_Data_and_Code.zip` and change into the `reproducibility/` directory before running anything.

The first two scripts use only the Python standard library and run on Python 3.10 or later:

```text
python recompute_revision.py
python rebuild_deployment_summaries.py
```

- `recompute_revision.py` checks the task sets, success states, and violation flags of the 60 runs against 6000 task records and 6000 planned tasks, recomputes within-run nearest-rank P99, the two-batch absolute means and within-seed paired differences, and verifies the sample count, median, dispersion, and descriptive sum-of-squares decomposition of the 45 profile units. Output goes to `recomputed/revision/` and should report PASS, 60 runs, 6000 successful tasks, 12 batched metric rows, and 45 profile units.
- `rebuild_deployment_summaries.py` rebuilds the 12 batch groups and 6 combined groups from the saved per-run metrics and repairs the historical batch-label error. It does not re-estimate task-level P99.

The third script pins NumPy 2.3.5 and pandas 3.0.1, which require Python 3.11 or later. Install the pinned versions first:

```text
python -m pip install -r requirements_recompute.txt
python recompute_supplemental_analyses.py
```

It recomputes the 25 RQ2 supplementary comparisons and the duration-resolved probability diagnosis from saved CSVs into `recomputed/supplemental_analyses/`. `requirements_recompute.txt` covers only this portable CSV recomputation, not the original CPU or GPU simulation environment.

None of the three creates new significance tests, reselects a statistical family, or invokes a GPU or simulator.

### TF32 archive

Unpack `TF32_Data_and_Code.zip` and change into the tf32_diagnostic/ directory before running:

```text
python scripts/analyze_tf32.py --input-root raw --output-dir recomputed_tf32
```

The script validates the complete design, recomputes the 36 round comparisons and the 12-row device-context summary, and writes `round_ratios.csv`, `device_context_summary.csv`, `all_timing_windows.csv`, `stage_summary.csv`, `output_differences_CPU_float64.csv`, `validation.json`, `figure_captions.json`, the checksum lists, and the figure.

Write to a new directory such as `recomputed_tf32` rather than to the bundled `analysis/`, so that the shipped results and checksum files are not overwritten.

Install the TF32 analysis dependencies with `python -m pip install -r requirements_analysis.txt`. Numerical aggregation uses the standard library; figure generation requires Matplotlib. The script discovers installed fonts and falls back to DejaVu Sans. It records package-relative input paths and output basenames. The complete command was verified on Windows; other platforms were not tested.

## 4. What the packages do and do not support

**Supported.** Rechecking the reported task counts, denominators, violation flags, within-run P99, batched and paired metrics, the 45-unit transfer decomposition, the 25 supplementary comparisons, and the TF32 timing summaries from the saved records.

**Not supported.** Rerunning GPU experiments from this delivery, recovering other omitted large-scale raw traces, reconstructing complete TF32 output matrices, or reconstructing complete historical controller states.

Seven diagnostic fields for adaptive_pool_local are retained from each of the 1000 original fixed-feedback NPZ streams, giving 1,800,000 prediction-before-update events. All source hashes matched the historical manifest. The extracts and source-to-extract checksums are in supplemental_analyses/diagnostics/h_events/. After installing requirements_recompute.txt, run python recompute_h_event_diagnostics.py from reproducibility/. The entry point uses the archived aggregation function to recompute stream, image-block, and condition summaries, including empirical quantiles, and checks them against the saved tables. Complete source NPZ files include additional arrays and are outside this delivery.

Complete TF32 output matrices were not retained. Saved output hashes and difference/reference norms support consistency checks, but do not allow independent recomputation of elementwise differences of the discarded matrices.

Some historical CPU interfaces still carry explicit source-location placeholders such as `<SOURCE_WORKSPACE>`. It is therefore not claimed that the entire experiment set can be rerun with a single command. The large-scale raw and trace shards, and part of the H diagnostic, are described in `README_20261013_history.md` and `supplemental_analyses/diagnostics/source_sha256.csv`.

The 71.72% / 28.28% split of the transfer decomposition is a descriptive sum-of-squares proportion. It is not a random-effects variance component, not the controlled shared fraction of the simulation, and not a recoverable-error proportion.

The two deployment batches correspond to seeds 20260911–15 and 20260916–20, five seeds x two configurations x three strategies each. Seed numbers are not proof of run dates. Within-batch consistency does not establish that historical parameter selection was independent of evaluation, and the two batches are not ten identically distributed replications in one environment.

This delivery further removes recorded local and remote machine directories. Some material also consists of earlier curated copies with connection addresses and user directories removed. `source_and_package_manifest_sha256.csv` lists hashes of the source and the copy separately; `package_manifest_sha256.csv` lists every file in the package and is an identifier of the current curated version, not a snapshot of a historical run. Source-location placeholders describe unavailable source locations and do not identify packaged files.

## 5. Environment

The three-host measurements used an RTX 5070 with PyTorch 2.13.0+cu132 / CUDA 13.2, and a T4 and an A100-PCIE-40GB with PyTorch 2.5.1+cu124 / CUDA 12.4. The TF32 diagnosis used PyTorch 2.5.1+cu124 / CUDA 12.4 on A100 and T4, and PyTorch 2.11.0+cu128 / CUDA 12.8 on the RTX 5070. The authors confirmed on 10 November 2026 that the RTX 5070 driver was 616.64 for all reported experiments. DRIVER_METADATA_CORRECTION.md and DRIVER_METADATA_CORRECTIONS.csv in each archive document the recorded metadata and the author-confirmed correction.

## 6. License, citation, and contact

Author-created source code is licensed under the MIT License in LICENSE_CODE.txt. Author-created data, figures, and explanatory documentation are licensed under CC BY 4.0 in LICENSE_DATA.txt. Both archives include these notices. Third-party materials and external dependencies retain their original terms. The archives are supplied for review and will be published with the article upon acceptance. License terms are available at https://opensource.org/license/mit and https://creativecommons.org/licenses/by/4.0/.

Cite Yujie Pang, Jiangyu Guo, Zhichao Hao, Xiao Guo, Mingjie Huang, and Hao Lü, Closed-Loop Online Performance Calibration for Deadline-Aware Scheduling in Heterogeneous GPU Pools, accompanying data and code, delivery 20261110. Add the article DOI when assigned. Repository identifiers may be added if the authors later deposit an additional archived copy.

Correspondence about the packages: use the corresponding-author contact details in the accompanying manuscript.

## 7. Verification of the portable delivery

On 6 November 2026, the renamed delivery passed archive CRC and current-manifest checks. More than seven million numeric CSV cells were checked against the supplied files and preserved exactly. Stored scientific graphics and the archived methods source document retain their original bytes. Frozen TF32 measurement scripts and protocols also retain their original bytes; the offline analysis interface was made portable.

The revised entry points reproduced 60 runs, 6000 successful tasks, 12 batch-metric rows, and 45 profile units. Supplemental recomputation reproduced 25 comparisons and 500 paired profile differences, the probability summaries from 1728 input run rows, and the saved 1000-stream heterogeneity means. NumPy 2.3.5 and pandas 3.0.1 matched the pinned versions. TF32 analysis passed the complete-design check with 2160 timing windows and 36 round comparisons; figure generation was checked using Matplotlib 3.11.2.

The 20261109 update verifies the complete-source hashes of all 1000 H-event streams and retains their seven required diagnostic fields without numerical changes. The new entry point recomputes event-level summaries and quantiles from 1,800,000 events and checks them against the saved stream, image-block, and condition tables. The previous checks and this update read retained records. Complete TF32 output matrices, other omitted traces, and complete historical controller snapshots remain outside the reproducible scope.