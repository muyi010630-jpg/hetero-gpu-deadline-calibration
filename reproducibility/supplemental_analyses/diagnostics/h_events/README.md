# Retained H event diagnostics

This directory contains the seven diagnostic fields used by the archived H-event aggregation for adaptive_pool_local, extracted without numerical changes from 1000 original fixed-feedback NPZ streams. Each stream contains 1800 prediction-before-update events, giving 1800000 events in total. The field order is stored in diagnostic_fields in each file.

The original source file hashes were checked against the historical source_sha256.csv before extraction. source_to_extract_sha256.csv records the full-source hash, extracted-file hash, event count, mode, and selected fields. Complete original NPZ files include other arrays and remain outside this delivery. Their historical hashes continue to identify the complete source files.

Run python recompute_h_event_diagnostics.py from the reproducibility directory after installing requirements_recompute.txt. The entry point calls the archived heterogeneity_summary function, checks extract hashes, and compares the reconstructed stream, image-block, and condition tables with the saved tables. Output is written to recomputed/h_events by default.

The extraction supports the reported H-event means, fractions, and empirical quantiles. It does not add simulations, alter the estimator, or introduce inferential tests. Data are covered by LICENSE_DATA.txt at the archive root.
