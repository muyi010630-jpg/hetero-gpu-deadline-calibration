# RTX5070 driver metadata correction

## Author confirmation

On 10 November 2026, the author explicitly confirmed in this chat that all RTX5070 experiments reported in the accompanying study used NVIDIA driver 616.64 and that the conflicting driver-version entries in the archived records were erroneous. Use 616.64 as the author-confirmed driver version for the RTX5070 original profiles, runtime profiles, deployment batches, and TF32 measurements covered by this study.

The correction is based on the author's explicit confirmation in this chat. The archived recordings and the author-confirmed value are distinguished in DRIVER_METADATA_CORRECTIONS.csv. The source archive for this correction is TF32_Data_and_Code.zip. Its SHA256 is 18dbdf0eb228a1476b273fc1e986f92b3e263125733143f9698e1873b2cbfe8f.

## Recorded values and locations

DRIVER_METADATA_CORRECTIONS.csv contains 9 driver-value records across 7 source files in this archive. Paths in the CSV are relative to the archive root directory. CSV data record numbers exclude the header. JSON locations identify the named field or command output. Each row records the original file hash, the recorded version, the author-confirmed version, the correction date, and the basis for the correction.

| Recorded driver version | Associated records |
| --- | --- |
| 616.92 | 9 |

## Interpretation of archived metadata

Source-file hashes identify the corresponding files in the source archive before this delivery correction. Original environment JSON files, recorded GPU-query output, measurement records, and archived analysis summaries are retained for traceability. The README environment paragraph states the author-confirmed value and points to this correction. For RTX5070 driver-version reporting, this author confirmation takes precedence over the conflicting driver values in archived records and dated collection notes. Other software and hardware fields retain their recorded interpretation.

Driver values were identified through driver-version fields, the RTX5070 row identity, and the driver_version column of recorded NVIDIA query output. Numerical timing or diagnostic values were excluded from the correction inventory. The correction documents metadata provenance. It supplies no new experimental measurements.

## Delivery checksums

The current delivery manifest covers the revised README and the two correction files. Historical source hashes continue to identify their original records. The correction date is 2026-11-10.
