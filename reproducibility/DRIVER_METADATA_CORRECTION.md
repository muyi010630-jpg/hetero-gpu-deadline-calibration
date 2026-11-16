# RTX5070 driver metadata correction

## Author confirmation

On 10 November 2026, the author explicitly confirmed in this chat that all RTX5070 experiments reported in the accompanying study used NVIDIA driver 616.64 and that the conflicting driver-version entries in the archived records were erroneous. Use 616.64 as the author-confirmed driver version for the RTX5070 original profiles, runtime profiles, deployment batches, and TF32 measurements covered by this study.

The correction is based on the author's explicit confirmation in this chat. The archived recordings and the author-confirmed value are distinguished in DRIVER_METADATA_CORRECTIONS.csv. The source archive for this correction is Reproducibility_Data_and_Code.zip. Its SHA256 is d1584a98a44a31c8044435ed03ec24a9ea7f2fac6f43a6a2fcbfd9ec660aa820.

## Recorded values and locations

DRIVER_METADATA_CORRECTIONS.csv contains 151 driver-value records across 32 source files in this archive. Paths in the CSV are relative to the archive root directory. CSV data record numbers exclude the header. JSON locations identify the named field or command output. Each row records the original file hash, the recorded version, the author-confirmed version, the correction date, and the basis for the correction.

| Recorded driver version | Associated records |
| --- | --- |
| 610.88 | 1 |
| 616.92 | 150 |

## Interpretation of archived metadata

The file hashes identify the source bytes preserved in this delivery. Original environment JSON files, Worker readiness CSV files, recorded GPU-query output, measurement records, and archived analysis summaries remain available for traceability. For RTX5070 driver-version reporting, this author confirmation takes precedence over the conflicting driver values in those records and in dated collection notes. Other software and hardware fields retain their recorded interpretation.

Driver values were identified through driver-version fields, the RTX5070 row identity, and the driver_version column of recorded NVIDIA query output. Numerical timing or diagnostic values were excluded from the correction inventory. The correction documents metadata provenance. It supplies no new experimental measurements.

## Delivery checksums

The current package manifest covers the revised README and the two correction files. Historical source hashes continue to identify their original records. The correction date is 2026-11-10.
