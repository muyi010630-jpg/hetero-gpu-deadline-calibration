# Portable delivery notes

This delivery uses English archive names and package-relative file identifiers. Local and remote machine paths in saved strings were replaced with package paths or explicit source-location placeholders. These textual changes do not create missing historical files. Filenames are mapped in PATH_RENAMING_MAP.csv.

PROVENANCE_TRANSFORMATIONS.csv retains the SHA256 of each original supplied file. The original records remain preserved by the authors outside this delivery. Those historical hashes identify the original bytes; current delivery hashes are separate.

Authoritative current checksums are package_manifest_sha256.csv for the main archive and delivery_manifest.json for the TF32 archive. Other hash tables retain recorded_source_sha256; hash_scope distinguishes a current_delivery target from a historical_source_not_in_archive. Historical source hashes inside frozen plans and logs remain historical identifiers, not assertions that the cleaned delivery bytes are unchanged.

Numerical observations, scientific tables, inferential families, random seeds, and stored image bytes are preserved. Some raw traces and H-event shards were not supplied. Complete TF32 output matrices were not retained. The historical controller snapshots and all original execution environments cannot be fully reconstructed.

The supported offline recomputation entry points locate packaged data relative to their own scripts. TF32 figure generation uses installed fonts with a DejaVu Sans fallback; no personal font directory is required. The archived GPU collection interface requires an independently configured environment and does not form part of the verified offline recomputation. It accepts TF32_LOCAL_PYTHON, TF32_REMOTE_WORKDIR, TF32_REMOTE_PYTHON, and TF32_HOST_KEYS.
