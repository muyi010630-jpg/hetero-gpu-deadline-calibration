"""Seal completed report/data files; does not alter frozen experimental artifacts."""
from pathlib import Path
from datetime import datetime, timezone
from collections import Counter
import csv
import hashlib
import json

root = Path(__file__).resolve().parents[1]
excluded = {"artifact_manifest.json", "artifact_manifest.csv"}
records = []
for path in sorted(root.rglob("*"), key=lambda p: p.as_posix()):
    if not path.is_file() or "__pycache__" in path.parts:
        continue
    rel = path.relative_to(root).as_posix()
    if rel in excluded:
        continue
    with path.open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    records.append({"path": rel, "bytes": path.stat().st_size, "sha256": digest})
checks = json.loads((root / "qa/artifact_checks.json").read_text(encoding="utf-8"))
analysis = json.loads((root / "03_heldout_closed_loop/analysis/analysis_metadata.json").read_text(encoding="utf-8"))
assert checks["passed"] and checks["trace_hashes_all_checked"]
assert sum(stage["runs"] for stage in checks["stages"].values()) == 5664
assert analysis["probe_gate"]["passed"] is False
manifest = {
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "status": "completed; probe gate failed; no active probing added",
    "experiment_root": str(root),
    "scope": "CPU feasibility prototype; not real GPU validation",
    "formal_runs": {key: value["runs"] for key, value in checks["stages"].items()},
    "file_count": len(records),
    "total_bytes": sum(row["bytes"] for row in records),
    "top_level_file_counts": dict(Counter(row["path"].split("/")[0] for row in records)),
    "exclusions": ["artifact_manifest.json", "artifact_manifest.csv", "**/__pycache__/**"],
    "verification": "Each included file hashed in full. Formal trace integrity and source freeze checked separately in qa/artifact_checks.json.",
    "files": records,
}
(root / "artifact_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
with (root / "artifact_manifest.csv").open("w", newline="", encoding="utf-8-sig") as handle:
    writer = csv.DictWriter(handle, fieldnames=["path", "bytes", "sha256"])
    writer.writeheader()
    writer.writerows(records)
print(json.dumps({"manifest_files": len(records), "total_bytes": manifest["total_bytes"], "probe_gate_passed": False}, ensure_ascii=False))
