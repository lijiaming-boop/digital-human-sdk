#!/usr/bin/env python3
"""Select a model only if it passes end-to-end FPS and lip-quality gates.

Usage: select_wav2lip_variant.py <reports-root> [selection.json]
Each direct child must contain report.json written by pipeline_lipsync_test.
"""
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
output = pathlib.Path(sys.argv[2]) if len(sys.argv) > 2 else root / "selection.json"
reports = {}
for path in sorted(root.glob("*/report.json")):
    try:
        reports[path.parent.name] = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"skip {path}: {exc}", file=sys.stderr)

if "fp32" not in reports:
    raise SystemExit("fp32/report.json is required as the quality baseline")

baseline = reports["fp32"]
base_corr = max(float(baseline.get("mouth_correlation", 0.0)),
                float(baseline.get("mouth_correlation_smooth", 0.0)))
base_mouth = float(baseline.get("mouth_mean", 0.0))
if not baseline.get("fps_ok", False) or base_mouth <= 1.0:
    raise SystemExit("FP32 baseline does not produce a usable end-to-end result")
eligible = []
for name, report in reports.items():
    corr = max(float(report.get("mouth_correlation", 0.0)),
               float(report.get("mouth_correlation_smooth", 0.0)))
    fps = float(report.get("content_fps", 0.0))
    infer = float(report.get("avg_inference_ms", float("inf")))
    # The absolute mouth-energy proxy is intentionally not a promotion gate: it is
    # input dependent. Candidates must preserve the FP32 baseline's mouth motion
    # and temporal correlation, then still pass human review for release.
    quality_ok = (float(report.get("mouth_mean", 0.0)) >= base_mouth * 0.90
                  and corr >= base_corr * 0.90)
    performance_ok = bool(report.get("fps_ok", False)) and fps >= 24.0
    report["quality_gate_passed"] = quality_ok
    report["performance_gate_passed"] = performance_ok
    if quality_ok and performance_ok:
        eligible.append((infer, -fps, name))

# The reference model is always retained as a safe rollback even when no candidate passes.
winner = min(eligible)[2] if eligible else "fp32"
result = {
    "winner": winner,
    "fallback": "fp32",
    "baseline_lip_correlation": base_corr,
    "selection_rule": "mouth motion and lip-energy correlation >= 90% of FP32, FPS >= 24; then lowest average inference latency (human visual review still required)",
    "reports": reports,
}
output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(f"winner={winner}; fallback=fp32; report={output}")
