#!/usr/bin/env python3
"""One-shot fix: add DH_API to public classes that have out-of-line
implementations in the core shared library but were missing the export
marker. On Windows the shared library therefore did not export them and
every test executable failed to link against the import library; on Linux
default symbol visibility masked the problem.

The script is idempotent: classes already carrying DH_API are skipped.
"""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# file -> class names that must carry DH_API
TARGETS = {
    "include/core/thread_base.h": ["ThreadBase"],
    "include/core/render_thread.h": ["RenderThread", "RenderMetrics"],
    "include/core/face_detector.h": ["FaceDetector"],
    "include/core/video_processor.h": ["VideoProcessor"],
    "include/core/worker_registry.h": ["WorkerRegistry"],
    "include/core/face_mask_generator.h": ["FaceMaskGenerator"],
    "include/core/image_loader.h": ["ImageLoader", "ImageLoaderException"],
    "include/core/av_sync.h": ["AVSync"],
    "include/core/face_aligner.h": ["FaceAligner"],
    "include/core/inference_worker.h": ["InferenceWorker", "InferenceMetrics"],
    "include/core/audio_sync_scheduler.h": ["AudioSyncScheduler"],
    "include/core/audio_processor.h": ["AudioProcessor"],
    "include/core/pipeline.h": ["Pipeline", "PipelineMetrics"],
    "include/core/frame_scheduler.h": ["FrameScheduler", "FrameStats"],
    "include/model/model_loader.h": ["ModelLoader"],
    "include/model/model_inferencer.h": ["ModelInferencer"],
    "include/model/output_processor.h": ["OutputProcessor"],
    "include/audio/audio_player.h": ["IAudioPlayer", "AudioPlayer"],
    "include/audio/audio_ring_buffer.h": ["RingBuffer"],
    "include/audio/audio_loader.h": ["AudioLoader", "AudioLoaderException"],
    "include/audio/audio_mel_feature_extract.h": ["MelFeatureExtract"],
    "include/audio/audio_vad.h": ["VoiceActivityDetector"],
    "include/audio/audio_preemphasis.h": ["PreEmphasis"],
    "include/audio/audio_framer.h": ["AudioFramer", "AudioFramerException"],
    "include/audio/audio_rms_normalize.h": ["RMSNormalize"],
    "include/audio/audio_noise_reduction.h": ["NoiseReduction"],
    "include/audio/audio_cmvn.h": ["CMVN"],
    "include/network/http_client.h": ["HttpClient"],
}

EXPORT_INCLUDE = '#include "digital_human/export.h"'


def ensure_export_include(text: str, rel: str) -> str:
    if EXPORT_INCLUDE in text:
        return text
    lines = text.splitlines(keepends=True)
    # insert after the first block of #include lines (or after #pragma once)
    insert_at = None
    saw_pragma = False
    last_include = None
    for i, line in enumerate(lines):
        if line.strip() == "#pragma once":
            saw_pragma = True
            continue
        if re.match(r'\s*#\s*include\s', line):
            last_include = i
        elif last_include is not None and line.strip() and not line.startswith("#"):
            break  # end of the include block
    if last_include is not None:
        insert_at = last_include + 1
    elif saw_pragma:
        for i, line in enumerate(lines):
            if line.strip() == "#pragma once":
                insert_at = i + 1
                break
    if insert_at is None:
        raise SystemExit(f"{rel}: cannot locate include block / #pragma once")
    lines.insert(insert_at, EXPORT_INCLUDE + "\n")
    print(f"  + inserted export.h include into {rel}")
    return "".join(lines)


def main() -> None:
    for rel, classes in TARGETS.items():
        path = ROOT / rel
        text = path.read_text(encoding="utf-8")
        original = text
        text = ensure_export_include(text, rel)
        for cls in classes:
            # class/struct keyword, optional DH_API, the exact class name
            pattern = re.compile(
                r"^(?P<indent>\s*)(?P<kw>class|struct)\s+(?!DH_API\b)"
                r"(?P<name>" + re.escape(cls) + r")\b",
                re.MULTILINE,
            )
            if not pattern.search(text):
                if re.search(r"^\s*(class|struct)\s+DH_API\s+" + re.escape(cls) + r"\b",
                             text, re.MULTILINE):
                    print(f"  = {rel}: {cls} already exported")
                    continue
                raise SystemExit(f"{rel}: declaration of {cls} not found")
            text = pattern.sub(lambda m: f"{m.group('indent')}{m.group('kw')} DH_API {m.group('name')}", text)
            print(f"  * {rel}: +DH_API {cls}")
        if text != original:
            path.write_text(text, encoding="utf-8", newline="\n")
    print("done")


if __name__ == "__main__":
    main()
