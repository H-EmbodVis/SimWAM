#!/usr/bin/env python3
"""Build every official Waymo test input from exact dense-extracted frames."""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import tempfile
import time

from preprocess_waymo_train import (
    COORDINATE_FRAME, DEFAULT_IMAGE_ROOT, DEFAULT_METADATA_DIR, LocalImages, atomic_json,
    command_info, decode_json, ego_state, image_uri, integer, relative_to_project,
    timestamp_from_record, validate_relative_image_path,
)

DEFAULT_METADATA = DEFAULT_METADATA_DIR / "test.jsonl"
# Official per-scene frame index shipped with the WOD E2E test split.
DEFAULT_FRAMES = Path("./data/waymo/test_sequence_frames_for_submission.json")
DEFAULT_PREFIX = "images"


def load_submission_tokens(path: str | Path) -> list[str]:
    path = Path(path)
    if path.suffix == ".json":
        mapping = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(mapping, dict):
            raise ValueError("Official frame JSON must map scene IDs to frame indices")
        tokens = []
        for scene, value in mapping.items():
            frame = integer(value, "submission frame index")
            if not scene or frame < 0:
                raise ValueError("Invalid scene/frame identity in official manifest")
            tokens.append(f"{scene}-{frame:03d}")
    else:
        tokens = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not tokens or len(tokens) != len(set(tokens)):
        raise ValueError("Official frame manifest is empty or contains duplicate tokens")
    return tokens


def build_test_record(raw: dict, prefix: str = DEFAULT_PREFIX) -> dict:
    token = raw["context_name"]
    scene, suffix = token.rsplit("-", 1)
    frame = integer(raw["frame_idx"], "frame_idx")
    if raw.get("split") != "test" or scene != raw["scene_id"] or int(suffix) != frame:
        raise ValueError("Test context_name, scene_id, frame_idx or split disagree")
    velocity, acceleration = ego_state(raw)
    raw_intent, nav, command, canonical = command_info(raw)
    current_image = image_uri(prefix, (raw.get("image_paths") or {}).get("FRONT"))
    if not current_image:
        raise ValueError(f"Test frame has no FRONT image: {token}")
    timestamp = timestamp_from_record(raw)
    real_timestamp = timestamp is not None and timestamp > 0
    return {
        "schema_version": "simwam_waymo_test_v1",
        "dataset": "waymo_wod_e2e", "split": "test",
        "clip_id": scene, "token": token, "frame_name": token,
        "current_frame_id": frame,
        "anchor_timestamp_us": timestamp if real_timestamp else frame * 100000,
        "timestamp_source": "metadata" if real_timestamp else "frame_index_10hz",
        "timestamp_reference": "source_time" if real_timestamp else "scene_relative",
        "intent": canonical, "source_intent": raw_intent,
        "nav_command": nav, "driving_command": command,
        "ego_velocity": velocity, "ego_acceleration": acceleration,
        "ego_status": {"ego_pose": [0.0, 0.0, 0.0], "ego_velocity": velocity,
                       "ego_acceleration": acceleration, "driving_command": command,
                       "in_global_frame": False},
        "coordinate_frame": COORDINATE_FRAME,
        "acceleration_correction": "raw_delta_v_divided_by_0.25s",
        "front_image": current_image,
        "future_front_images_1s_to_4s": [],
        "video_frame_mode": "current_only", "video_frame_offsets_s": [0.0],
        "prediction_horizon_s": 5, "prediction_hz": 4,
        "prediction_times_4hz": [i / 4 for i in range(1, 21)],
        "action_dim": 2, "has_heading": False,
        "has_future_ground_truth": False,
    }


def select_test_records(metadata: Path, tokens: list[str], prefix: str):
    selected, seen = {}, set()
    digest = hashlib.sha256()
    scanned = 0
    required = set(tokens)
    with metadata.open("rb") as handle:
        for line in handle:
            digest.update(line)
            if not line.strip():
                continue
            scanned += 1
            raw = decode_json(line)
            token = raw.get("context_name")
            if token not in required:
                continue
            if token in seen:
                raise ValueError(f"Duplicate required frame in metadata: {token}")
            seen.add(token)
            selected[token] = build_test_record(raw, prefix)
    missing = required - seen
    if missing:
        raise ValueError(f"Metadata misses {len(missing)} exact submission frames: {sorted(missing)[:5]}")
    return [selected[token] for token in tokens], scanned, digest.hexdigest()


def run(args):
    metadata, frames = Path(args.metadata).resolve(), Path(args.frames).resolve()
    output = Path(args.output).absolute()
    summary = output.with_suffix(".summary.json")
    for target in (output, summary):
        if target.resolve() in {metadata, frames}:
            raise ValueError("Outputs must differ from source metadata and the official manifest")
        if target.exists() and not args.force:
            raise FileExistsError(f"Output exists: {target}; use another output or --force")
    tokens = load_submission_tokens(frames)
    validate_relative_image_path(args.image_prefix)
    image_root = Path(args.image_root)
    if args.image_check != "none" and not image_root.is_dir():
        raise ValueError(f"--image-root does not exist: {image_root}")
    start = time.monotonic()
    before = metadata.stat()
    records, scanned, digest = select_test_records(metadata, tokens, args.image_prefix)
    print(f"Selected {len(records)} exact test frames from {scanned} metadata records.", flush=True)
    images = LocalImages("decode" if args.image_check == "decode" else "list", image_root)

    def check(record):
        if args.image_check == "none":
            return True
        if not images.resolve(record["front_image"]).is_file():
            return False
        return images.decodable(record["front_image"])

    missing = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for index, (record, ok) in enumerate(zip(records, pool.map(check, records)), 1):
            if not ok:
                missing.append(record["token"])
            if index % 200 == 0 or index == len(records):
                print(f"Checked current FRONT images: {index}/{len(records)}; missing/unreadable={len(missing)}.", flush=True)
    if missing:
        raise ValueError(f"{len(missing)} required current images are unavailable: {missing[:10]}")
    after = metadata.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError("Source metadata changed during preprocessing")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile("w", dir=output.parent, prefix=output.name+".", suffix=".tmp",
                                         delete=False, encoding="utf-8") as handle:
            temporary = Path(handle.name)
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
        temporary.replace(output)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    report = {
        "metadata": relative_to_project(metadata), "metadata_rows": scanned, "metadata_sha256": digest,
        "official_frames": relative_to_project(frames),
        "official_frames_sha256": hashlib.sha256(frames.read_bytes()).hexdigest(),
        "output": relative_to_project(output), "output_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "num_samples": len(records), "unique_tokens": len(set(tokens)),
        "exact_frame_matches": len(records), "fallback_to_earlier_frame": 0,
        "command_counts": dict(Counter(record["intent"] for record in records)),
        "image_check": args.image_check, "image_prefix": args.image_prefix,
        "image_root": relative_to_project(image_root),
        "image_path_convention": "manifest paths are relative to image_root",
        "current_images_checked": len(records) if args.image_check != "none" else 0,
        "future_ground_truth_included": False, "preference_trajectories_included": False,
        "raw_acceleration_scale": 4.0, "coordinate_frame": COORDINATE_FRAME,
        "prediction_grid_seconds": [i/4 for i in range(1, 21)],
        "elapsed_seconds": round(time.monotonic()-start, 2),
    }
    atomic_json(summary, report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", default=str(DEFAULT_METADATA))
    parser.add_argument("--frames", default=str(DEFAULT_FRAMES))
    parser.add_argument("--image-prefix", default=DEFAULT_PREFIX,
                        help="Relative prefix prepended to each metadata FRONT path")
    parser.add_argument("--image-root", default=str(DEFAULT_IMAGE_ROOT),
                        help="Directory the manifest image paths are relative to")
    parser.add_argument("--output", default="./data/waymo_test_front_current_input.jsonl")
    parser.add_argument("--image-check", choices=("head", "decode", "none"), default="head",
                        help="head: verify files exist under --image-root; decode: also open them; none: metadata only")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("workers must be positive")
    run(args)


if __name__ == "__main__":
    main()
