#!/usr/bin/env python3
"""Convert native Waymo 20x2 test predictions to official binproto/tar.gz files."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
import tarfile

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from preprocess_waymo_test import DEFAULT_FRAMES, load_submission_tokens
from waymo_submission_proto import DEFAULT_WAYMO_SRC, load_submission_proto


def load_native_predictions(path: Path, expected_tokens: list[str]):
    records = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            token = row.get("token")
            if not isinstance(token, str) or not token or token in records:
                raise ValueError(f"{path}:{line_number}: missing or duplicate token")
            if row.get("frame_name", token) != token:
                raise ValueError(f"frame_name differs from token: {token}")
            if row.get("trajectory_hz", 4) != 4 or row.get("trajectory_horizon_s", 5) != 5:
                raise ValueError("Predictions must already be 5s at 4Hz")
            xy = np.asarray(row["predicted_traj"], dtype=np.float64)
            if xy.shape != (20, 2) or not np.isfinite(xy).all():
                raise ValueError(f"{token}: expected exactly 20 finite XY points; no interpolation or heading conversion")
            xy = xy.astype(np.float32)
            if not np.isfinite(xy).all():
                raise ValueError(f"{token}: coordinates overflow the official float32 fields")
            records[token] = xy
    missing, extra = set(expected_tokens)-records.keys(), records.keys()-set(expected_tokens)
    if missing or extra:
        raise ValueError(f"Submission coverage mismatch: missing={len(missing)}, extra={len(extra)}")
    return [(token, records[token]) for token in expected_tokens]


def validate_metadata(metadata):
    result = dict(metadata)
    strings = ("account_name", "unique_method_name", "affiliation", "description", "method_link", "num_model_parameters")
    for key in strings:
        value = result.get(key) or ""
        if not isinstance(value, str):
            raise ValueError(f"{key} must be a string")
        result[key] = value.strip()
    for key in ("authors", "public_model_names"):
        value = result.get(key) or []
        if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
            raise ValueError(f"{key} must be a list of strings")
        result[key] = value
    flag = result.get("uses_public_model_pretraining")
    if flag is not None and not isinstance(flag, bool):
        raise ValueError("uses_public_model_pretraining must be an explicit boolean")
    if result["account_name"] and not re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", result["account_name"]):
        raise ValueError("account_name must be the registered Waymo email")
    if result["num_model_parameters"] and not re.fullmatch(r"[1-9]\d*[KMBT]", result["num_model_parameters"]):
        raise ValueError("num_model_parameters must be an integer with K/M/B/T suffix")
    missing = [key for key in ("account_name", "unique_method_name", "num_model_parameters") if not result[key]]
    if flag is None:
        missing.append("uses_public_model_pretraining")
    if flag is True and not result["public_model_names"]:
        missing.append("public_model_names")
    return result, missing


def convert_submission(predictions_path, frames_path, output_dir, metadata, *,
                       prefix="simwam_waymo", num_shards=1, allow_incomplete_metadata=False,
                       force=False, waymo_src=DEFAULT_WAYMO_SRC):
    predictions_path, frames_path = Path(predictions_path).absolute(), Path(frames_path).absolute()
    output_dir = Path(output_dir).absolute()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", prefix):
        raise ValueError("Submission prefix must be a filename, without directory separators")
    expected = load_submission_tokens(frames_path)
    if not 1 <= num_shards <= len(expected):
        raise ValueError("num_shards must be between 1 and the number of test frames")
    metadata, missing = validate_metadata(metadata)
    if missing and not allow_incomplete_metadata:
        raise ValueError("Missing official submission metadata: " + ", ".join(missing)
                         + ". Fill the metadata JSON, or explicitly request a draft with --allow-incomplete-metadata.")
    predictions = load_native_predictions(predictions_path, expected)
    archive = Path(str(output_dir) + ".tar.gz")
    if (output_dir.exists() or archive.exists()) and not force:
        raise FileExistsError("Output directory/archive exists; choose a new output or use --force")
    if output_dir.resolve() in {predictions_path.resolve(), frames_path.resolve()}:
        raise ValueError("Output must differ from prediction and frame input files")
    proto = load_submission_proto(str(waymo_src))
    output_dir.mkdir(parents=True, exist_ok=True)
    if force:
        for old in output_dir.glob(prefix + ".binproto-*"):
            old.unlink()
    files, checks = [], []
    seen = []
    for index in range(num_shards):
        start = index * len(predictions) // num_shards
        stop = (index + 1) * len(predictions) // num_shards
        records = predictions[start:stop]
        if not records:
            raise ValueError("Shard plan would produce an empty shard")
        message = proto.E2EDChallengeSubmission(submission_type=proto.E2ED_SUBMISSION)
        for token, xy in records:
            message.predictions.add(
                frame_name=token,
                trajectory=proto.TrajectoryPrediction(pos_x=xy[:, 0].tolist(), pos_y=xy[:, 1].tolist()),
            )
        for key in ("account_name", "unique_method_name", "affiliation", "description", "method_link", "num_model_parameters"):
            if metadata[key]:
                setattr(message, key, metadata[key])
        message.authors.extend(metadata["authors"])
        message.public_model_names.extend(metadata["public_model_names"])
        if metadata.get("uses_public_model_pretraining") is not None:
            message.uses_public_model_pretraining = metadata["uses_public_model_pretraining"]
        path = output_dir / f"{prefix}.binproto-{index:05d}-of-{num_shards:05d}"
        path.write_bytes(message.SerializeToString())
        parsed = proto.E2EDChallengeSubmission.FromString(path.read_bytes())
        if parsed.submission_type != proto.E2ED_SUBMISSION or len(parsed.predictions) != len(records):
            raise RuntimeError("Invalid protobuf roundtrip")
        for actual, (token, xy) in zip(parsed.predictions, records):
            if actual.frame_name != token:
                raise RuntimeError("Official frame name changed during serialization")
            restored = np.column_stack([actual.trajectory.pos_x, actual.trajectory.pos_y]).astype(np.float32)
            np.testing.assert_array_equal(restored, xy)
            seen.append(actual.frame_name)
        files.append(path)
        checks.append({"name": path.name, "num_predictions": len(records),
                       "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    if seen != expected:
        raise RuntimeError("Serialized submission has missing/duplicate/out-of-order frames")
    with tarfile.open(archive, "w:gz", format=tarfile.GNU_FORMAT) as handle:
        for path in files:
            handle.add(path, arcname=path.name)
    with tarfile.open(archive, "r:gz") as handle:
        if handle.getnames() != [path.name for path in files]:
            raise RuntimeError("Submission tar must contain only flat binproto shards")
        for path in files:
            if handle.extractfile(path.name).read() != path.read_bytes():
                raise RuntimeError("Archived protobuf differs from the validated shard")
    report = {
        "num_predictions": len(predictions), "num_required_frames": len(expected),
        "exact_frame_coverage": True, "num_points": 20, "frequency_hz": 4, "horizon_s": 5,
        "coordinates": "ego frame, x forward, y left, meters",
        "trajectory_resampling": False, "protobuf_xy_roundtrip_exact": True,
        "format_valid": True, "submission_ready": not missing,
        "missing_required_metadata": missing, "metadata": metadata,
        "predictions_path": str(predictions_path),
        "predictions_sha256": hashlib.sha256(predictions_path.read_bytes()).hexdigest(),
        "official_frames": str(frames_path),
        "official_frames_sha256": hashlib.sha256(frames_path.read_bytes()).hexdigest(),
        "source_proto": proto.source_proto, "source_proto_sha256": proto.source_proto_sha256,
        "official_descriptor_sha256": proto.descriptor_sha256,
        "shards": checks, "tar_layout": "flat GNU", "archive": str(archive),
        "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
    }
    (output_dir / "submission_manifest.json").write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True, help="Native SimWAM predictions.jsonl")
    parser.add_argument("--frames", default=str(DEFAULT_FRAMES))
    parser.add_argument("--metadata-json", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--prefix", default="simwam_waymo")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--allow-incomplete-metadata", action="store_true",
                        help="Create a clearly marked draft while identity fields remain blank")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--waymo-src", default=DEFAULT_WAYMO_SRC)
    args = parser.parse_args()
    report = convert_submission(
        args.predictions, args.frames, args.output_dir,
        json.loads(Path(args.metadata_json).read_text()),
        prefix=args.prefix, num_shards=args.num_shards,
        allow_incomplete_metadata=args.allow_incomplete_metadata, force=args.force, waymo_src=args.waymo_src,
    )
    print(json.dumps({key: report[key] for key in (
        "num_predictions", "exact_frame_coverage", "format_valid", "submission_ready",
        "missing_required_metadata", "archive", "archive_sha256")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
