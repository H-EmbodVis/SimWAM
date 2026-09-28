#!/usr/bin/env python3
r"""Build SimWAM Waymo samples directly from extracted native WOD-E2E metadata.

Input is extracted/metadata/{training,val}.jsonl, NOT the processed v2a/10 Hz
ChatML. Preserve native 4 Hz XY targets (5 s / 20 points by default); do not
synthesize headings. Training video contains current FRONT plus +1..+4 s.
Validation defaults to current-image-only rows with valid scored preference
trajectories, preserved as native XY arrays plus their original scores.

Raw past-state velocities/accelerations are in the CURRENT ego coordinate
system. Raw acceleration stores delta-v rather than delta-v / 0.25 s: multiply
by four, preserving the final provided value (the last velocity is duplicated).

The existing metadata omits timestamps, and sampled source TFRecords contain
timestamp_micros=0. In auto mode, a scene without usable timestamps gets a
scene-relative frame_idx / camera_hz timeline, explicitly marked in the output.
Real metadata timestamps or --timestamps-jsonl take precedence. Matching always
uses the scene's time index and only images actually present under --image-root.

Examples:
    python scripts/preprocess_waymo_train.py --split training
    python scripts/preprocess_waymo_train.py --trajectory-horizon-s 4
    python scripts/preprocess_waymo_train.py --max-scenes 1 --limit 3 \
        --image-check decode --workers 1 --output ./data/waymo_smoke.jsonl

Output shares PhysicalAI's clip/token/image/ego fields, but uses
future_traj_4hz with action_dim=2, and stores image paths relative to
--image-root. No raw data, source code or existing output is overwritten
unless --force is supplied for the output files.
"""

from __future__ import annotations

import argparse
import bisect
from collections import Counter, defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path, PurePosixPath
import tempfile
import time
from typing import Any, Callable, Iterator

try:
    import msgspec
except ImportError:
    decode_json = json.loads
else:
    decode_json = msgspec.json.decode

PROJECT_ROOT = Path(__file__).resolve().parents[1]
# Native WOD-E2E extracted metadata; override with --metadata or WAYMO_METADATA_DIR.
DEFAULT_METADATA_DIR = Path(os.environ.get("WAYMO_METADATA_DIR", "./data/waymo/metadata"))
# Manifest image paths are `<image-prefix>/<path-from-metadata>`, stored relative
# to the dataset's --image-root so no absolute or remote location is ever recorded.
DEFAULT_IMAGE_PREFIX = "images"
DEFAULT_IMAGE_ROOT = Path("./data/waymo")
VIDEO_OFFSETS_S = (0.0, 1.0, 2.0, 3.0, 4.0)
TRAJECTORY_HZ = 4
RAW_HISTORY_POINTS = 16
RAW_FUTURE_POINTS = 20
RAW_HISTORY_DT_S = 0.25
COORDINATE_FRAME = "ego_at_anchor_x_forward_y_left"
COMMANDS = {
    "UNKNOWN": (-1, (0, 0, 0, 1), "UNKNOWN"),
    "GO_STRAIGHT": (0, (0, 1, 0, 0), "GO STRAIGHT"),
    "GO_LEFT": (1, (1, 0, 0, 0), "TURN LEFT"),
    "GO_RIGHT": (2, (0, 0, 1, 0), "TURN RIGHT"),
}
RAW_INTENTS = {0: "UNKNOWN", 1: "GO_STRAIGHT", 2: "GO_LEFT", 3: "GO_RIGHT"}


class RejectedSample(ValueError):
    """An expected data-quality rejection, reported by reason in the summary."""

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


@dataclass
class FrameRef:
    scene_id: str
    frame_id: int
    token: str
    offset: int
    length: int
    front_image: str | None
    source_timestamp_us: int | None
    timestamp_us: int = 0
    timestamp_source: str = ""


def integer(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
    try:
        result = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if not isinstance(value, (str, int)) and value != result:
        raise ValueError(f"{name} must be an integer")
    return result


def validate_relative_image_path(path: str) -> str:
    """Reject anything that is not a plain relative POSIX path.

    Manifests must stay portable, so absolute paths, URL schemes and parent
    directory escapes are refused at build time rather than at training time.
    """
    if not isinstance(path, str) or not path:
        raise ValueError("Expected a nonempty relative image path")
    if "://" in path or path.startswith("//"):
        raise ValueError(f"Expected a local relative image path, got a URL: {path!r}")
    if path.startswith("/") or path.startswith("~"):
        raise ValueError(f"Expected a relative image path, got: {path!r}")
    parts = PurePosixPath(path).parts
    if not parts or ".." in parts:
        raise ValueError("Image paths must not contain '..'")
    return path


def image_uri(prefix: str, path: Any) -> str | None:
    if path is None:
        return None
    if not isinstance(path, str) or not path:
        raise ValueError("FRONT image path must be a nonempty string")
    relative = path if not prefix else prefix.strip("/") + "/" + path.lstrip("/")
    return validate_relative_image_path(relative)


def timestamp_from_record(record: dict) -> int | None:
    candidates = [
        record[k] for k in ("timestamp_micros", "timestamp_us", "anchor_timestamp_us")
        if record.get(k) is not None
    ]
    if not candidates:
        return None
    values = [integer(v, "timestamp") for v in candidates]
    if min(values) < 0 or len(set(values)) != 1:
        raise ValueError("Negative or conflicting source timestamps")
    return values[0]


def load_timestamp_overrides(path: Path | None) -> dict[tuple[str, int], int]:
    result = {}
    if path is None:
        return result
    with path.open("rb") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = decode_json(line)
            key = (str(row["scene_id"]), integer(row["frame_idx"], "frame_idx"))
            ts = timestamp_from_record(row)
            if ts is None or key in result:
                raise ValueError(f"{path}:{line_number}: missing timestamp or duplicate frame")
            result[key] = ts
    return result


def index_metadata(
    path: Path,
    image_prefix: str,
    split: str,
    camera_hz: float,
    time_mode: str,
    timestamp_overrides: dict[tuple[str, int], int] | None = None,
) -> tuple[dict[str, list[FrameRef]], dict[str, int]]:
    """Index ALL frames before applying output limits, including frames without GT."""
    scenes: dict[str, list[FrameRef]] = defaultdict(list)
    keys: set[tuple[str, int]] = set()
    tokens: set[str] = set()
    overrides = timestamp_overrides or {}
    counters: Counter = Counter()
    with path.open("rb") as handle:
        line_number = 0
        while True:
            offset = handle.tell()
            line = handle.readline()
            if not line:
                break
            line_number += 1
            if not line.strip():
                continue
            try:
                row = decode_json(line)
                scene = row["scene_id"]
                if not isinstance(scene, str) or not scene:
                    raise ValueError("scene_id must be a nonempty string")
                fid = integer(row["frame_idx"], "frame_idx")
                if fid < 0:
                    raise ValueError("frame_idx must be nonnegative")
                key = (scene, fid)
                token = row.get("context_name") or f"{scene}-{fid:03d}"
                if not isinstance(token, str):
                    raise ValueError("context_name must be a string")
                if key in keys or token in tokens:
                    raise ValueError("Duplicate scene/frame identity or token")
                if row.get("split", split) != split:
                    raise ValueError("Metadata split differs from --split")
                ts = overrides[key] if key in overrides else timestamp_from_record(row)
                ref = FrameRef(
                    scene, fid, token, offset, len(line),
                    image_uri(image_prefix, (row.get("image_paths") or {}).get("FRONT")),
                    ts,
                    timestamp_source="timestamps_jsonl" if key in overrides else "metadata",
                )
            except Exception as exc:
                # Do not echo arbitrary source lines or signed URLs into error logs.
                raise ValueError(f"{path}:{line_number}: invalid frame metadata ({type(exc).__name__})") from None
            keys.add(key)
            tokens.add(token)
            scenes[scene].append(ref)
            counters["metadata_records"] += 1
    if not scenes:
        raise ValueError("Metadata contains no frames")
    unused = set(overrides) - keys
    if unused:
        raise ValueError(f"Timestamp overrides contain {len(unused)} unknown scene/frame keys")
    for scene, frames in scenes.items():
        source_times = [f.source_timestamp_us for f in frames]
        usable = any(ts is not None and ts > 0 for ts in source_times)
        use_source = time_mode == "timestamp" or (time_mode == "auto" and usable)
        if use_source:
            if any(ts is None for ts in source_times):
                raise ValueError(f"{scene}: incomplete timestamps; do not mix source and inferred times")
            if len(frames) > 1 and len(set(source_times)) != len(frames):
                raise ValueError(f"{scene}: duplicate/zero-only source timestamps")
            for frame in frames:
                frame.timestamp_us = int(frame.source_timestamp_us)
        else:
            for frame in frames:
                frame.timestamp_us = round(frame.frame_id * 1_000_000 / camera_hz)
                frame.timestamp_source = "frame_index_10hz" if camera_hz == 10 else "frame_index_camera_hz"
        frames.sort(key=lambda f: (f.timestamp_us, f.frame_id))
        if any(b.timestamp_us <= a.timestamp_us for a, b in zip(frames, frames[1:])):
            raise ValueError(f"{scene}: frame timeline is not strictly increasing")
        counters["scenes_source_time" if use_source else "scenes_frame_index_time"] += 1
    return dict(scenes), dict(counters)


def vector_pairs(container: dict, x_key: str, y_key: str, size: int) -> list[list[float]]:
    x, y = container.get(x_key), container.get(y_key)
    if not isinstance(x, list) or not isinstance(y, list) or len(x) != size or len(y) != size:
        raise RejectedSample(f"invalid_{x_key}_{y_key}_shape")
    try:
        pairs = [[float(a), float(b)] for a, b in zip(x, y)]
    except (TypeError, ValueError, OverflowError):
        raise RejectedSample(f"invalid_{x_key}_{y_key}_values") from None
    if any(not math.isfinite(v) for point in pairs for v in point):
        raise RejectedSample(f"nonfinite_{x_key}_{y_key}")
    return pairs


def ego_state(record: dict) -> tuple[list[float], list[float]]:
    past = record.get("past_trajectory")
    if not isinstance(past, dict):
        raise RejectedSample("missing_past_trajectory")
    xy = vector_pairs(past, "pos_x", "pos_y", RAW_HISTORY_POINTS)
    velocity = vector_pairs(past, "vel_x", "vel_y", RAW_HISTORY_POINTS)
    acceleration = vector_pairs(past, "accel_x", "accel_y", RAW_HISTORY_POINTS)
    if any(abs(v) > 1e-3 for v in xy[-1]):
        raise RejectedSample("past_origin_not_current_ego")
    # Check interior knots: official current velocity and acceleration duplicate
    # the preceding values, so diff(v[-2:]) is NOT the current acceleration.
    for i in range(1, RAW_HISTORY_POINTS - 1):
        for axis in (0, 1):
            delta_v = velocity[i][axis] - velocity[i - 1][axis]
            if not math.isclose(acceleration[i][axis], delta_v, rel_tol=2e-4, abs_tol=2e-5):
                raise RejectedSample("acceleration_not_raw_delta_v")
    return velocity[-1], [v / RAW_HISTORY_DT_S for v in acceleration[-1]]


def command_info(record: dict) -> tuple[str, int, list[int], str]:
    raw = record.get("intent")
    raw_id = record.get("intent_id")
    if raw_id is not None:
        try:
            from_id = RAW_INTENTS[integer(raw_id, "intent_id")]
        except (KeyError, ValueError):
            raise RejectedSample("invalid_intent_id") from None
        if raw is None:
            raw = from_id
        elif raw != from_id:
            raise RejectedSample("intent_id_mismatch")
    raw = "UNKNOWN" if raw is None else raw
    if not isinstance(raw, str) or raw not in COMMANDS:
        raise RejectedSample("invalid_intent")
    nav, onehot, canonical = COMMANDS[raw]
    return raw, nav, list(onehot), canonical


def nearest_frame(frames: list[FrameRef], times: list[int], target_us: int, tolerance_us: int) -> FrameRef | None:
    i = bisect.bisect_left(times, target_us)
    candidates = [frames[j] for j in (i - 1, i) if 0 <= j < len(frames)]
    if not candidates:
        return None
    best = min(candidates, key=lambda f: (abs(f.timestamp_us - target_us), f.timestamp_us, f.frame_id))
    return best if abs(best.timestamp_us - target_us) <= tolerance_us else None


def select_video_frames(
    anchor: FrameRef, frames: list[FrameRef], times: list[int], tolerance_us: int,
    require_future_images: bool = True,
) -> list[FrameRef]:
    if not anchor.front_image or anchor.front_image not in {f.front_image for f in frames}:
        raise RejectedSample("current_image_missing")
    selected = [anchor]
    for seconds in VIDEO_OFFSETS_S[1:] if require_future_images else ():
        match = nearest_frame(frames, times, anchor.timestamp_us + round(seconds * 1_000_000), tolerance_us)
        if match is None:
            raise RejectedSample(f"future_image_missing_{int(seconds)}s")
        selected.append(match)
    if len({f.front_image for f in selected}) != len(selected):
        raise RejectedSample("repeated_video_image")
    if any(b.timestamp_us <= a.timestamp_us for a, b in zip(selected, selected[1:])):
        raise RejectedSample("nonincreasing_video_times")
    return selected


def preference_trajectories(record: dict) -> list[dict]:
    """Keep native reference XY and scores, including the source's current point."""
    source = record.get("preference_trajectories") or []
    if not isinstance(source, list):
        raise RejectedSample("invalid_preference_trajectories")
    result = []
    for candidate in source:
        if not isinstance(candidate, dict):
            raise RejectedSample("invalid_preference_trajectory")
        x = candidate.get("pos_x")
        if not isinstance(x, list) or not x:
            raise RejectedSample("invalid_preference_trajectory")
        try:
            xy = vector_pairs(candidate, "pos_x", "pos_y", len(x))
        except RejectedSample:
            raise RejectedSample("invalid_preference_trajectory") from None
        score = candidate.get("preference_score")
        if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score) or not 0 <= score <= 10:
            raise RejectedSample("invalid_preference_score")
        result.append({
            "pos_x": [p[0] for p in xy],
            "pos_y": [p[1] for p in xy],
            "preference_score": float(score),
        })
    return result


def build_record(
    record: dict, anchor: FrameRef, selected: list[FrameRef], horizon_s: int,
    preference_only: bool = False,
) -> dict:
    if len(selected) not in (1, len(VIDEO_OFFSETS_S)):
        raise ValueError("Expected either one current image or a full five-image training window")
    references = preference_trajectories(record)
    if preference_only and not references:
        raise RejectedSample("missing_preference_trajectories")
    future = record.get("future_trajectory")
    if not isinstance(future, dict):
        raise RejectedSample("missing_future_trajectory")
    xy = vector_pairs(future, "pos_x", "pos_y", RAW_FUTURE_POINTS)
    count = horizon_s * TRAJECTORY_HZ
    mask = future.get("valid_mask")
    if mask is not None and (len(mask) < count or not all(mask[:count])):
        raise RejectedSample("invalid_future_mask")
    velocity, acceleration = ego_state(record)
    raw_intent, nav, command, canonical = command_info(record)
    result = {
        "schema_version": "simwam_waymo_xy_v1",
        "dataset": "waymo_wod_e2e",
        "split": record.get("split", "training"),
        "clip_id": anchor.scene_id,
        "token": anchor.token,
        "current_frame_id": anchor.frame_id,
        "anchor_timestamp_us": anchor.timestamp_us,
        "timestamp_source": anchor.timestamp_source,
        "timestamp_reference": "scene_relative" if anchor.timestamp_source.startswith("frame_index") else "source_time",
        "intent": canonical,
        "source_intent": raw_intent,
        "nav_command": nav,
        "driving_command": command,
        "ego_velocity": velocity,
        "ego_acceleration": acceleration,
        "ego_status": {
            "ego_pose": [0.0, 0.0, 0.0],
            "ego_velocity": velocity,
            "ego_acceleration": acceleration,
            "driving_command": command,
            "in_global_frame": False,
        },
        "coordinate_frame": COORDINATE_FRAME,
        "acceleration_correction": "raw_delta_v_divided_by_0.25s",
        "front_image": selected[0].front_image,
        "future_front_images_1s_to_4s": [f.front_image for f in selected[1:]],
        "video_frame_timestamps_us": [f.timestamp_us for f in selected],
        "video_frame_offsets_s": [(f.timestamp_us - anchor.timestamp_us) / 1_000_000 for f in selected],
        "video_target_offsets_s": list(VIDEO_OFFSETS_S) if len(selected) > 1 else [0.0],
        "video_frame_mode": "current_plus_future" if len(selected) > 1 else "current_only",
        "future_traj_4hz": xy[:count],
        "future_times_4hz": [(i + 1) / TRAJECTORY_HZ for i in range(count)],
        "future_valid_mask_4hz": [1] * count,
        "trajectory_horizon_s": horizon_s,
        "trajectory_hz": TRAJECTORY_HZ,
        "action_dim": 2,
        "has_heading": False,
        "trajectory_mode": "absolute",
    }
    if references:
        result["preference_trajectories"] = references
        result["num_preference_trajectories"] = len(references)
    return result


class LocalImages:
    """List each FRONT directory once per scene; optionally decode selected JPGs."""

    def __init__(self, mode: str, image_root: str | Path):
        self.mode = mode
        self.image_root = Path(image_root)
        self._listings: dict[str, set[str]] = {}

    def resolve(self, relative_path: str) -> Path:
        return self.image_root / relative_path

    def available(self, frames: list[FrameRef]) -> set[str]:
        wanted = {f.front_image for f in frames if f.front_image}
        if self.mode == "none":
            return wanted
        groups: dict[str, set[str]] = defaultdict(set)
        for relative in wanted:
            directory, _, name = relative.rpartition("/")
            groups[directory + "/"].add(name)
        result = set()
        for directory, names in sorted(groups.items()):
            listing = self._listings.get(directory)
            if listing is None:
                path = self.image_root / directory
                listing = self._listings[directory] = (
                    {entry.name for entry in path.iterdir() if entry.is_file()}
                    if path.is_dir() else set()
                )
            for name in sorted(names & listing):
                result.add(directory + name)
        return result

    def decodable(self, relative_path: str) -> bool:
        if self.mode != "decode":
            return True
        from PIL import Image, UnidentifiedImageError

        try:
            with Image.open(self.resolve(relative_path)) as image:
                image.load()
                return image.width > 0 and image.height > 0
        except (UnidentifiedImageError, OSError, ValueError):
            return False


def process_scene(
    metadata: Path, frames: list[FrameRef], images: LocalImages,
    horizon_s: int, tolerance_us: int, limit: int | None,
    anchor_tokens: set[str] | None = None,
    preference_only: bool = False,
    require_future_images: bool = True,
) -> tuple[list[dict], Counter]:
    available = images.available(frames)
    visible = [f for f in frames if f.front_image in available]
    times = [f.timestamp_us for f in visible]
    checked: dict[str, bool] = {}
    output = []
    counts: Counter = Counter(scenes_processed=1)
    with metadata.open("rb") as handle:
        for anchor in frames:
            if anchor_tokens is not None and anchor.token not in anchor_tokens:
                continue
            counts["anchors_examined"] += 1
            try:
                raw = None
                if preference_only:
                    handle.seek(anchor.offset)
                    raw = decode_json(handle.read(anchor.length))
                    if not raw.get("preference_trajectories"):
                        raise RejectedSample("missing_preference_trajectories")
                selected = select_video_frames(anchor, visible, times, tolerance_us, require_future_images)
                if raw is None:
                    handle.seek(anchor.offset)
                    raw = decode_json(handle.read(anchor.length))
                record = build_record(raw, anchor, selected, horizon_s, preference_only)
                for image in selected:
                    if image.front_image not in checked:
                        checked[image.front_image] = images.decodable(image.front_image)
                    if not checked[image.front_image]:
                        raise RejectedSample("selected_image_unreadable")
            except RejectedSample as exc:
                counts[f"rejected_{exc.reason}"] += 1
                continue
            output.append(record)
            counts["samples_built"] += 1
            if record.get("preference_trajectories"):
                counts["samples_with_preference_trajectories"] += 1
            if limit is not None and len(output) >= limit:
                break
    return output, counts


def ordered_bounded_map(pool: ThreadPoolExecutor, function: Callable, items: list, window: int) -> Iterator:
    """Bound queued results so a slow scene cannot retain the entire dataset."""
    pending = deque()
    iterator = iter(items)
    for _ in range(window):
        try:
            pending.append(pool.submit(function, next(iterator)))
        except StopIteration:
            break
    try:
        while pending:
            result = pending.popleft().result()
            yield result
            try:
                pending.append(pool.submit(function, next(iterator)))
            except StopIteration:
                pass
    finally:
        for future in pending:
            future.cancel()


def atomic_json(path: Path, payload: dict):
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        try:
            json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
            handle.close()
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)


def relative_to_project(path: Path) -> str:
    """Report paths relative to the repository root so summaries stay portable."""
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def run(args: argparse.Namespace) -> dict:
    metadata = Path(args.metadata or DEFAULT_METADATA_DIR / f"{args.split}.jsonl").resolve()
    preference_suffix = "_pref_only" if args.preference_only else ""
    video_tag = "video4s" if args.require_future_images else "current"
    output = Path(args.output or PROJECT_ROOT / "data" / (
        f"waymo_{args.split}_front_{video_tag}_traj{args.trajectory_horizon_s}s_4hz_xy{preference_suffix}.jsonl"
    )).resolve()
    summary_path = output.with_suffix(".summary.json")
    if output == metadata or summary_path == metadata or output == summary_path:
        raise ValueError("Output and summary must differ from the source metadata and each other")
    if args.timestamps_jsonl and Path(args.timestamps_jsonl).resolve() in (output, summary_path):
        raise ValueError("Output must differ from the source timestamp index")
    if args.tokens_file and Path(args.tokens_file).resolve() in (output, summary_path):
        raise ValueError("Output must differ from the source token selection")
    if any(p.exists() for p in (output, summary_path)) and not args.force:
        raise FileExistsError("Output or summary already exists; choose a new --output or pass --force")
    validate_relative_image_path(args.image_prefix)
    image_root = Path(args.image_root)
    if args.image_check != "none" and not image_root.is_dir():
        raise ValueError(f"--image-root does not exist: {image_root}")
    source_stat = metadata.stat()
    fingerprint = (source_stat.st_size, source_stat.st_mtime_ns)
    start = time.monotonic()
    overrides = load_timestamp_overrides(Path(args.timestamps_jsonl) if args.timestamps_jsonl else None)
    scenes, index_counts = index_metadata(
        metadata, args.image_prefix, args.split, args.camera_hz, args.time_mode, overrides
    )
    print(f"Indexed {index_counts['metadata_records']} frames in {len(scenes)} scenes.", flush=True)
    print(f"Timeline: {index_counts}.", flush=True)
    scene_ids = sorted(scenes)
    requested_tokens = None
    if args.tokens_file:
        requested_tokens = {
            line.strip() for line in Path(args.tokens_file).read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
        if not requested_tokens:
            raise ValueError("--tokens-file contains no tokens")
        known_tokens = {frame.token for frames in scenes.values() for frame in frames}
        missing_tokens = requested_tokens - known_tokens
        if missing_tokens:
            raise ValueError(f"--tokens-file contains {len(missing_tokens)} tokens absent from metadata")
        scene_ids = [
            scene_id for scene_id in scene_ids
            if any(frame.token in requested_tokens for frame in scenes[scene_id])
        ]
    if args.max_scenes is not None:
        scene_ids = scene_ids[:args.max_scenes]
    images = LocalImages(args.image_check, image_root)
    counts: Counter = Counter()
    commands: Counter = Counter()
    time_sources: Counter = Counter()
    written = 0
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=output.parent, delete=False) as handle:
        temporary = Path(handle.name)
        try:
            def process(scene_id):
                return process_scene(
                    metadata, scenes[scene_id], images, args.trajectory_horizon_s,
                    round(args.frame_tolerance_ms * 1000), args.limit,
                    requested_tokens,
                    args.preference_only,
                    args.require_future_images,
                )
            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                results = ordered_bounded_map(pool, process, scene_ids, args.workers * 2)
                try:
                    for records, scene_counts in results:
                        counts.update(scene_counts)
                        for record in records:
                            if args.limit is not None and written >= args.limit:
                                break
                            handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
                            written += 1
                            commands[record["intent"]] += 1
                            time_sources[record["timestamp_source"]] += 1
                        if counts["scenes_processed"] % 100 == 0:
                            print(f"Scenes {counts['scenes_processed']}/{len(scene_ids)}; samples {written}.", flush=True)
                        if args.limit is not None and written >= args.limit:
                            break
                finally:
                    results.close()
            now = metadata.stat()
            if (now.st_size, now.st_mtime_ns) != fingerprint:
                raise RuntimeError("Source metadata changed during preprocessing; output was not published")
            if not written:
                raise RuntimeError(f"No samples survived: {dict(counts)}")
            handle.close()
            os.replace(temporary, output)
        finally:
            temporary.unlink(missing_ok=True)
    summary = {
        "schema_version": "simwam_waymo_xy_v1",
        "input": relative_to_project(metadata), "output": relative_to_project(output),
        "source_size_bytes": fingerprint[0], "source_mtime_ns": fingerprint[1],
        "num_samples": written, "index": index_counts, "processing": dict(counts),
        "command_counts": dict(commands), "timestamp_sources": dict(time_sources),
        "trajectory_horizon_s": args.trajectory_horizon_s, "trajectory_hz": TRAJECTORY_HZ,
        "action_dim": 2, "has_heading": False, "coordinate_frame": COORDINATE_FRAME,
        "video_target_offsets_s": list(VIDEO_OFFSETS_S) if args.require_future_images else [0.0],
        "require_future_images": args.require_future_images,
        "raw_acceleration_scale": 1 / RAW_HISTORY_DT_S,
        "raw_acceleration_delta_v_check": True,
        "camera_hz": args.camera_hz, "time_mode": args.time_mode,
        "timestamps_jsonl": args.timestamps_jsonl, "frame_tolerance_ms": args.frame_tolerance_ms,
        "image_prefix": args.image_prefix, "image_check": args.image_check,
        "image_root": relative_to_project(image_root),
        "image_path_convention": "manifest paths are relative to image_root",
        "limit": args.limit, "max_scenes": args.max_scenes,
        "tokens_file": args.tokens_file,
        "requested_token_count": len(requested_tokens) if requested_tokens is not None else None,
        "preference_only": args.preference_only,
        "elapsed_seconds": round(time.monotonic() - start, 2),
        "loader_note": "Load with simwam.datasets.waymo.waymo_dataset.WaymoVideoDataset (20x2 XY actions).",
    }
    atomic_json(summary_path, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", choices=("training", "val"), default="training")
    parser.add_argument("--metadata", help="Native extracted metadata JSONL; defaults to <metadata-dir>/<split>.jsonl")
    parser.add_argument("--output", help="New output JSONL; default filename includes split and trajectory horizon")
    parser.add_argument("--trajectory-horizon-s", type=int, choices=(4, 5), default=5)
    parser.add_argument(
        "--preference-only", action=argparse.BooleanOptionalAction, default=None,
        help="Require scored reference trajectories (default: enabled for val, disabled for training)",
    )
    parser.add_argument(
        "--require-future-images", action=argparse.BooleanOptionalAction, default=None,
        help="Require the +1..+4 s image window (default: training only; val uses current image only)",
    )
    parser.add_argument("--image-prefix", default=DEFAULT_IMAGE_PREFIX,
                        help="Relative prefix prepended to each metadata FRONT path")
    parser.add_argument("--image-root", default=str(DEFAULT_IMAGE_ROOT),
                        help="Directory the manifest image paths are relative to")
    parser.add_argument("--image-check", choices=("list", "decode", "none"), default="list",
                        help="list: verify files exist under --image-root; decode: also open/decode selected images; none: metadata only")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--time-mode", choices=("auto", "timestamp", "frame_index"), default="auto")
    parser.add_argument("--timestamps-jsonl", help="Optional per-frame {scene_id,frame_idx,timestamp_us} JSONL")
    parser.add_argument("--camera-hz", type=float, default=10.0,
                        help="Nominal image cadence ONLY when using frame-index scene-relative time")
    parser.add_argument("--frame-tolerance-ms", type=float, default=50.0)
    parser.add_argument("--limit", type=int, help="Maximum written samples; the full image timeline is still indexed")
    parser.add_argument("--max-scenes", type=int, help="Process first N scenes after full metadata indexing")
    parser.add_argument("--tokens-file", help="Only emit anchors from this one-token-per-line file; retain all frames for time matching")
    parser.add_argument("--force", action="store_true", help="Replace output/summary, never the inputs")
    args = parser.parse_args(argv)
    if args.preference_only is None:
        args.preference_only = args.split == "val"
    if args.require_future_images is None:
        args.require_future_images = args.split == "training"
    for name in ("workers", "camera_hz"):
        value = getattr(args, name)
        if not math.isfinite(value) or value <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    for name in ("limit", "max_scenes"):
        value = getattr(args, name)
        if value is not None and value <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if not math.isfinite(args.frame_tolerance_ms) or not 0 <= args.frame_tolerance_ms < 500:
        parser.error("--frame-tolerance-ms must be in [0,500)")
    return args


def main():
    try:
        run(parse_args())
    except (OSError, ValueError, RuntimeError) as exc:
        raise SystemExit(str(exc)) from None


if __name__ == "__main__":
    main()
