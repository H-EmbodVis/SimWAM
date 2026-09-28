"""Load the exact official protobuf descriptors without a generated-code version lock.

An installed protobuf runtime can be older than the supplied *_pb2.py files.
Reading their literal serialized descriptors keeps the official schema unchanged
and works with both runtimes, without upgrading the shared training environment.

Point ``WAYMO_OPEN_DATASET_SRC`` (or ``--waymo-src``) at the ``src/`` directory of
a https://github.com/waymo-research/waymo-open-dataset checkout.
"""
from __future__ import annotations

import ast
from functools import lru_cache
import hashlib
import importlib
import os
from pathlib import Path
from types import SimpleNamespace

from google.protobuf import descriptor_pb2, descriptor_pool, message_factory

DEFAULT_WAYMO_SRC = os.environ.get("WAYMO_OPEN_DATASET_SRC", "")
SUBMISSION_PROTO = "waymo_open_dataset/protos/end_to_end_driving_submission.proto"


def descriptor_bytes(path: Path) -> bytes:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr == "AddSerializedFile" and node.args:
                value = ast.literal_eval(node.args[0])
                if isinstance(value, bytes):
                    return value
            if node.func.attr == "FileDescriptor":
                for keyword in node.keywords:
                    if keyword.arg == "serialized_pb":
                        value = ast.literal_eval(keyword.value)
                        if isinstance(value, bytes):
                            return value
    raise ValueError(f"No official serialized protobuf descriptor in {path}")


@lru_cache(maxsize=4)
def load_submission_proto(waymo_src: str = DEFAULT_WAYMO_SRC):
    if not waymo_src:
        raise ValueError(
            "Set WAYMO_OPEN_DATASET_SRC (or pass --waymo-src) to the `src/` directory of a "
            "waymo-open-dataset checkout: https://github.com/waymo-research/waymo-open-dataset"
        )
    root = Path(waymo_src).expanduser().resolve()
    if not (root / SUBMISSION_PROTO).is_file():
        raise FileNotFoundError(
            f"Missing {root / SUBMISSION_PROTO}; check WAYMO_OPEN_DATASET_SRC points at `src/`."
        )
    pool = descriptor_pool.DescriptorPool()
    loaded, blobs = set(), {}

    def load(name):
        if name in loaded:
            return
        if name.startswith("google/protobuf/"):
            module = importlib.import_module(name[:-6].replace("/", ".") + "_pb2")
            blob = module.DESCRIPTOR.serialized_pb
        else:
            path = root / (name[:-6] + "_pb2.py")
            blob = descriptor_bytes(path)
        desc = descriptor_pb2.FileDescriptorProto.FromString(blob)
        if desc.name != name:
            raise ValueError(f"Descriptor file name mismatch: {name} / {desc.name}")
        for dependency in desc.dependency:
            load(dependency)
        pool.AddSerializedFile(blob)
        loaded.add(name)
        blobs[name] = blob

    load(SUBMISSION_PROTO)

    def message(name):
        descriptor = pool.FindMessageTypeByName("waymo.open_dataset." + name)
        if hasattr(message_factory, "GetMessageClass"):
            return message_factory.GetMessageClass(descriptor)
        return message_factory.MessageFactory(pool).GetPrototype(descriptor)

    submission = message("E2EDChallengeSubmission")
    return SimpleNamespace(
        TrajectoryPrediction=message("TrajectoryPrediction"),
        FrameTrajectoryPredictions=message("FrameTrajectoryPredictions"),
        E2EDChallengeSubmission=submission,
        E2ED_SUBMISSION=submission.DESCRIPTOR.enum_types_by_name["SubmissionType"].values_by_name["E2ED_SUBMISSION"].number,
        descriptor_sha256=hashlib.sha256(blobs[SUBMISSION_PROTO]).hexdigest(),
        source_proto=str(root / SUBMISSION_PROTO),
        source_proto_sha256=hashlib.sha256((root / SUBMISSION_PROTO).read_bytes()).hexdigest(),
    )
