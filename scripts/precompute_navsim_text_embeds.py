import hashlib
import logging
import multiprocessing as mp
import os
import re
import uuid
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import hydra
import torch
import torch.distributed as dist
from hydra.utils import instantiate
from omegaconf import DictConfig, ListConfig, OmegaConf
from tqdm import tqdm

from simwam.datasets.navsim.navsim_dataset import NavSimVideoDataset
from simwam.models.wan22.helpers.loader import _load_registered_model, _resolve_configs
from simwam.models.wan22.wan_video_text_encoder import HuggingfaceTokenizer
from simwam.utils.config_resolvers import register_default_resolvers
from simwam.utils.logging_config import get_logger, setup_logging

register_default_resolvers()
logger = get_logger(__name__)

DEFAULT_MODEL_ID = "Wan-AI/Wan2.2-TI2V-5B"
DEFAULT_TOKENIZER_MODEL_ID = "Wan-AI/Wan2.1-T2V-1.3B"
DEFAULT_BATCH_SIZE = 16
DEFAULT_PROMPT_COLLECT_WORKERS = max(1, os.cpu_count() or 1)
NAVSIM_DATASET_TARGET = "simwam.datasets.navsim.navsim_dataset.NavSimVideoDataset"

_PROMPT_DATASET: NavSimVideoDataset | None = None


def _init_distributed():
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 1:
        return False, 0, 1, 0

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    if not dist.is_initialized():
        dist.init_process_group(backend=backend, init_method="env://")

    return True, dist.get_rank(), dist.get_world_size(), local_rank


def _to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"1", "true", "yes", "y"}:
            return True
        if text in {"0", "false", "no", "n"}:
            return False
    raise ValueError(f"Cannot parse bool value: {value}")


def _iter_navsim_nodes(node: Any, path: str = "data"):
    if isinstance(node, DictConfig):
        target = node.get("_target_")
        if target == NAVSIM_DATASET_TARGET:
            yield path, node
        for key, value in node.items():
            yield from _iter_navsim_nodes(value, f"{path}.{key}")
    elif isinstance(node, ListConfig):
        for idx, value in enumerate(node):
            yield from _iter_navsim_nodes(value, f"{path}[{idx}]")


def _model_id_to_enc_id(model_id: str) -> str:
    base = str(model_id).split("/")[-1]
    enc_id = re.sub(r"[^a-z0-9]+", "", base.lower())
    return enc_id or "textenc"


def _atomic_torch_save(payload: dict[str, torch.Tensor], output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.parent / f".{output_path.name}.tmp.{uuid.uuid4().hex}"
    torch.save(payload, str(tmp_path))
    os.replace(tmp_path, output_path)


def _init_prompt_worker(dataset_cfg: dict[str, Any]):
    global _PROMPT_DATASET
    dataset = instantiate(OmegaConf.create(dataset_cfg))
    if not isinstance(dataset, NavSimVideoDataset):
        raise TypeError(f"Expected NavSimVideoDataset, got {type(dataset)}")
    _PROMPT_DATASET = dataset


def _build_dynamic_prompt_from_token(token: str) -> str:
    if _PROMPT_DATASET is None:
        raise RuntimeError("Prompt worker dataset is not initialized.")

    scene = _PROMPT_DATASET.scene_loader.get_scene_from_token(token)
    _, high_cmd_one_hot, speed_mps, acc_mps2 = _PROMPT_DATASET._extract_ego_features(scene)
    hist_xyh = torch.tensor(
        scene.get_history_trajectory(
            num_trajectory_frames=_PROMPT_DATASET.scene_filter.num_history_frames
        ).poses,
        dtype=torch.float32,
    )
    return _PROMPT_DATASET.build_prompt_fixed(
        hist_xyh=hist_xyh,
        high_cmd_one_hot=high_cmd_one_hot,
        speed_mps=speed_mps,
        acc_mps2=acc_mps2,
        use_dynamic_prompt=_PROMPT_DATASET.use_dynamic_prompt,
    )


def _build_dynamic_prompt(dataset: NavSimVideoDataset, token: str) -> str:
    scene = dataset.scene_loader.get_scene_from_token(token)
    _, high_cmd_one_hot, speed_mps, acc_mps2 = dataset._extract_ego_features(scene)
    hist_xyh = torch.tensor(
        scene.get_history_trajectory(
            num_trajectory_frames=dataset.scene_filter.num_history_frames
        ).poses,
        dtype=torch.float32,
    )
    return dataset.build_prompt_fixed(
        hist_xyh=hist_xyh,
        high_cmd_one_hot=high_cmd_one_hot,
        speed_mps=speed_mps,
        acc_mps2=acc_mps2,
        use_dynamic_prompt=dataset.use_dynamic_prompt,
    )


def _collect_dynamic_prompts(
    dataset: NavSimVideoDataset,
    dataset_cfg: dict[str, Any],
    num_workers: int,
) -> list[str]:
    global _PROMPT_DATASET

    if not dataset.tokens:
        return []

    worker_count = min(max(1, num_workers), len(dataset.tokens))
    if worker_count == 1:
        prompts = []
        for token in tqdm(
            dataset.tokens,
            desc=f"Collecting navsim prompts len={dataset.context_len}",
            unit="prompt",
            dynamic_ncols=True,
        ):
            prompts.append(_build_dynamic_prompt(dataset, token))
        return prompts

    chunksize = max(1, len(dataset.tokens) // (worker_count * 8))
    start_method = "fork" if os.name == "posix" else "spawn"
    ctx = mp.get_context(start_method)
    if start_method == "fork":
        _PROMPT_DATASET = dataset

    with ProcessPoolExecutor(
        max_workers=worker_count,
        mp_context=ctx,
        initializer=None if start_method == "fork" else _init_prompt_worker,
        initargs=() if start_method == "fork" else (dataset_cfg,),
    ) as executor:
        return list(
            tqdm(
                executor.map(_build_dynamic_prompt_from_token, dataset.tokens, chunksize=chunksize),
                total=len(dataset.tokens),
                desc=f"Collecting navsim prompts len={dataset.context_len}",
                unit="prompt",
                dynamic_ncols=True,
            )
        )


def _collect_prompt_groups(
    data_cfg: DictConfig,
    *,
    prompt_collect_workers: int = DEFAULT_PROMPT_COLLECT_WORKERS,
) -> dict[int, dict[str, set[Path]]]:
    prompt_groups: dict[int, dict[str, set[Path]]] = defaultdict(lambda: defaultdict(set))

    for node_path, node in _iter_navsim_nodes(data_cfg):
        dataset = instantiate(node)
        if not isinstance(dataset, NavSimVideoDataset):
            raise TypeError(f"Expected NavSimVideoDataset for `{node_path}`, got {type(dataset)}")
        if dataset.text_embedding_cache_dir is None:
            raise ValueError(f"Missing `text_embedding_cache_dir` for `{node_path}`.")

        cache_dir = Path(dataset.text_embedding_cache_dir).expanduser()
        prompt_count_before_dedup = 0
        if not dataset.use_dynamic_prompt:
            hist_xyh = torch.tensor(
                torch.zeros((dataset.scene_filter.num_history_frames, 3), dtype=torch.float32),
                dtype=torch.float32,
            )
            prompt = dataset.build_prompt_fixed(
                hist_xyh=hist_xyh,
                high_cmd_one_hot=torch.zeros(4, dtype=torch.float32),
                speed_mps=0.0,
                acc_mps2=0.0,
                use_dynamic_prompt=dataset.use_dynamic_prompt,
            )
            prompt_groups[dataset.context_len][prompt].add(cache_dir)
            prompt_count_before_dedup = 1
        else:
            dataset_cfg = OmegaConf.to_container(node, resolve=True)
            prompts = _collect_dynamic_prompts(dataset, dataset_cfg, prompt_collect_workers)
            for prompt in prompts:
                prompt_groups[dataset.context_len][prompt].add(cache_dir)
                prompt_count_before_dedup += 1

        logger.info(
            "Scanned `%s`: samples=%d cache_dir=%s context_len=%d dynamic_prompt=%s prompts(before dedup)=%d",
            node_path,
            len(dataset.tokens),
            cache_dir,
            dataset.context_len,
            dataset.use_dynamic_prompt,
            prompt_count_before_dedup,
        )

    return prompt_groups


def _encode_group(
    *,
    prompts_to_dirs: dict[str, set[Path]],
    context_len: int,
    enc_id: str,
    tokenizer: HuggingfaceTokenizer,
    text_encoder,
    device: str,
    overwrite: bool,
    rank: int,
    world_size: int,
    is_distributed: bool,
) -> None:
    prompts = sorted(prompts_to_dirs.keys())
    cache_dirs = sorted({cache_dir for dirs in prompts_to_dirs.values() for cache_dir in dirs})
    stats = {str(cache_dir): {"new": 0, "overwrite": 0, "skip": 0} for cache_dir in cache_dirs}

    prompts = prompts[rank::world_size] if is_distributed else prompts
    if not overwrite:
        filtered_prompts = []
        for prompt in prompts:
            hashed = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
            filename = f"{hashed}.t5_len{context_len}.{enc_id}.pt"
            target_dirs = prompts_to_dirs[prompt]
            fully_cached = True
            for cache_dir in target_dirs:
                if not (cache_dir / filename).exists():
                    fully_cached = False
                    break
            if fully_cached:
                for cache_dir in target_dirs:
                    stats[str(cache_dir)]["skip"] += 1
            else:
                filtered_prompts.append(prompt)
        prompts = filtered_prompts

    over_length_local = 0
    with tqdm(
        total=len(prompts),
        desc=f"Encoding navsim prompts len={context_len} (rank {rank}/{world_size})"
        if is_distributed
        else f"Encoding navsim prompts len={context_len}",
        unit="prompt",
        dynamic_ncols=True,
        disable=is_distributed and rank != 0,
    ) as pbar:
        with torch.no_grad():
            for start in range(0, len(prompts), DEFAULT_BATCH_SIZE):
                batch_prompts = prompts[start : start + DEFAULT_BATCH_SIZE]
                ids, mask = tokenizer(batch_prompts, return_mask=True, add_special_tokens=True)
                ids = ids.to(device)
                mask = mask.to(device=device, dtype=torch.bool)
                over_length_local += int(mask.all(dim=1).sum().item())
                context = text_encoder(ids, mask)

                for i, prompt in enumerate(batch_prompts):
                    hashed = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
                    payload = {
                        "context": context[i].detach().to(device="cpu", dtype=torch.bfloat16).contiguous(),
                        "mask": mask[i].detach().to(device="cpu", dtype=torch.bool).contiguous(),
                    }
                    for cache_dir in prompts_to_dirs[prompt]:
                        cache_path = cache_dir / f"{hashed}.t5_len{context_len}.{enc_id}.pt"
                        key = str(cache_dir)
                        if cache_path.exists() and not overwrite:
                            stats[key]["skip"] += 1
                            continue
                        if cache_path.exists():
                            stats[key]["overwrite"] += 1
                        else:
                            stats[key]["new"] += 1
                        _atomic_torch_save(payload, cache_path)

                pbar.update(len(batch_prompts))

    if is_distributed:
        reduce_device = torch.device(device) if device.startswith("cuda") else torch.device("cpu")
        over_tensor = torch.tensor([over_length_local], device=reduce_device, dtype=torch.long)
        dist.all_reduce(over_tensor, op=dist.ReduceOp.SUM)
        over_length = int(over_tensor.item())

        counts_tensor = torch.tensor(
            [[stats[str(cache_dir)]["new"], stats[str(cache_dir)]["overwrite"], stats[str(cache_dir)]["skip"]] for cache_dir in cache_dirs],
            device=reduce_device,
            dtype=torch.long,
        )
        dist.all_reduce(counts_tensor, op=dist.ReduceOp.SUM)
        if rank == 0:
            for idx, cache_dir in enumerate(cache_dirs):
                key = str(cache_dir)
                stats[key]["new"] = int(counts_tensor[idx, 0].item())
                stats[key]["overwrite"] = int(counts_tensor[idx, 1].item())
                stats[key]["skip"] = int(counts_tensor[idx, 2].item())
    else:
        over_length = over_length_local

    if (not is_distributed) or rank == 0:
        logger.info(
            "Finished context_len=%d: unique_prompts=%d over_length=%d",
            context_len,
            len(prompts_to_dirs),
            over_length,
        )
        for cache_dir in cache_dirs:
            key = str(cache_dir)
            logger.info(
                "Cache dir: %s | new=%d overwrite=%d skip=%d",
                key,
                stats[key]["new"],
                stats[key]["overwrite"],
                stats[key]["skip"],
            )


@hydra.main(config_path="../configs", config_name="train", version_base="1.3")
def main(cfg: DictConfig):
    setup_logging(log_level=logging.INFO)

    is_distributed, rank, world_size, local_rank = _init_distributed()
    if is_distributed and rank == 0:
        logger.info("Distributed enabled: world_size=%d", world_size)

    if cfg.data is None:
        raise ValueError("`cfg.data` is required.")
    if cfg.model is None:
        raise ValueError("`cfg.model` is required.")

    overwrite = _to_bool(cfg.get("overwrite", True))
    prompt_collect_workers = int(cfg.get("prompt_collect_workers", DEFAULT_PROMPT_COLLECT_WORKERS))
    if prompt_collect_workers < 1:
        raise ValueError(f"`prompt_collect_workers` must be >= 1, got {prompt_collect_workers}")

    prompt_groups = _collect_prompt_groups(
        cfg.data,
        prompt_collect_workers=prompt_collect_workers,
    )
    if not prompt_groups:
        raise ValueError("No NavSim dataset nodes found under `cfg.data`.")

    if torch.cuda.is_available():
        device = f"cuda:{local_rank}" if is_distributed else "cuda"
    else:
        device = "cpu"

    model_id = str(cfg.model.get("model_id", DEFAULT_MODEL_ID))
    tokenizer_model_id = str(cfg.model.get("tokenizer_model_id", DEFAULT_TOKENIZER_MODEL_ID))
    redirect_common_files = bool(cfg.model.get("redirect_common_files", True))
    enc_id = _model_id_to_enc_id(model_id)

    logger.info(
        "Preparing NavSim text encoder with model_id=%s tokenizer_model_id=%s device=%s overwrite=%s prompt_collect_workers=%d",
        model_id,
        tokenizer_model_id,
        device,
        overwrite,
        prompt_collect_workers,
    )

    _, text_config, _, tokenizer_config = _resolve_configs(
        model_id=model_id,
        tokenizer_model_id=tokenizer_model_id,
        redirect_common_files=redirect_common_files,
    )
    text_config.download_if_necessary()
    tokenizer_config.download_if_necessary()

    text_encoder = _load_registered_model(
        text_config.path,
        "wan_video_text_encoder",
        torch_dtype=torch.bfloat16,
        device=device,
    ).eval()

    for context_len in sorted(prompt_groups):
        tokenizer = HuggingfaceTokenizer(
            name=tokenizer_config.path,
            seq_len=context_len,
            clean="whitespace",
        )
        if (not is_distributed) or rank == 0:
            cache_dir_count = len(
                {cache_dir for dirs in prompt_groups[context_len].values() for cache_dir in dirs}
            )
            logger.info(
                "Encoding NavSim prompt group: context_len=%d unique_prompts=%d cache_dirs=%d",
                context_len,
                len(prompt_groups[context_len]),
                cache_dir_count,
            )

        _encode_group(
            prompts_to_dirs=prompt_groups[context_len],
            context_len=context_len,
            enc_id=enc_id,
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            device=device,
            overwrite=overwrite,
            rank=rank,
            world_size=world_size,
            is_distributed=is_distributed,
        )

    if is_distributed and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

