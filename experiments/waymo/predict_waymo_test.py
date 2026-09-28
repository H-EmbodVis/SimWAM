#!/usr/bin/env python3
"""Run action-only inference on all official Waymo test frames."""
from datetime import timedelta
import json
import logging
import math
import os
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from accelerate import Accelerator
from accelerate.utils import InitProcessGroupKwargs, broadcast_object_list, set_seed
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
import torch

from simwam.datasets.waymo.checkpoint import load_waymo_checkpoint
from simwam.datasets.waymo.prediction import predict_waymo_test
from simwam.utils.config_resolvers import register_default_resolvers

register_default_resolvers()
logger = logging.getLogger(__name__)


@hydra.main(version_base="1.3", config_path="../../configs", config_name="predict_waymo_test")
def main(cfg: DictConfig):
    checkpoint = Path(str(cfg.ckpt)).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    options = cfg.PREDICTION
    if options.device not in {"cuda", "cpu"}:
        raise ValueError("PREDICTION.device must be cuda or cpu")
    if options.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    precision = str(cfg.mixed_precision)
    dtype = {"no": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[precision]
    os.environ["ACCELERATE_USE_DEEPSPEED"] = "false"
    accelerator = Accelerator(
        cpu=options.device == "cpu", mixed_precision=precision,
        kwargs_handlers=[InitProcessGroupKwargs(timeout=timedelta(hours=2))],
    )
    set_seed(int(options.seed))
    shared = [str(Path(str(options.output_dir)).absolute()) if accelerator.is_main_process else None]
    broadcast_object_list(shared, from_process=0)
    output = Path(shared[0])
    output.mkdir(parents=True, exist_ok=True)
    cfg.PREDICTION.output_dir = str(output)
    logging.basicConfig(
        level=logging.INFO, format=f"%(asctime)s [rank {accelerator.process_index}] %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(output/f"predict_rank_{accelerator.process_index:03d}.log")],
        force=True,
    )
    if not cfg.model.skip_dit_load_from_pretrain:
        raise ValueError("Test inference requires full trained weights and skip_dit_load_from_pretrain=true")
    if int(cfg.model.action_dit_config.action_dim) != 2 or cfg.model.mot_attention_mask_mode != "isolated":
        raise ValueError("Waymo test requires XY actions and isolated action-only conditioning")
    dataset = instantiate(cfg.data.train)
    model = instantiate(cfg.model, model_dtype=dtype, device=str(accelerator.device))
    checkpoint_info = load_waymo_checkpoint(model, checkpoint)
    model.eval()
    if accelerator.is_main_process:
        parameters = sum(parameter.numel() for parameter in model.parameters())
        checkpoint_info.update(
            num_model_parameters_exact=parameters,
            num_model_parameters=f"{math.ceil(parameters / 1_000_000)}M",
            uses_public_model_pretraining=True,
            public_model_names=[str(cfg.model.model_id)],
            parameter_count_note="Loaded inference model, including Video/Action DiTs, VAE and proprio; fixed text context is cached.",
        )
        OmegaConf.save(cfg, output/"config.yaml", resolve=True)
        (output/"checkpoint.json").write_text(json.dumps(checkpoint_info, indent=2)+"\n")
        logger.info("Checkpoint step=%s, unique model parameters=%d", checkpoint_info["step"], parameters)
    accelerator.wait_for_everyone()
    result = predict_waymo_test(
        model, dataset, accelerator, output, num_inference_steps=int(options.num_inference_steps),
        seed=int(options.seed), max_samples=options.max_samples,
    )
    if accelerator.is_main_process:
        logger.info("Finished %d test frames; complete=%s; output=%s",
                    result["num_predictions"], result["complete_input_coverage"], output)
    accelerator.end_training()


if __name__ == "__main__":
    main()
