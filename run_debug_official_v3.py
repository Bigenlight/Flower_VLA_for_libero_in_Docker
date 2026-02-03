import sys
import os
import torch
import hydra
import numpy as np
from pathlib import Path
from omegaconf import OmegaConf
from safetensors.torch import load_file

# --- PATH FIXER ---
current_path = Path(__file__).parent.absolute()
libero_path = current_path / "LIBERO"
if libero_path.exists(): sys.path.insert(0, str(libero_path))
sys.path.insert(0, str(current_path))
# ------------------

from flower.evaluation.flower_eval_libero import EvaluateLibero

def run_official_final():
    print("--- FLOWER VLA: Official Evaluation (v3: NoGrad Fix) ---")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. Load Config
    ckpt_path = Path("checkpoints/flower_libero_90")
    config_path = ckpt_path / "config.yaml"
    print(f"Loading config from {config_path}...")
    cfg = OmegaConf.load(config_path)
    
    # [Patch 1] Fix Missing 'lang_folder'
    if "lang_folder" not in cfg.datamodule.datasets.lang_dataset:
        cfg.datamodule.datasets.lang_dataset.lang_folder = "dummy_path"
    
    # [Patch 2] Force the correct Wrapper (HulcWrapper is correct for this repo!)
    # This wrapper handles the gripper binarization [-1, 1] which fixes jitter.
    if "env_cfg" not in cfg.callbacks.rollout_lh:
         cfg.callbacks.rollout_lh.env_cfg = {"_target_": "flower.wrappers.hulc_wrapper.HulcWrapper"}

    # 2. Instantiate Model
    print("Instantiating model...")
    model = hydra.utils.instantiate(cfg.model)
    
    # 3. Load Weights
    print("Loading weights...")
    state_dict = load_file(ckpt_path / "model.safetensors")
    if "state_dict" in state_dict: state_dict = state_dict["state_dict"]
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    
    # 4. Instantiate Transforms
    print("Compiling transforms...")
    # The official code uses 'val' transforms for eval
    transforms = hydra.utils.instantiate(cfg.datamodule.transforms.val)

    # 5. Initialize Evaluator
    evaluator = EvaluateLibero(
        model=model,
        transforms=transforms,
        log_dir=Path("eval_logs_official"),
        benchmark_name="libero_90",
        num_sequences=1,
        max_steps=600,
        num_videos=1,
        n_eval=1,
        task_embedding_format="clip",
        device=device
    )
    
    # 6. Run Task 1
    target_task_id = 1
    print(f"\n🚀 Running Task {target_task_id}...")
    task = evaluator.benchmark_instance.get_task(target_task_id)
    task_emb = evaluator.benchmark_instance.task_embs[target_task_id]
    print(f"Instruction: {task.language}")
    
    # --- CRITICAL FIX: torch.no_grad() ---
    # This prevents the "Can't call numpy() on Tensor that requires grad" error
    with torch.no_grad():
        success = evaluator.evaluate_task(
            model=model,
            task_i=task,
            task_emb=task_emb,
            task_str=f"official_fix_task_{target_task_id}",
            idx=target_task_id,
            store_video=True
        )
    
    print(f"\n✅ Finished. Success: {success}")
    print(f"🎥 Video saved in: eval_logs_official")

if __name__ == "__main__":
    run_official_final()