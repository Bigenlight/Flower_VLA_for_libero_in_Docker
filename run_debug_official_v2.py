import sys
import os
import torch
import hydra
import numpy as np
from pathlib import Path
from omegaconf import OmegaConf
from safetensors.torch import load_file  # <--- CRITICAL FIX

# --- PATH FIXER ---
current_path = Path(__file__).parent.absolute()
libero_path = current_path / "LIBERO"
if libero_path.exists(): sys.path.insert(0, str(libero_path))
sys.path.insert(0, str(current_path))
# ------------------

from flower.evaluation.flower_eval_libero import EvaluateLibero, get_log_dir

def run_official_patched():
    print("--- FLOWER VLA: Official Evaluation (Fixed Loader) ---")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. Load Config
    ckpt_path = Path("checkpoints/flower_libero_90")
    config_path = ckpt_path / "config.yaml"
    print(f"Loading config from {config_path}...")
    cfg = OmegaConf.load(config_path)
    
    # [Patch] Inject missing keys to prevent crashes
    if "lang_folder" not in cfg.datamodule.datasets.lang_dataset:
        cfg.datamodule.datasets.lang_dataset.lang_folder = "dummy_lang_path"
    
    # [Patch] Force the wrapper to be compatible if missing
    # This ensures the environment gets the right action processing
    if "env_cfg" not in cfg.callbacks.rollout_lh:
         cfg.callbacks.rollout_lh.env_cfg = {"_target_": "flower.wrappers.hulc_wrapper.HulcWrapper"}

    # 2. Instantiate Model
    print("Instantiating model architecture...")
    model = hydra.utils.instantiate(cfg.model)
    
    # 3. Load Weights (SAFECTORS FIX)
    print("Loading weights from .safetensors...")
    state_dict = load_file(ckpt_path / "model.safetensors") # <--- Fixed loading method
    
    # Handle potentially nested keys (common in PL checkpoints)
    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
        
    # Load into model
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"⚠️ Missing keys (Safe to ignore for VLA): {len(missing)}")
    
    model.to(device)
    model.eval()
    
    # 4. Instantiate Transforms
    print("Compiling transforms...")
    transforms = hydra.utils.instantiate(cfg.datamodule.transforms.val)

    # 5. Initialize Official Evaluator
    log_dir = Path("eval_logs_official")
    log_dir.mkdir(exist_ok=True)
    
    print("Initializing EvaluateLibero...")
    evaluator = EvaluateLibero(
        model=model,
        transforms=transforms,
        log_dir=log_dir,
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
    
    # This calls the author's exact evaluation loop
    success = evaluator.evaluate_task(
        model=model,
        task_i=task,
        task_emb=task_emb,
        task_str=f"official_debug_task_{target_task_id}",
        idx=target_task_id,
        store_video=True
    )
    
    print(f"\n✅ Finished. Success: {success}")
    print(f"🎥 Video saved in: {log_dir}")

if __name__ == "__main__":
    run_official_patched()