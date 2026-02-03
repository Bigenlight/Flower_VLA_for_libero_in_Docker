import sys
import os
import torch
import hydra
import numpy as np
from pathlib import Path
from omegaconf import OmegaConf

# --- PATH FIXER ---
# Ensure LIBERO and current folder are visible
current_path = Path(__file__).parent.absolute()
libero_path = current_path / "LIBERO"
if libero_path.exists(): sys.path.insert(0, str(libero_path))
sys.path.insert(0, str(current_path))
# ------------------

from flower.evaluation.flower_eval_libero import EvaluateLibero, get_log_dir
from flower.models.flower import FLOWERVLA  # Import model class directly if needed

def run_official_patched():
    print("--- FLOWER VLA: Official Evaluation (Manual Patch) ---")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # 1. Load & Patch Configuration
    ckpt_path = Path("checkpoints/flower_libero_90")
    config_path = ckpt_path / "config.yaml"
    
    print(f"Loading config from {config_path}...")
    cfg = OmegaConf.load(config_path)
    
    # --- THE PATCH: Inject missing keys to prevent crashes ---
    # This fixes the 'Missing key lang_folder' error
    if "lang_folder" not in cfg.datamodule.datasets.lang_dataset:
        print("🔧 Patching missing 'lang_folder' in config...")
        cfg.datamodule.datasets.lang_dataset.lang_folder = "dummy_lang_path"
        
    # Patch 2: Ensure wrapper config exists if missing
    if "env_cfg" not in cfg.callbacks.rollout_lh:
         print("🔧 Patching missing 'env_cfg'...")
         cfg.callbacks.rollout_lh.env_cfg = {"_target_": "flower.wrappers.hulc_wrapper.HulcWrapper"}
    # ---------------------------------------------------------

    # 2. Instantiate Model Manually (Bypassing utils.py)
    print("Instantiating model from config...")
    # We use hydra to build the model class exactly as defined in YAML
    model = hydra.utils.instantiate(cfg.model)
    
    # Load Weights
    print("Loading weights...")
    state_dict = torch.load(ckpt_path / "model.safetensors", map_location=device)
    
    # Handle 'state_dict' wrapper if present (common in Lightning)
    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    
    # Clean keys (remove 'module.' or 'vlm.' prefixes if needed)
    # The authors' loader does this, we might need to be careful.
    # For now, try direct load.
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"⚠️ Warning: Missing keys (Safe to ignore if minor): {len(missing)}")
    
    model.to(device)
    model.eval()
    
    # 3. Instantiate Transforms
    print("Compiling transforms...")
    transforms = hydra.utils.instantiate(cfg.datamodule.transforms.val)

    # 4. Setup Log Dir
    log_dir = Path("eval_logs_debug")
    log_dir.mkdir(exist_ok=True)

    # 5. Initialize Official Evaluator
    # This class handles the Init States, Envs, and Action Processing for us!
    print("Initializing EvaluateLibero...")
    evaluator = EvaluateLibero(
        model=model,
        transforms=transforms,
        log_dir=log_dir,
        benchmark_name="libero_90",
        num_sequences=1,
        max_steps=600,  # 30 seconds
        num_videos=1,   # Save video
        n_eval=1,       # Run 1 episode
        task_embedding_format="clip",
        device=device
    )
    
    # 6. Run Task 1
    target_task_id = 1
    print(f"\n🚀 Running Task {target_task_id}...")
    
    # Get Task Info
    task = evaluator.benchmark_instance.get_task(target_task_id)
    task_emb = evaluator.benchmark_instance.task_embs[target_task_id]
    
    print(f"Instruction: {task.language}")
    
    # Run!
    success = evaluator.evaluate_task(
        model=model,
        task_i=task,
        task_emb=task_emb,
        task_str=f"debug_task_{target_task_id}",
        idx=target_task_id,
        store_video=True
    )
    
    print(f"\n✅ Finished. Success: {success}")
    print(f"🎥 Video saved in: {log_dir}")

if __name__ == "__main__":
    run_official_patched()