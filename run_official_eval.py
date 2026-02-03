import sys
import os
from pathlib import Path
import cv2
import numpy as np
import torch
import hydra
import torchvision.transforms as T
from tqdm import tqdm
from omegaconf import OmegaConf

# --- PATH FIXER ---
current_path = Path(__file__).parent.absolute()
libero_path = current_path / "LIBERO"
if libero_path.exists(): sys.path.append(str(libero_path))
sys.path.append(str(current_path))
# ------------------

from libero.libero import benchmark, get_libero_path
from flower.evaluation.utils import load_mode_from_safetensor

def run_final_test():
    print(f"--- FLOWER VLA Final Integrated Test ---")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. Load Model
    ckpt_path = Path("checkpoints/flower_libero_90")
    if not ckpt_path.exists():
        print(f"❌ Error: Checkpoint not found at {ckpt_path}")
        return

    print(f"Loading model from {ckpt_path}...")
    model = load_mode_from_safetensor(ckpt_path).to(device)
    model.eval()
    model.reset()

    # 2. Setup Benchmark & Task
    benchmark_instance = benchmark.get_benchmark_dict()["libero_90"]()
    # Task 1: Close top drawer and put bowl
    task_id = 1 
    task = benchmark_instance.get_task(task_id)
    print(f"Task: {task.name}")
    print(f"Instruction: '{task.language}'")

    # 3. Setup Environment
    bddl_file = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    from libero.libero.envs import OffScreenRenderEnv
    env = OffScreenRenderEnv(
        bddl_file_name=bddl_file,
        camera_heights=224,
        camera_widths=224,
        render_gpu_device_id=0
    )
    
    # 4. FIXED: Action Scaling (Prevents Jitter)
    # 0.05 is standard for Libero delta actions
    ACTION_SCALE = 0.05 
    print(f"Applying manual action scaling: {ACTION_SCALE}")

    # 5. FIXED: Init State (Prevents Confusion)
    try:
        init_states = benchmark_instance.get_task_init_states(task_id)
        obs = env.set_init_state(init_states[0])
        print("✅ Initial state loaded.")
    except Exception as e:
        print(f"⚠️ Warning: Using random reset ({e})")
        obs = env.reset()
    
    # 6. FIXED: Compile Transforms
    print("Compiling transforms...")
    cfg = OmegaConf.load(ckpt_path / "config.yaml")
    val_transforms_cfg = cfg.datamodule.transforms.val
    
    transforms = {}
    for key, transform_list_cfg in val_transforms_cfg.items():
        t_list = [hydra.utils.instantiate(t_cfg) for t_cfg in transform_list_cfg]
        transforms[key] = T.Compose(t_list)

    # 7. Run Loop
    video_writer = cv2.VideoWriter("result_final.mp4", cv2.VideoWriter_fourcc(*'mp4v'), 20.0, (448, 224))
    
    # Warmup
    for _ in range(5): env.step(np.zeros(7))

    for step in tqdm(range(600)):
        img_static = obs['agentview_image'] 
        img_gripper = obs['robot0_eye_in_hand_image']
        
        # --- FIX A: Convert NumPy -> Tensor (CHW) ---
        # Solves "TypeError: Unexpected type numpy"
        static_tensor_in = torch.from_numpy(img_static.copy()).permute(2, 0, 1)
        gripper_tensor_in = torch.from_numpy(img_gripper.copy()).permute(2, 0, 1)
        
        # --- FIX B: Add Time Dimension (B, T, C, H, W) ---
        # Solves "ValueError: expected 5, got 4"
        # 1. Apply Transform -> (C, H, W)
        # 2. unsqueeze(0) -> Batch (1, C, H, W)
        # 3. unsqueeze(0) -> Time (1, 1, C, H, W)
        static_tensor = transforms['rgb_static'](static_tensor_in).unsqueeze(0).unsqueeze(0).to(device)
        gripper_tensor = transforms['rgb_gripper'](gripper_tensor_in).unsqueeze(0).unsqueeze(0).to(device)
        
        # Same for robot state
        robot_obs = torch.tensor(obs['robot0_joint_pos']).unsqueeze(0).unsqueeze(0).to(device)
        gripper_state = torch.tensor(obs['robot0_gripper_qpos']).unsqueeze(0).unsqueeze(0).to(device)

        obs_dict = {
            "rgb_obs": {"rgb_static": static_tensor, "rgb_gripper": gripper_tensor},
            "robot_obs": robot_obs, "gripper_states": gripper_state
        }

        with torch.no_grad():
            action = model.step(obs_dict, {"lang_text": task.language})
        
        raw_action = action.detach().cpu().numpy().flatten()
        
        # --- FIX C: Apply Scaling ---
        # Solves "Robot Jitter/Speed"
        scaled_action = raw_action.copy()
        scaled_action[:6] = raw_action[:6] * ACTION_SCALE
        
        obs, _, done, _ = env.step(scaled_action)
        
        combined = np.concatenate((img_static, img_gripper), axis=1)
        if combined.dtype != np.uint8: combined = (combined * 255).astype(np.uint8)
        video_writer.write(cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))
        
        if done:
            print("🎉 Success!")
            break

    video_writer.release()
    env.close()
    print("✅ Video saved to result_final.mp4")

if __name__ == "__main__":
    run_final_test()