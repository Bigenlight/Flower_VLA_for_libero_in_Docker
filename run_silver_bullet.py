import sys
import os
import torch
import numpy as np
import cv2
import hydra
import torchvision.transforms as T
from pathlib import Path
from tqdm import tqdm
from omegaconf import OmegaConf

# --- PATH FIXER ---
current_path = Path(__file__).parent.absolute()
libero_path = current_path / "LIBERO"
if libero_path.exists(): sys.path.insert(0, str(libero_path))
sys.path.insert(0, str(current_path))
# ------------------

from libero.libero import benchmark, get_libero_path
from flower.evaluation.utils import load_mode_from_safetensor

def run_silver_bullet():
    print(f"--- FLOWER VLA: The Silver Bullet Test ---")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. Load Model (Using the safe loader we know works)
    ckpt_path = Path("checkpoints/flower_libero_90")
    print(f"Loading model from {ckpt_path}...")
    try:
        model = load_mode_from_safetensor(ckpt_path).to(device)
        model.eval()
        model.reset()
    except Exception as e:
        print(f"❌ Load Error: {e}")
        return

    # 2. Setup Task
    # We stick to Task 1 because we know it works in the env
    benchmark_instance = benchmark.get_benchmark_dict()["libero_90"]()
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
    
    # 4. Initialize Physics
    try:
        init_states = benchmark_instance.get_task_init_states(task_id)
        obs = env.set_init_state(init_states[0])
        print("✅ Initial state loaded.")
    except:
        obs = env.reset()

    # 5. Compile Transforms
    print("Compiling transforms...")
    cfg = OmegaConf.load(ckpt_path / "config.yaml")
    transforms = {}
    for key, t_list in cfg.datamodule.transforms.val.items():
        transforms[key] = T.Compose([hydra.utils.instantiate(t) for t in t_list])

    # 6. Run Loop with WRAPPER LOGIC
    video_writer = cv2.VideoWriter("silver_bullet.mp4", cv2.VideoWriter_fourcc(*'mp4v'), 20.0, (448, 224))
    
    # Action Config
    # If the robot is still too fast, lower this.
    # If it barely moves, raise it.
    ACTION_SCALE = 0.2 
    
    for step in tqdm(range(600)):
        img_static = obs['agentview_image'] 
        img_gripper = obs['robot0_eye_in_hand_image']
        
        # [Fix 1] NumPy -> Tensor (CHW)
        static_tensor = torch.from_numpy(img_static.copy()).permute(2, 0, 1)
        gripper_tensor = torch.from_numpy(img_gripper.copy()).permute(2, 0, 1)
        
        # [Fix 2] Transforms + Dimensions (B, T, C, H, W)
        static_in = transforms['rgb_static'](static_tensor).unsqueeze(0).unsqueeze(0).to(device)
        gripper_in = transforms['rgb_gripper'](gripper_tensor).unsqueeze(0).unsqueeze(0).to(device)
        
        # Robot State
        robot_obs = torch.tensor(obs['robot0_joint_pos']).unsqueeze(0).unsqueeze(0).to(device)
        gripper_state = torch.tensor(obs['robot0_gripper_qpos']).unsqueeze(0).unsqueeze(0).to(device)

        obs_dict = {
            "rgb_obs": {"rgb_static": static_in, "rgb_gripper": gripper_in},
            "robot_obs": robot_obs, "gripper_states": gripper_state
        }

        with torch.no_grad():
            action = model.step(obs_dict, {"lang_text": task.language})
        
        # [Fix 3] Process Action
        raw_action = action.detach().cpu().numpy().flatten()
        
        # Apply Scaling (Velocity Control)
        scaled_action = raw_action.copy()
        scaled_action[:6] = raw_action[:6] * ACTION_SCALE
        
        # Apply Gripper Binarization (From HulcWrapper!)
        # -1 = Closed, 1 = Open
        scaled_action[-1] = 1.0 if raw_action[-1] > 0 else -1.0
        
        obs, _, done, _ = env.step(scaled_action)
        
        combined = np.concatenate((img_static, img_gripper), axis=1)
        if combined.dtype != np.uint8: combined = (combined * 255).astype(np.uint8)
        video_writer.write(cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))
        
        if done:
            print("🎉 Success!")
            break

    video_writer.release()
    env.close()
    print("✅ Video saved to silver_bullet.mp4")

if __name__ == "__main__":
    run_silver_bullet()