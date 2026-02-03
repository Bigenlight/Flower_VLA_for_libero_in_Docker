import sys
import os
from pathlib import Path
import cv2
import numpy as np
import torch
import torchvision.transforms as T
from tqdm import tqdm

# --- PATH FIXER ---
current_path = Path(__file__).parent.absolute()
root_path = current_path 
libero_path = root_path / "LIBERO"
if libero_path.exists(): sys.path.append(str(libero_path))
sys.path.append(str(root_path))
# ------------------

try:
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
except ImportError:
    print("❌ Critical Error: Could not import LIBERO.")
    sys.exit(1)

from flower.evaluation.utils import load_mode_from_safetensor

def get_transforms():
    # Standard transforms for FLOWER (Florence-2 based)
    return T.Compose([
        T.ToTensor(),
        T.Resize((112, 112), antialias=True),
        T.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], 
                    std=[0.26862954, 0.26130258, 0.27577711])
    ])

def run_interactive_v2():
    print(f"--- FLOWER VLA Interactive Mode v2.1 (Init State Fix) ---")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. Select Model
    print("\nSelect Brain (Model):")
    print("  [1] flower_libero_10 (Simple)")
    print("  [2] flower_libero_90 (Complex)")
    choice = input("Choice (default 2): ").strip()
    
    if choice == "1":
        ckpt_name = "flower_libero_10"
        bench_name = "libero_10"
    else:
        ckpt_name = "flower_libero_90"
        bench_name = "libero_90"

    checkpoint_dir = root_path / "checkpoints" / ckpt_name
    # Fix for potentially renamed/patched configs
    if not (checkpoint_dir / "config.yaml").exists() and (checkpoint_dir / "config.json").exists():
         print("⚠️ Found config.json but expected config.yaml. Please run the fix_config.py script first!")
         return

    print(f"Loading {ckpt_name}...")
    model = load_mode_from_safetensor(checkpoint_dir).to(device)
    model.eval()
    model.reset()
    print("✅ Model loaded.")

    # 2. Select Task
    benchmark_instance = benchmark.get_benchmark_dict()[bench_name]()
    
    # Simple search
    query = input("\nSearch task (e.g. 'cabinet', 'drawer', 'bowl'): ").lower()
    found = []
    for i in range(benchmark_instance.get_num_tasks()):
        name = benchmark_instance.get_task(i).name
        if query in name.lower():
            found.append((i, name))
            
    if not found: 
        print("No tasks found.")
        return
        
    for idx, name in found[:15]: 
        print(f"[{idx}] {name}")
    if len(found) > 15: print("...")
    
    try:
        task_id = int(input("Task ID: "))
        task = benchmark_instance.get_task(task_id)
    except:
        print("Invalid Task ID")
        return
    
    # 3. Setup Env & LOAD INIT STATE
    bddl_file = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    env = OffScreenRenderEnv(
        bddl_file_name=bddl_file, 
        camera_heights=224, camera_widths=224, 
        render_gpu_device_id=0
    )
    
    # --- CRITICAL FIX: Load Init States ---
    # This mirrors flower_eval_libero.py logic
    print("\nLoading specific initial states for this task...")
    try:
        # Libero stores valid start states for every task
        init_states = benchmark_instance.get_task_init_states(task_id) 
        # We pick the first one (index 0) to ensure a valid starting pose
        # The model was trained to start from these specific states!
        obs = env.set_init_state(init_states[0])
        print("✅ Initial state loaded successfully.")
    except Exception as e:
        print(f"⚠️ Warning: Could not load init state ({e}). Using random reset (Expect Jitter!).")
        obs = env.reset()
    # --------------------------------------

    # 4. Run
    default_instr = task.language
    print(f"\nDefault Instruction: '{default_instr}'")
    user_instruction = input(f"Command (Press Enter to use default): ").strip()
    if not user_instruction:
        user_instruction = default_instr
    
    transforms = get_transforms()
    # Use a descriptive filename
    safe_task_name = task.name[:30]
    video_path = root_path / f"result_{bench_name}_{task_id}.mp4"
    video_writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*'mp4v'), 20.0, (448, 224))
    
    # Physics Warmup
    for _ in range(5): obs, _, _, _ = env.step(np.zeros(7))

    print(f"Running task with instruction: '{user_instruction}'")
    
    # Run loop
    for step in tqdm(range(600)):
        img_static = obs['agentview_image'] 
        img_gripper = obs['robot0_eye_in_hand_image']
        
        # Add dimensions: (Batch=1, Time=1, C, H, W)
        static_tensor = transforms(img_static).unsqueeze(0).unsqueeze(0).to(device)
        gripper_tensor = transforms(img_gripper).unsqueeze(0).unsqueeze(0).to(device)
        
        # Robot state also needs batching
        robot_obs = torch.tensor(obs['robot0_joint_pos']).unsqueeze(0).unsqueeze(0).to(device)
        gripper_state = torch.tensor(obs['robot0_gripper_qpos']).unsqueeze(0).unsqueeze(0).to(device)

        obs_dict = {
            "rgb_obs": {"rgb_static": static_tensor, "rgb_gripper": gripper_tensor},
            "robot_obs": robot_obs, "gripper_states": gripper_state
        }

        with torch.no_grad():
            action = model.step(obs_dict, {"lang_text": user_instruction})
        
        obs, _, done, _ = env.step(action.detach().cpu().numpy().flatten())
        
        combined = np.concatenate((img_static, img_gripper), axis=1)
        if combined.dtype != np.uint8: combined = (combined * 255).astype(np.uint8)
        video_writer.write(cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))
        
        if done:
            print("🎉 Success! The environment detected task completion.")
            break

    video_writer.release()
    env.close()
    print(f"✅ Video saved to {video_path}")

if __name__ == "__main__":
    run_interactive_v2()