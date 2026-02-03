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
    return T.Compose([
        T.ToTensor(),
        T.Resize((112, 112), antialias=True),
        T.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], 
                    std=[0.26862954, 0.26130258, 0.27577711])
    ])

def run_interactive():
    print(f"--- STEP 4: FLOWER VLA Interactive Mode ---")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. Load Model
    checkpoint_dir = root_path / "checkpoints/flower_libero_10"
    print(f"Loading model... (this takes a few seconds)")
    model = load_mode_from_safetensor(checkpoint_dir).to(device)
    model.eval()
    model.reset()
    print("✅ Model ready.")

    # 2. Setup Environment (Task 0: Soup + Sauce Scene)
    benchmark_instance = benchmark.get_benchmark_dict()["libero_10"]()
    task = benchmark_instance.get_task(0) 
    
    bddl_file = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    env = OffScreenRenderEnv(
        bddl_file_name=bddl_file, 
        camera_heights=224, camera_widths=224, 
        render_gpu_device_id=0
    )
    obs = env.reset()

    # 3. INTERACTIVE PROMPT
    print("\n" + "="*40)
    print(f"Scene Loaded: {task.name}")
    print("Available Objects: Alphabet Soup, Tomato Sauce, Basket")
    print("Example commands:")
    print(" - 'put the alphabet soup in the basket'")
    print(" - 'put the tomato sauce in the basket'")
    print(" - 'move the soup to the right'")
    print("="*40)
    
    user_instruction = input("\n🤖 Enter your command: ")
    print(f"\nProcessing command: '{user_instruction}'...")

    # 4. Run Inference
    transforms = get_transforms()
    video_path = root_path / "interactive_result.mp4"
    video_writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*'mp4v'), 20.0, (448, 224))
    
    # Warmup
    for _ in range(20): obs, _, _, _ = env.step(np.zeros(7))

    # Run for 600 steps (30 seconds)
    for step in tqdm(range(600)):
        img_static = obs['agentview_image'] 
        img_gripper = obs['robot0_eye_in_hand_image']
        
        # Prepare Tensors (B, T, C, H, W)
        static_tensor = transforms(img_static).unsqueeze(0).unsqueeze(0).to(device)
        gripper_tensor = transforms(img_gripper).unsqueeze(0).unsqueeze(0).to(device)
        robot_obs = torch.tensor(obs['robot0_joint_pos']).unsqueeze(0).unsqueeze(0).to(device)
        gripper_state = torch.tensor(obs['robot0_gripper_qpos']).unsqueeze(0).unsqueeze(0).to(device)

        obs_dict = {
            "rgb_obs": {"rgb_static": static_tensor, "rgb_gripper": gripper_tensor},
            "robot_obs": robot_obs, "gripper_states": gripper_state
        }

        # INJECT USER TEXT HERE
        with torch.no_grad():
            action = model.step(obs_dict, {"lang_text": user_instruction})
        
        obs, _, done, _ = env.step(action.detach().cpu().numpy().flatten())
        
        # Render
        combined = np.concatenate((img_static, img_gripper), axis=1)
        if combined.dtype != np.uint8: combined = (combined * 255).astype(np.uint8)
        video_writer.write(cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))
        
        if done:
            print(f"🎉 Success! Task completed in {step} steps.")
            break

    video_writer.release()
    env.close()
    print(f"✅ Video saved to {video_path}")

if __name__ == "__main__":
    run_interactive()