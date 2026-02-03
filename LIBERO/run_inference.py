import os
import cv2
import numpy as np
import torch
import torchvision.transforms as T
from tqdm import tqdm
from pathlib import Path

# Import LIBERO
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

# Import FLOWER utilities
# We use the existing loader which handles the config parsing automatically
from flower.evaluation.utils import load_mode_from_safetensor

def get_transforms():
    """
    Replicates the 'val' transforms from libero_transforms.yaml
    1. Resize to 112x112
    2. Scale 0-1
    3. Normalize with CLIP stats
    """
    return T.Compose([
        T.ToTensor(),                              # Converts HWC (0-255) -> CHW (0.0-1.0)
        T.Resize((112, 112), antialias=True),      # Downsample 224 -> 112
        T.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073],
            std=[0.26862954, 0.26130258, 0.27577711]
        )
    ])

def run_inference():
    print("--- STEP 3: FLOWER VLA Inference Test ---")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # ---------------------------------------------------------
    # 1. Load the Model
    # ---------------------------------------------------------
    checkpoint_path = Path("checkpoints/flower_libero_10")
    print(f"Loading model from: {checkpoint_path}")
    
    # This utility loads the architecture based on the config.yaml inside the folder
    model = load_mode_from_safetensor(checkpoint_path)
    model = model.to(device)
    model.eval() # Set to inference mode (disable dropout, etc)
    model.reset() # Reset the action chunking buffer
    
    print("✅ Model loaded successfully.")

    # ---------------------------------------------------------
    # 2. Setup LIBERO Environment
    # ---------------------------------------------------------
    benchmark_name = "libero_10"
    benchmark_dict = benchmark.get_benchmark_dict()
    benchmark_instance = benchmark_dict[benchmark_name]()
    
    # Task 0: Living Room Scene 2 (Alphabet Soup + Tomato Sauce)
    task_id = 0 
    task = benchmark_instance.get_task(task_id)
    print(f"Task: {task.name}")
    print(f"Instruction: '{task.language}'")

    bddl_folder = get_libero_path("bddl_files")
    bddl_file = os.path.join(bddl_folder, task.problem_folder, task.bddl_file)

    # Note: We render at 224 for higher quality visualization, 
    # but we will resize to 112 for the model.
    env_args = {
        "bddl_file_name": bddl_file,
        "camera_heights": 224, 
        "camera_widths": 224,
        "render_gpu_device_id": 0
    }
    env = OffScreenRenderEnv(**env_args)
    env.reset()

    # ---------------------------------------------------------
    # 3. Inference Loop
    # ---------------------------------------------------------
    transforms = get_transforms()
    video_writer = cv2.VideoWriter("inference_result.mp4", cv2.VideoWriter_fourcc(*'mp4v'), 20.0, (448, 224))
    
    print("Starting interaction...")
    
    # We define the goal ONCE. The model needs it every step.
    # The dictionary key "lang_text" is what the model expects (see flower.py logic)
    goal_dict = {
        "lang_text": task.language 
    }

    # Warmup simulation
    dummy_action = np.zeros(7)
    for _ in range(10): env.step(dummy_action)

    obs = env.get_observation()
    
    # Run for 200 steps (approx 10 seconds)
    for step in tqdm(range(200)):
        
        # A. Preprocess Images
        # Env gives: (224, 224, 3) numpy array
        img_static = obs['agentview_image']
        img_gripper = obs['robot0_eye_in_hand_image']
        
        # Apply Transforms: Numpy -> Tensor -> Resize(112) -> Norm -> Batch Dim
        # Outcome shape: (1, 3, 112, 112)
        static_tensor = transforms(img_static).unsqueeze(0).to(device)
        gripper_tensor = transforms(img_gripper).unsqueeze(0).to(device)
        
        # B. Construct Observation Dict for Model
        # Structure matches flower/models/flower.py input expectations
        obs_dict = {
            "rgb_obs": {
                "rgb_static": static_tensor,
                "rgb_gripper": gripper_tensor
            },
            # We map LIBERO robot state to model keys
            "robot_obs": torch.tensor(obs['robot0_joint_pos']).unsqueeze(0).to(device),
            "gripper_states": torch.tensor(obs['robot0_gripper_qpos']).unsqueeze(0).to(device)
        }

        # C. Inference
        with torch.no_grad():
            # Model returns action: (1, 1, 7) or (1, 7) depending on chunking
            action_tensor = model.step(obs_dict, goal_dict)
        
        # D. Post-process Action
        # Detach -> CPU -> Numpy -> Squeeze batch dims -> (7,)
        action = action_tensor.detach().cpu().numpy().flatten()
        
        # E. Step Environment
        obs, reward, done, info = env.step(action)
        
        # F. Visualization (Save the 224px version, not the 112px version)
        combined_img = np.concatenate((img_static, img_gripper), axis=1)
        # Flip RGB to BGR for OpenCV video
        combined_img = cv2.cvtColor(combined_img, cv2.COLOR_RGB2BGR)
        # Ensure correct scale (Libero is sometimes float 0-1, sometimes uint8 0-255)
        if combined_img.dtype != np.uint8:
            combined_img = (combined_img * 255).astype(np.uint8)
            
        video_writer.write(combined_img)
        
        if done:
            print("Success!")
            break

    video_writer.release()
    env.close()
    print("✅ Inference finished. Video saved to 'inference_result.mp4'")

if __name__ == "__main__":
    run_inference()