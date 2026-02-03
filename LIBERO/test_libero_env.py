import os
import cv2
import numpy as np
import torch
import time
from tqdm import tqdm

# Import LIBERO
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

def test_environment():
    print("--- STEP 2: LIBERO Environment Test ---")

    # 1. Setup Benchmark
    benchmark_name = "libero_10"
    print(f"Loading benchmark: {benchmark_name}")
    benchmark_dict = benchmark.get_benchmark_dict()
    benchmark_instance = benchmark_dict[benchmark_name]()
    
    # 2. Pick a specific task (Task 0 is usually 'pick up black bowl')
    task_id = 0
    task = benchmark_instance.get_task(task_id)
    task_name = task.name
    print(f"Selected Task: {task_name}")
    print(f"Language Instruction: {task.language}")

    # 3. Locate BDDL file
    bddl_folder = get_libero_path("bddl_files")
    bddl_file = os.path.join(bddl_folder, task.problem_folder, task.bddl_file)
    print(f"Loading BDDL: {bddl_file}")

    # 4. Initialize Environment
    # We match the Flower config: 224x224 resolution
    env_args = {
        "bddl_file_name": bddl_file,
        "camera_heights": 224,
        "camera_widths": 224,
        "render_gpu_device_id": 0
    }

    try:
        env = OffScreenRenderEnv(**env_args)
        print("✅ Environment initialized successfully.")
    except Exception as e:
        print(f"❌ Failed to init env: {e}")
        return

    # 5. Reset and Run Loop
    print("Starting simulation loop...")
    env.reset()
    
    # Initialize Video Writer
    video_path = "debug_render.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # Concatenating AgentView (224) + EyeInHand (224) = 448 width
    video_writer = cv2.VideoWriter(video_path, fourcc, 20.0, (448, 224))

    # Dummy Action: [x, y, z, ax, ay, az, gripper]
    # 0.0 means "hold position" (since it's delta control)
    # -1.0 gripper usually means "open"
    dummy_action = np.zeros(7)
    dummy_action[-1] = -1.0 

    for step in tqdm(range(60)): # Run for 3 seconds (20Hz)
        obs, reward, done, info = env.step(dummy_action)
        
        # Extract images (Standard Libero keys)
        # Flower expects: 'agentview_image' and 'robot0_eye_in_hand_image'
        img_static = obs['agentview_image'] 
        img_gripper = obs['robot0_eye_in_hand_image']
        
        # LIBERO returns images flipped? Sometimes. 
        # But usually they are [H, W, C] in range [0, 1] or [0, 255].
        # Let's assume standard numpy uint8 behavior for now or handle float.
        
        # Concatenate for visualization
        # Note: CV2 expects BGR, LIBERO usually gives RGB. Convert if colors look weird.
        combined_img = np.concatenate((img_static, img_gripper), axis=1)
        
        # Ensure it is uint8 [0, 255] for video writer
        if combined_img.dtype == np.float32 or combined_img.dtype == np.float64:
             combined_img = (combined_img * 255).astype(np.uint8)
        
        # RGB to BGR for OpenCV
        combined_img = cv2.cvtColor(combined_img, cv2.COLOR_RGB2BGR)
        
        video_writer.write(combined_img)

    video_writer.release()
    env.close()
    print(f"✅ Test Complete. Video saved to {video_path}")
    print("Please check the video. If you see the robot arm and the table, we are ready for Step 3.")

if __name__ == "__main__":
    test_environment()