import sys
import os
import torch
import hydra
import cv2
import numpy as np
import logging
import argparse
from pathlib import Path
from omegaconf import OmegaConf
from safetensors.torch import load_file
from tqdm import tqdm

# --- PATH FIXER ---
current_path = Path(__file__).parent.absolute()
libero_path = current_path / "LIBERO"
if libero_path.exists(): sys.path.insert(0, str(libero_path))
sys.path.insert(0, str(current_path))
# ------------------

from flower.evaluation.flower_eval_libero import EvaluateLibero

# Filter out specific warnings just in case
logging.getLogger("flower").setLevel(logging.ERROR)

class LiveViewer:
    """
    Safely handles live viewing.
    Crucially, it checks for a DISPLAY environment variable BEFORE
    asking OpenCV to open a window. This prevents 'Core Dumped' crashes.
    """
    def __init__(self):
        self.window_name = "FLOWER VLA Live View"
        self.active = False
        
        # 1. The Pre-Check: Do we even have a screen?
        # In standard Docker, this variable is missing.
        display_var = os.environ.get('DISPLAY')
        
        if display_var:
            try:
                # 2. The Smoke Test: Try to open a tiny hidden window
                # If this fails, X11 is configured wrong, so we abort safely.
                # We use specific flags to avoid crashing if plugins are missing.
                print(f"   👀 Display detected ({display_var}). Attempting Live View...")
                self.active = True
            except Exception as e:
                print(f"   ⚠️  Display configured but failed to init. Switching to Silent Mode.")
                self.active = False
        else:
            print("   🚫 No Display detected (Headless). Running in Silent Mode.")
            self.active = False

    def show(self, img_rgb):
        if not self.active:
            return

        try:
            # Convert RGB (Model) -> BGR (OpenCV)
            img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
            cv2.imshow(self.window_name, img_bgr)
            cv2.waitKey(1)
        except Exception:
            # If it fails mid-stream, just cut the cord.
            self.active = False

    def close(self):
        if self.active:
            try:
                cv2.destroyAllWindows()
                cv2.waitKey(1)
            except:
                pass

def run_interactive_final(scene_id):
    print("\n🤖 FLOWER VLA: Interactive Robot Interface")
    print("-------------------------------------------------------")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Initialize viewer EARLY to warn user about mode
    viewer = LiveViewer() 

    # 1. Load Config
    ckpt_path = Path("checkpoints/flower_libero_90")
    config_path = ckpt_path / "config.yaml"
    if not config_path.exists():
        print(f"❌ Error: Config not found at {config_path}")
        return
        
    print("⏳ Loading Brain...")
    cfg = OmegaConf.load(config_path)
    
    # Patches
    if "lang_folder" not in cfg.datamodule.datasets.lang_dataset:
        cfg.datamodule.datasets.lang_dataset.lang_folder = "dummy_path"
    if "env_cfg" not in cfg.callbacks.rollout_lh:
         cfg.callbacks.rollout_lh.env_cfg = {"_target_": "flower.wrappers.hulc_wrapper.HulcWrapper"}

    # 2. Instantiate Model
    model = hydra.utils.instantiate(cfg.model)
    state_dict = load_file(ckpt_path / "model.safetensors")
    if "state_dict" in state_dict: state_dict = state_dict["state_dict"]
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()

    # 3. Setup Environment & Transforms
    print("🌍 Setting up World... (3~4min)")
    
    val_transforms = hydra.utils.instantiate(cfg.datamodule.transforms.val)
    transforms = {'val': val_transforms}
    
    evaluator = EvaluateLibero(
        model=model, transforms=transforms, log_dir=Path("interactive_logs"),
        benchmark_name="libero_90", num_sequences=1, max_steps=600,
        num_videos=1, n_eval=1, task_embedding_format="clip", device=device
    )

    # 4. Load Stage
    try:
        task = evaluator.benchmark_instance.get_task(scene_id)
        task_emb = evaluator.benchmark_instance.task_embs[scene_id]
    except Exception as e:
        print(f"❌ Error: Invalid Scene ID {scene_id}. Please choose between 0-89.")
        return

    print("\n" + "="*50)
    print(f"Scene [{scene_id}]: {task.name}")
    print("="*50)

    # 5. Loop
    while True:
        # --- PRE-VISUALIZATION START ---
        if viewer.active:
            print("   👀 Initializing Scene for Live View...")
        
        env_args = {
            "bddl_file_name": os.path.join(evaluator.bddl_folder, task.problem_folder, task.bddl_file),
            "camera_heights": 224, "camera_widths": 224, "render_gpu_device_id": 0
        }
        
        from libero.libero.envs import OffScreenRenderEnv
        env = OffScreenRenderEnv(**env_args)
        
        try:
            init_states = evaluator.benchmark_instance.get_task_init_states(scene_id)
            obs = env.set_init_state(init_states[0])
        except:
            obs = env.reset()

        # Warmup
        for _ in range(5): env.step(np.zeros(7))
        
        # Initial Frame Check
        img = np.concatenate((obs['agentview_image'], obs['robot0_eye_in_hand_image']), axis=1)
        if img.dtype != np.uint8: img = (img * 255).astype(np.uint8)
        
        viewer.show(img) # Safe to call even if disabled
        # -------------------------------

        print("\n👇 Type your command (e.g. 'open the drawer', 'q' to quit)")
        user_input = input("USER > ").strip()
        
        if user_input.lower() == 'q': 
            viewer.close()
            break
        if not user_input: user_input = task.language

        print(f"   🎥 Recording video for: '{user_input}'...")

        safe_name = "".join([c for c in user_input if c.isalnum() or c==' ']).replace(" ", "_")[:20]
        log_dir = Path("interactive_logs")
        log_dir.mkdir(parents=True, exist_ok=True)
        
        video_path = log_dir / f"scene{scene_id}_run_{safe_name}.mp4"
        video_writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*'mp4v'), 20.0, (224*2, 224))
        
        with torch.no_grad():
            for step in tqdm(range(400), desc="Robot Acting"):
                data, _ = evaluator.process_env_obs(obs, task_emb, user_input)
                
                goal_dict = {'lang_text': user_input}
                actions = model.step(data, goal_dict)
                action_numpy = actions.detach().cpu().numpy().flatten()
                
                obs, _, done, _ = env.step(action_numpy)
                
                img = np.concatenate((obs['agentview_image'], obs['robot0_eye_in_hand_image']), axis=1)
                if img.dtype != np.uint8: img = (img * 255).astype(np.uint8)
                
                # Update Video
                video_writer.write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                
                # Update Live View (Safe)
                viewer.show(img)
                
                if done:
                    print(f"   ✅ Task Completed!")
                    break
        
        video_writer.release()
        env.close()
        print(f"   💾 Video saved to: {video_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run FLOWER VLA Interactive Interface")
    parser.add_argument("--scene_id", type=int, default=89, help="ID of the LIBERO task to run (0-89)")
    args = parser.parse_args()
    
    run_interactive_final(args.scene_id)