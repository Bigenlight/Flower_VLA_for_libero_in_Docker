# run_robot.py

import sys
import os
import torch
import hydra
import cv2
import numpy as np
import logging
import argparse
import threading
import time
import json
import queue
from pathlib import Path
from omegaconf import OmegaConf
from safetensors.torch import load_file
from tqdm import tqdm
from flask import Flask, Response, request, redirect, url_for, jsonify

# Import the templates
from templates import HTML_LOBBY, HTML_STREAM

# --- FLASK & WEB SERVER ---
app = Flask(__name__)

# --- GLOBAL STATE MANAGERS ---
class SystemState:
    def __init__(self):
        self.ready = False
        self.loading_progress = "Initializing..."
        self.selected_scene_id = None
        self.selected_scene_desc = ""
        self.frame = None
        self.lock = threading.Lock()
        self.command_queue = queue.Queue()

state = SystemState()

# --- PROMPT FORMATTER (EXPOSED) ---
def generate_flower_prompt(instruction, robot="Franka Panda", action="Delta End-Effector"):
    """
    Generates the exact prompt string the FLOWER VLM expects.
    We do this manually here to ensure we know EXACTLY what the model sees.
    """
    meta_info = f"Agent Type: 1-arm {robot}, Action Space: {action}, "
    raw_prompt = f"{meta_info} Task Instruction: {instruction}"
    
    # Clean whitespace (Critical step from original code)
    clean_prompt = ' '.join(line.strip() for line in raw_prompt.split('\n')).strip()
    return clean_prompt

# --- SCENE DATABASE ---
SCENES = {
    0: "KITCHEN_SCENE10: close the top drawer of the cabinet",
    1: "KITCHEN_SCENE10: close the top drawer of the cabinet and put the black bowl on top of it",
    2: "KITCHEN_SCENE10: put the black bowl in the top drawer of the cabinet",
    3: "KITCHEN_SCENE10: put the butter at the back in the top drawer of the cabinet and close it",
    4: "KITCHEN_SCENE10: put the butter at the front in the top drawer of the cabinet and close it",
    5: "KITCHEN_SCENE10: put the chocolate pudding in the top drawer of the cabinet and close it",
    6: "KITCHEN_SCENE1: open the bottom drawer of the cabinet",
    7: "KITCHEN_SCENE1: open the top drawer of the cabinet",
    8: "KITCHEN_SCENE1: open the top drawer of the cabinet and put the bowl in it",
    9: "KITCHEN_SCENE1: put the black bowl on the plate",
    10: "KITCHEN_SCENE1: put the black bowl on top of the cabinet",
    11: "KITCHEN_SCENE2: open the top drawer of the cabinet",
    12: "KITCHEN_SCENE2: put the black bowl at the back on the plate",
    13: "KITCHEN_SCENE2: put the black bowl at the front on the plate",
    14: "KITCHEN_SCENE2: put the middle black bowl on the plate",
    15: "KITCHEN_SCENE2: put the middle black bowl on top of the cabinet",
    16: "KITCHEN_SCENE2: stack the black bowl at the front on the black bowl in the middle",
    17: "KITCHEN_SCENE2: stack the middle black bowl on the back black bowl",
    18: "KITCHEN_SCENE3: put the frying pan on the stove",
    19: "KITCHEN_SCENE3: put the moka pot on the stove",
    20: "KITCHEN_SCENE3: turn on the stove",
    21: "KITCHEN_SCENE3: turn on the stove and put the frying pan on it",
    22: "KITCHEN_SCENE4: close the bottom drawer of the cabinet",
    23: "KITCHEN_SCENE4: close the bottom drawer of the cabinet and open the top drawer",
    24: "KITCHEN_SCENE4: put the black bowl in the bottom drawer of the cabinet",
    25: "KITCHEN_SCENE4: put the black bowl on top of the cabinet",
    26: "KITCHEN_SCENE4: put the wine bottle in the bottom drawer of the cabinet",
    27: "KITCHEN_SCENE4: put the wine bottle on the wine rack",
    28: "KITCHEN_SCENE5: close the top drawer of the cabinet",
    29: "KITCHEN_SCENE5: put the black bowl in the top drawer of the cabinet",
    30: "KITCHEN_SCENE5: put the black bowl on the plate",
    31: "KITCHEN_SCENE5: put the black bowl on top of the cabinet",
    32: "KITCHEN_SCENE5: put the ketchup in the top drawer of the cabinet",
    33: "KITCHEN_SCENE6: close the microwave",
    34: "KITCHEN_SCENE6: put the yellow and white mug to the front of the white mug",
    35: "KITCHEN_SCENE7: open the microwave",
    36: "KITCHEN_SCENE7: put the white bowl on the plate",
    37: "KITCHEN_SCENE7: put the white bowl to the right of the plate",
    38: "KITCHEN_SCENE8: put the right moka pot on the stove",
    39: "KITCHEN_SCENE8: turn off the stove",
    40: "KITCHEN_SCENE9: put the frying pan on the cabinet shelf",
    41: "KITCHEN_SCENE9: put the frying pan on top of the cabinet",
    42: "KITCHEN_SCENE9: put the frying pan under the cabinet shelf",
    43: "KITCHEN_SCENE9: put the white bowl on top of the cabinet",
    44: "KITCHEN_SCENE9: turn on the stove",
    45: "KITCHEN_SCENE9: turn on the stove and put the frying pan on it",
    46: "LIVING_ROOM_SCENE1: pick up the alphabet soup and put it in the basket",
    47: "LIVING_ROOM_SCENE1: pick up the cream cheese box and put it in the basket",
    48: "LIVING_ROOM_SCENE1: pick up the ketchup and put it in the basket",
    49: "LIVING_ROOM_SCENE1: pick up the tomato sauce and put it in the basket",
    50: "LIVING_ROOM_SCENE2: pick up the alphabet soup and put it in the basket",
    51: "LIVING_ROOM_SCENE2: pick up the butter and put it in the basket",
    52: "LIVING_ROOM_SCENE2: pick up the milk and put it in the basket",
    53: "LIVING_ROOM_SCENE2: pick up the orange juice and put it in the basket",
    54: "LIVING_ROOM_SCENE2: pick up the tomato sauce and put it in the basket",
    55: "LIVING_ROOM_SCENE3: pick up the alphabet soup and put it in the tray",
    56: "LIVING_ROOM_SCENE3: pick up the butter and put it in the tray",
    57: "LIVING_ROOM_SCENE3: pick up the cream cheese and put it in the tray",
    58: "LIVING_ROOM_SCENE3: pick up the ketchup and put it in the tray",
    59: "LIVING_ROOM_SCENE3: pick up the tomato sauce and put it in the tray",
    60: "LIVING_ROOM_SCENE4: pick up the black bowl on the left and put it in the tray",
    61: "LIVING_ROOM_SCENE4: pick up the chocolate pudding and put it in the tray",
    62: "LIVING_ROOM_SCENE4: pick up the salad dressing and put it in the tray",
    63: "LIVING_ROOM_SCENE4: stack the left bowl on the right bowl and place them in the tray",
    64: "LIVING_ROOM_SCENE4: stack the right bowl on the left bowl and place them in the tray",
    65: "LIVING_ROOM_SCENE5: put the red mug on the left plate",
    66: "LIVING_ROOM_SCENE5: put the red mug on the right plate",
    67: "LIVING_ROOM_SCENE5: put the white mug on the left plate",
    68: "LIVING_ROOM_SCENE5: put the yellow and white mug on the right plate",
    69: "LIVING_ROOM_SCENE6: put the chocolate pudding to the left of the plate",
    70: "LIVING_ROOM_SCENE6: put the chocolate pudding to the right of the plate",
    71: "LIVING_ROOM_SCENE6: put the red mug on the plate",
    72: "LIVING_ROOM_SCENE6: put the white mug on the plate",
    73: "STUDY_SCENE1: pick up the book and place it in the front compartment of the caddy",
    74: "STUDY_SCENE1: pick up the book and place it in the left compartment of the caddy",
    75: "STUDY_SCENE1: pick up the book and place it in the right compartment of the caddy",
    76: "STUDY_SCENE1: pick up the yellow and white mug and place it to the right of the caddy",
    77: "STUDY_SCENE2: pick up the book and place it in the back compartment of the caddy",
    78: "STUDY_SCENE2: pick up the book and place it in the front compartment of the caddy",
    79: "STUDY_SCENE2: pick up the book and place it in the left compartment of the caddy",
    80: "STUDY_SCENE2: pick up the book and place it in the right compartment of the caddy",
    81: "STUDY_SCENE3: pick up the book and place it in the front compartment of the caddy",
    82: "STUDY_SCENE3: pick up the book and place it in the left compartment of the caddy",
    83: "STUDY_SCENE3: pick up the book and place it in the right compartment of the caddy",
    84: "STUDY_SCENE3: pick up the red mug and place it to the right of the caddy",
    85: "STUDY_SCENE3: pick up the white mug and place it to the right of the caddy",
    86: "STUDY_SCENE4: pick up the book in the middle and place it on the cabinet shelf",
    87: "STUDY_SCENE4: pick up the book on the left and place it on top of the shelf",
    88: "STUDY_SCENE4: pick up the book on the right and place it on the cabinet shelf",
    89: "STUDY_SCENE4: pick up the book on the right and place it under the cabinet shelf"
}

# --- FLASK ROUTES ---
@app.route("/")
def index():
    from flask import render_template_string
    return render_template_string(HTML_LOBBY, ready=state.ready, progress=state.loading_progress, scenes=SCENES)

@app.route("/start/<int:scene_id>")
def start_scene(scene_id):
    if not state.ready: return redirect(url_for('index'))
    state.selected_scene_id = scene_id
    state.selected_scene_desc = SCENES.get(scene_id, "Unknown Task")
    return redirect(url_for('stream_page'))

@app.route("/stream")
def stream_page():
    from flask import render_template_string
    if state.selected_scene_id is None: return redirect(url_for('index'))
    return render_template_string(HTML_STREAM, desc=state.selected_scene_desc)

@app.route("/submit_command", methods=['POST'])
def submit_command():
    data = request.json
    cmd = data.get('command', '')
    max_steps = int(data.get('max_steps', 400)) # Default if missing
    
    if cmd:
        state.command_queue.put((cmd, max_steps))
        return jsonify({"status": "ok", "received": cmd, "steps": max_steps})
    return jsonify({"status": "empty"})

def gen_frames():
    # Send blank frame if nothing is ready
    blank = np.zeros((224, 448, 3), dtype=np.uint8)
    _, enc = cv2.imencode(".jpg", blank)
    blank_bytes = b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + bytearray(enc) + b'\r\n'
    
    while True:
        with state.lock:
            if state.frame is None: 
                yield blank_bytes
                time.sleep(0.1)
                continue
            
            flag, encoded = cv2.imencode(".jpg", state.frame)
            if not flag: continue
        
        yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encoded) + b'\r\n')
        time.sleep(0.04)

@app.route("/video_feed")
def video_feed():
    return Response(gen_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")

# --- PATH FIXER ---
current_path = Path(__file__).parent.absolute()
libero_path = current_path / "LIBERO"
if libero_path.exists(): sys.path.insert(0, str(libero_path))
sys.path.insert(0, str(current_path))
# ------------------

from flower.evaluation.flower_eval_libero import EvaluateLibero
logging.getLogger("flower").setLevel(logging.ERROR)

def run_web_server():
    cli = sys.modules['flask.cli']
    cli.show_server_banner = lambda *x: None
    app.logger.setLevel(logging.ERROR)
    logging.getLogger('werkzeug').setLevel(logging.ERROR)
    app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False)

def main_logic():
    print("\n🤖 FLOWER VLA: Web Controller Launching...")
    
    # 1. Start Web Thread
    web_thread = threading.Thread(target=run_web_server, daemon=True)
    web_thread.start()
    print("   🌐 Go to http://localhost:5000 to manage the robot.")

    # 2. Heavy Loading
    state.loading_progress = "Loading Configuration..."
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt_path = Path("checkpoints/flower_libero_90")
    config_path = ckpt_path / "config.yaml"
    
    if not config_path.exists():
        print("❌ Config not found!")
        return

    print("⏳ Loading Brain... (This takes a few minutes)")
    state.loading_progress = "Loading Brain Model (Wait ~2 mins)..."
    
    cfg = OmegaConf.load(config_path)
    # Patches
    if "lang_folder" not in cfg.datamodule.datasets.lang_dataset:
        cfg.datamodule.datasets.lang_dataset.lang_folder = "dummy_path"
    if "env_cfg" not in cfg.callbacks.rollout_lh:
         cfg.callbacks.rollout_lh.env_cfg = {"_target_": "flower.wrappers.hulc_wrapper.HulcWrapper"}

    model = hydra.utils.instantiate(cfg.model)
    state_dict = load_file(ckpt_path / "model.safetensors")
    if "state_dict" in state_dict: state_dict = state_dict["state_dict"]
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()

    # --- CRITICAL FIX: DISABLE INTERNAL FORMATTING ---
    # We will format the prompt manually to ensure we know exactly what we are sending.
    # This monkey-patch prevents the model from adding "Agent Type..." a second time.
    print("   🔧 Monkey-patching model to disable hidden prompt formatting...")
    model.format_instruction = lambda x: x
    # -------------------------------------------------

    state.loading_progress = "Setting up World Environment..."
    print("🌍 Setting up World...")
    
    val_transforms = hydra.utils.instantiate(cfg.datamodule.transforms.val)
    transforms = {'val': val_transforms}
    
    evaluator = EvaluateLibero(
        model=model, transforms=transforms, log_dir=Path("interactive_logs"),
        benchmark_name="libero_90", num_sequences=1, max_steps=600,
        num_videos=1, n_eval=1, task_embedding_format="clip", device=device
    )

    # 3. System Ready
    state.loading_progress = "Ready!"
    state.ready = True
    print("\n✅ System Ready! Waiting for Scene Selection on Web Interface...")

    # 4. Main Event Loop
    while True:
        # Wait for selection from Web
        while state.selected_scene_id is None:
            time.sleep(0.5)
        
        scene_id = state.selected_scene_id
        
        # --- SCENE START ---
        try:
            task = evaluator.benchmark_instance.get_task(scene_id)
            # We fetch task_emb just to initialize scene, but we WON'T use it for inference.
            task_emb = evaluator.benchmark_instance.task_embs[scene_id]
        except:
            print(f"❌ Error loading scene {scene_id}")
            state.selected_scene_id = None
            continue

        print("\n" + "="*50)
        print(f"🚀 Launching Scene [{scene_id}]: {task.name}")
        print("="*50)
        
        env_args = {
            "bddl_file_name": os.path.join(evaluator.bddl_folder, task.problem_folder, task.bddl_file),
            "camera_heights": 224, "camera_widths": 224, "render_gpu_device_id": 0
        }
        
        from libero.libero.envs import OffScreenRenderEnv
        env = OffScreenRenderEnv(**env_args)
        
        # INITIAL RESET TO SHOW FIRST FRAME
        try:
            env.reset()
            init_states = evaluator.benchmark_instance.get_task_init_states(scene_id)
            obs = env.set_init_state(init_states[0])
        except:
            obs = env.reset()
            
        # Initial Frame update
        img = np.concatenate((obs['agentview_image'], obs['robot0_eye_in_hand_image']), axis=1)
        if img.dtype != np.uint8: img = (img * 255).astype(np.uint8)
        
        with state.lock:
            state.frame = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        print(f"\n⚡ Waiting for Web Command...")
        
        # Interactive Loop
        while True:
            # WAIT FOR COMMAND (from Web Queue)
            try:
                # Get tuple: (command, max_steps)
                cmd_data = state.command_queue.get(timeout=0.5)
                raw_input = cmd_data[0]
                max_steps = cmd_data[1]
                
            except queue.Empty:
                continue # Keep waiting
                
            if not raw_input: raw_input = task.language

            # --- PREPARE PROMPT ---
            # 1. Format the raw input into the FLOWER template
            final_prompt = generate_flower_prompt(raw_input)
            print(f"   📨 Received Raw: '{raw_input}'")
            print(f"   📝 Formatted to: '{final_prompt}'") # DEBUG: See exactly what goes in
            
            # --- CRITICAL FIX: HARD RESET BEFORE SCENE SETUP ---
            print(f"   🔄 Resetting environment for new action...")
            
            env.reset() 
            
            try:
                init_states = evaluator.benchmark_instance.get_task_init_states(scene_id)
                obs = env.set_init_state(init_states[0])
            except:
                obs = env.reset()
            
            for _ in range(5): env.step(np.zeros(7))
            # ---------------------------------------------------

            print(f"   🎥 Recording video...")

            safe_name = "".join([c for c in raw_input if c.isalnum() or c==' ']).replace(" ", "_")[:20]
            log_dir = Path("interactive_logs")
            log_dir.mkdir(parents=True, exist_ok=True)
            
            video_path = log_dir / f"scene{scene_id}_run_{safe_name}.mp4"
            video_writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*'mp4v'), 20.0, (224*2, 224))
            
            with torch.no_grad():
                for step in tqdm(range(max_steps), desc=f"Robot Acting"):


                    # 1. Capture observation
                    # If using the standard Libero/Robosuite env, this might be float or uint8
                    agent_view = obs['agentview_image']
                    wrist_view = obs['robot0_eye_in_hand_image']

                    # --- CRITICAL FIX: Ensure Model Sees the Image ---
                    # If images are float (0-1), scale to 0-255 and cast to uint8.
                    # If we don't do this, model sees all ZEROs.
                    if agent_view.dtype != np.uint8:
                        if agent_view.max() <= 1.0:
                            agent_view = (agent_view * 255).astype(np.uint8)
                        else:
                            agent_view = agent_view.astype(np.uint8)
                            
                    if wrist_view.dtype != np.uint8:
                        if wrist_view.max() <= 1.0:
                            wrist_view = (wrist_view * 255).astype(np.uint8)
                        else:
                            wrist_view = wrist_view.astype(np.uint8)

                    # Update the dict before passing to evaluator
                    obs['agentview_image'] = agent_view
                    obs['robot0_eye_in_hand_image'] = wrist_view
                    # -----------------------------------------------

                    # 2. Process for Model
                    # Now 'data' will contain valid image tensors
                    data, _ = evaluator.process_env_obs(obs, task_emb, final_prompt)

                    # 3. Manual Prompting (Bypassing Embeddings)
                    # We recreate the dictionary manually to ensure the model re-encodes 
                    # the text on-the-fly using the "Agent Type..." prompt.
                    goal_dict = {'lang_text': final_prompt}

                    # 4. Step Model
                    actions = model.step(data, goal_dict)
                    action_numpy = actions.detach().cpu().numpy().flatten()

                    # 5. Execute Action
                    # Note: If the robot moves too fast/slow, you might need to scale this action_numpy.
                    # Libero usually handles [-1, 1], but check if un-normalization is needed.
                    obs, _, done, _ = env.step(action_numpy)
                    
                    img = np.concatenate((obs['agentview_image'], obs['robot0_eye_in_hand_image']), axis=1)
                    if img.dtype != np.uint8: img = (img * 255).astype(np.uint8)
                    
                    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                    
                    video_writer.write(img_bgr)
                    
                    with state.lock:
                        state.frame = img_bgr
                    
                    if done:
                        print(f"   ✅ Task Completed!")
                        break
            
            video_writer.release()
            print(f"   💾 Video saved to: {video_path}")
            print(f"   ⚡ Ready for next command...")
        
        env.close()

if __name__ == "__main__":
    main_logic()