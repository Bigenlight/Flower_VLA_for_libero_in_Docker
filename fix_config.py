import os
import yaml
from pathlib import Path

def fix_flower_config():
    # Target directory
    ckpt_dir = Path("checkpoints/flower_libero_90")
    
    # 1. Identify the file (json or yaml)
    json_path = ckpt_dir / "config.json"
    yaml_path = ckpt_dir / "config.yaml"
    
    target_file = None
    
    if json_path.exists():
        print(f"Found config.json at {json_path}")
        target_file = json_path
    elif yaml_path.exists():
        print(f"Found config.yaml at {yaml_path}")
        target_file = yaml_path
    else:
        print("❌ Error: Could not find config.json or config.yaml in flower_libero_90")
        return

    # 2. Read the content
    print("Reading configuration...")
    with open(target_file, 'r') as f:
        # We use safe_load, but since the file might have '!!python/object' tags, 
        # we treat it as plain text to avoid import errors, 
        # or use UnsafeLoader if pyyaml allows.
        # Safer strategy: Read as text, replace string, write back.
        content = f.read()

    # 3. Apply Fixes
    # Fix A: Disable loading the developer's local path
    if "load_pretrained: true" in content:
        print("🔧 Patching: Setting load_pretrained to false")
        content = content.replace("load_pretrained: true", "load_pretrained: false")
    else:
        print("INFO: load_pretrained is already false or missing.")

    # 4. Save as config.yaml
    print(f"💾 Saving fixed config to {yaml_path}...")
    with open(yaml_path, 'w') as f:
        f.write(content)
        
    # 5. Cleanup
    if target_file == json_path:
        print("🧹 Removing old config.json...")
        os.remove(json_path)

    print("✅ Success! Config is fixed.")

if __name__ == "__main__":
    fix_flower_config()