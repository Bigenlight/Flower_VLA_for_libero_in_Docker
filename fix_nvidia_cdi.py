import sys
import yaml

input_file = "/etc/cdi/nvidia.yaml"

try:
    with open(input_file, 'r') as f:
        data = yaml.safe_load(f)

    # 1. Fix global edits
    if 'containerEdits' in data and 'mounts' in data['containerEdits']:
        original_count = len(data['containerEdits']['mounts'])
        data['containerEdits']['mounts'] = [
            m for m in data['containerEdits']['mounts']
            if 'mps-control' not in m.get('hostPath', '') and 'mps-server' not in m.get('hostPath', '')
        ]
        print(f"Removed {original_count - len(data['containerEdits']['mounts'])} global MPS mounts.")

    # 2. Fix per-device edits (just in case)
    if 'devices' in data:
        for device in data['devices']:
            if 'containerEdits' in device and 'mounts' in device['containerEdits']:
                mounts = device['containerEdits']['mounts']
                device['containerEdits']['mounts'] = [
                    m for m in mounts
                    if 'mps-control' not in m.get('hostPath', '') and 'mps-server' not in m.get('hostPath', '')
                ]

    # 3. Save back
    with open(input_file, 'w') as f:
        yaml.dump(data, f, default_flow_style=False)
    print("Success: nvidia.yaml has been sanitized.")

except Exception as e:
    print(f"Error: {e}")
