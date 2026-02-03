import sys

path = "/etc/cdi/nvidia.yaml"
output_lines = []
skipping = False
skip_indent = -1

with open(path, 'r') as f:
    for line in f:
        stripped = line.lstrip()
        # Calculate indentation (number of spaces)
        indent = len(line) - len(stripped)
        
        # 1. Detect start of a "poisoned" block
        if stripped.startswith("- hostPath:") and "nvidia-cuda-mps" in line:
            skipping = True
            skip_indent = indent
            continue # Delete this line

        # 2. logic to handle the lines INSIDE the bad block
        if skipping:
            # If this line is indented deeper than the start, it belongs to the bad block
            if indent > skip_indent:
                continue # Delete this line
            
            # If indentation returns to normal (or less), the block is over
            if indent <= skip_indent and stripped:
                skipping = False
                # Do not continue; keep this line, it's the start of the next good block
        
        # 3. Keep good lines
        output_lines.append(line)

with open(path, 'w') as f:
    f.writelines(output_lines)
print("Success: nvidia.yaml repaired and structure preserved.")
