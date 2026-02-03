import sys

input_path = "/etc/cdi/nvidia.yaml"
output_lines = []
skip_block = False
block_indent = -1

with open(input_path, 'r') as f:
    for line in f:
        # Determine indentation level
        stripped = line.lstrip()
        indent = len(line) - len(stripped)
        
        # Check if this line starts a bad block
        if stripped.startswith("- hostPath:") and "nvidia-cuda-mps" in line:
            skip_block = True
            block_indent = indent
            continue # Skip this line

        # If we are currently skipping a block...
        if skip_block:
            # If indentation is deeper, it's part of the block -> Skip it
            if indent > block_indent:
                continue
            # If indentation matches or is shallower, the block is over
            # But we must be careful: is it a new block or end of section?
            # Safe bet: if it's a new list item (same indent), stop skipping
            if indent <= block_indent and stripped:
                skip_block = False

        # If not skipping, keep the line
        if not skip_block:
            output_lines.append(line)

# Write the clean file back
with open(input_path, 'w') as f:
    f.writelines(output_lines)
print("Sanitization complete.")
