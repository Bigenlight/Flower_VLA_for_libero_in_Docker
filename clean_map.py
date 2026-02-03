lines = []
skip = False

with open("/etc/cdi/nvidia.yaml", "r") as f:
    for line in f:
        # If we find the start of an MPS block, start skipping
        if "- hostPath:" in line and "nvidia-cuda-mps" in line:
            skip = True
        
        # If we are skipping, check if we hit the NEXT valid block (stops skipping)
        if skip and "- hostPath:" in line and "nvidia-cuda-mps" not in line:
            skip = False

        # If we are not skipping, keep the line
        if not skip:
            lines.append(line)

with open("/etc/cdi/nvidia.yaml", "w") as f:
    f.writelines(lines)

print("Cleaned nvidia.yaml: All MPS blocks removed.")
