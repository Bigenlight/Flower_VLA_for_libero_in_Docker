# Using FlowerVLA on Libero with Docker

Refer the original page for more info.

https://github.com/intuitive-robots/flower_vla_calvin


## Prerequisites

- Linux OS (Ubuntu 22.04 recommended)

- 30GB of space

- NVIDIA GPU (>3GB VRAM) with drivers installed

- Docker & NVIDIA Container Toolkit installed.

## Usage

```
git clone https://github.com/Bigenlight/Flower_VLA_for_libero_in_Docker.git

# 1. Create a workspace folder
mkdir -p checkpoints/flower_libero_90
mkdir -p interactive_logs

# 2. Go into the folder
cd flower_vla_workspace
```

- Pull image

```
docker pull bigenlight/flower_vla:v5
```

- Run image

```
docker run -itd \
  --name flower_vla \
  --device nvidia.com/gpu=0 \
  -v $(pwd)/checkpoints:/app/checkpoints \
  -v $(pwd)/interactive_logs:/app/interactive_logs \
  bigenlight/flower_vla:v5 \
  /bin/bash
```

- Enter the container.

```
docker exec -it flower_vla /bin/bash
```

- Inside container, run the code for launching Liberp inference.

```
python run_robot.py --scene_id 89 # choose scene id from 0 to 89
```

## Citation (from original page)

If you found the code usefull, please cite our work: (arxiv coming very soon)

```bibtex
@inproceedings{
reuss2025flower,
title={{FLOWER}: Democratizing Generalist Robot Policies with Efficient Vision-Language-Flow Models},
author={Moritz Reuss and Hongyi Zhou and Marcel R{\"u}hle and {\"O}mer Erdin{\c{c}} Ya{\u{g}}murlu and Fabian Otto and Rudolf Lioutikov},
booktitle={9th Annual Conference on Robot Learning},
year={2025},
url={https://openreview.net/forum?id=JeppaebLRD}
}
```
