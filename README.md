# FlowerVLA on Libero (Docker)

This is a Dockerized implementation of FlowerVLA for the Libero env inference.

![alt text](sample.gif)

Original Repository: https://github.com/intuitive-robots/flower_vla_calvin

Currently using the **FLOWER VLA libero_90** model: https://huggingface.co/mbreuss/flower_libero_90

## Prerequisites

- **OS:** Linux (Ubuntu 20.04/22.04 recommended)
- **Storage:** At least **44GB** of free disk space
- **GPU:** NVIDIA GPU (>8GB VRAM recommended) with drivers installed
- **Software:** Docker & [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) installed.

## Installisation and usage

You don't need to clone the code (it's all inside the Docker image), but you need a directory for the output videos.

```
git clone https://github.com/Bigenlight/Flower_VLA_for_libero_in_Docker.git

cd Flower_VLA_for_libero_in_Docker

mkdir -p interactive_logs
```

- Pull image (download may take time)

```
docker pull bigenlight/flower_vla:v6
```

- Start the Robot Server

```
docker run -itd \
  --name flower_vla \
  --gpus all \
  -v $(pwd)/interactive_logs:/app/interactive_logs \
  bigenlight/flower_vla:v6 \
  /bin/bash
```

- Enter the container

```
docker exec -it flower_vla /bin/bash
```

- Inside container, run the code for launching Libero inference

```
python run_robot.py --scene_id 89 # You can choose any scene_id from 0 to 89
```

> Note: The robot runs in Silent Mode (no GUI window) when inside a container. Once the task is finished, check the interactive_logs/ folder on your host machine to watch the generated .mp4 video.

## Citation (from original page)

If you found the code usefull, please cite our work:

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
