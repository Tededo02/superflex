<p align="center">
 <h1 align="center">SuperFlex: Deformable Superquadrics for Point Cloud Decomposition</h2></h1>
<p align="center">
<a href="https://gabrieltavernini.github.io/">Gabriel Tavernini</a><sup>1*</sup>,
<a href="https://elisabettafedele.github.io/">Elisabetta Fedele</a><sup>1*</sup>,
<a href="https://sites.google.com/site/tiagonovellodebrito">Tiago Novello</a><sup>2,3</sup>,
<a href="https://geometry.stanford.edu/?member=guibas">Leonidas Guibas</a><sup>2</sup>,
<a href="https://people.inf.ethz.ch/pomarc/">Marc Pollefeys</a><sup>1</sup>,
<a href="https://francisengelmann.github.io/">Francis Engelmann</a><sup>4</sup>
<br>
<sup>1</sup>ETH Zurich,
<sup>2</sup>Stanford University,
<sup>3</sup>IMPA
<sup>4</sup>USI Lugano <br>
</p>
<h2 align="center">ECCV 2026</h2>
<h3 align="center"><a href="https://github.com/gabrieltavernini/superflex">Code</a> | <a href="https://arxiv.org/abs/2607.01015">Paper</a> | <a href="https://superflex3d.github.io">Project Page</a> </h3>
<div align="center"></div>
</p>
<p align="center"> 
<a href="">
<img src="https://superflex3d.github.io/static/figures/teaser.jpg" alt="Logo" width="100%">
</a>
</p>
<p align="center">
<strong>SuperFlex</strong> expands the expressive power and applicability of superquadric decompositions by utilizing tapering and bending.
</p>
<br>


## 🚀 Quick Start

### Environment Setup

For CPU or Apple Silicon inference, install the lightweight environment. The demo
uses native PyTorch voxelization and interpolation on CPU/MPS, so neither CUDA nor
`torch-cluster` nor the training sampler is needed:

```bash
git clone --branch metal-support https://github.com/Tededo02/superflex.git
cd superflex

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install inference dependencies
pip install -r requirements-inference.txt
pip install -e .
```

For the full training environment, also run `pip install -r requirements.txt` and
`python setup_sampler.py build_ext --inplace`. CUDA inference retains the original
PVCNN extension and requires a working CUDA toolkit and `ninja`.

### Download Pre-trained Models

Download the checkpoints:

```bash
bash scripts/download_checkpoints.sh
```

Alternatively, you can download the individual folders using the links below.

| Dataset | Deformations | Robust | Supervised | Link |
|:--------|:--------:|:-----------:|:-----:|:-----|
| ShapeNet | ❌ | ❌ | ❌ | [base](https://drive.google.com/drive/folders/1yz4rwho0rsQGb5pEQcPIucZUjohOlfBN?usp=sharing) |
| ShapeNet | ✅ | ❌ | ❌ | [superflex](https://drive.google.com/drive/folders/1Ospu8mFz4kEQf5a6pnOOx87z-xWR6kQe?usp=sharing) |
| ShapeNet + ASE | ✅ | ✅ | ❌ |[robust unsup.](https://drive.google.com/drive/folders/1bw-HHwQYKC62uUHF9t8QVdx_XVibESQb?usp=sharing) |
| ShapeNet + ASE | ✅ | ✅ | ✅ |[robust](https://drive.google.com/drive/folders/1cHbtZLToQRontte6CyzvQpL5IHUqk-x4?usp=sharing) |

### Inference Example
Once downloaded, run the robust checkpoint on the included lamp using CPU:
```bash
python demo_viser.py --device cpu
```

On an Apple Silicon Mac, select its GPU with MPS:

```bash
python demo_viser.py --device mps
```

`--device auto` prefers CUDA, then MPS, then CPU. The default is CPU. An explicitly
requested unavailable device produces an error. MPS uses float32 native PyTorch
operations; unsupported operators are reported instead of silently switching to CPU.

Open the local Viser URL printed in the terminal (normally
`http://127.0.0.1:8080`) and press Ctrl+C in the terminal to stop it.
Predictions are saved to `outputs/test.npz`. Mesh extraction and visualization
run on CPU even when model inference uses the GPU.

Use your own single-object point cloud or mesh without editing the script:

```bash
python demo_viser.py --device mps --input /path/to/object.ply --z-up
python demo_viser.py --device cpu --input examples/chair.glb
```

Omit `--z-up` for Y-up inputs. Set `NUM_POINTS` near the top of `demo_viser.py`
to change the inference point count (default: 10000). Any positive integer can be
used; multiples of 4096 are not required. Point clouds are randomly resampled to
that count (with replacement for smaller clouds); meshes are sampled on their surfaces.
Inputs are centered and scaled by default. `--seed` makes resampling repeatable.
Both predictions and displayed points are restored to the original input frame.

To save results and exit without a viewer:

```bash
python demo_viser.py --device cpu --no-viewer --output outputs/lamp_cpu.npz --mesh-output outputs/lamp_cpu.glb
```

Use `--checkpoint-dir checkpoints/superflex` to select another checkpoint,
`--port 8081` to change the viewer port, or `python demo_viser.py --help` to list
all options. Run numerical regression checks with
`python -m unittest discover -s tests -v` (MPS checks skip when unavailable).
Use `--no-viewer --benchmark-runs 10` to measure steady-state inference; first-use
GPU initialization can make the initial inference slower.

<p align="center">
  <img src="https://superflex3d.github.io/static/figures/viser/demo0.jpg" width="32%" />
  <img src="https://superflex3d.github.io/static/figures/viser/demo1.jpg" width="32%" />
  <img src="https://superflex3d.github.io/static/figures/viser/demo2.jpg" width="32%" />
</p>

### Optimization Example
You can run an optimization example by doing:
```bash
python demo_optimization.py
```

<p align="center">
  <img src="https://superflex3d.github.io/static/figures/viser/optimization_start.jpg" width="49%" />
  <img src="https://superflex3d.github.io/static/figures/viser/optimization_end.jpg" width="49%" />
</p>

## 🪑 Inference on ShapeNet 

### Download Data

Download the ShapeNet dataset (73.4 GB):

```bash
bash scripts/download_shapenet.sh
```

The dataset will be saved to `data/ShapeNet/`. After having downloaded ShapeNet and the checkpoints, the following project structure is expected:
```
superflex/
├── checkpoints/          # Checkpoints storage
│   ├── unsup_robust/     # Checkpoint and config for unsup_robust
│   ├── robust/           # Checkpoint and config for robust
│   └── superflex/        # Checkpoint and config for superflex
├── data/                 # Dataset storage
│   └── ShapeNet/         # ShapeNet dataset
├── examples/             # Inference example
│   └── chair.glb         # ABO chair mesh
│   └── lamp.ply          # ShapeNet lamp pointcloud
├── scripts/              # Utility scripts
├── superflex/            # Main package
├── train/                # Training scripts
└── requirements.txt      # Dependencies
```
### Inference and visualization on test set

Generate and visualize results on ShapeNet test set:
```bash
bash scripts/run_on_shapenet.sh 
```
> **Note:** Saving the .npz file and mesh generation may take time depending on the size of the dataset and of the chosen resolution for the superquadrics, respectively.

### Optimization

Optimize and visualize results on ShapeNet test set:
```bash
bash scripts/run_optimization.sh 
```

### Training (optional)

If you want to retrain the network yourself you can either opt for single or multi-gpu training as follows.

**Single GPU training:**
```bash
python train/train.py
```

**Multi-GPU training (4 GPUs):**
```bash
torchrun --nproc_per_node=4 train/train.py
```
> **Note:** Weights & Biases is disabled by default but you can activate it in the [training config](configs/train.yaml).



## 🏡 Inference on Full Scenes 
We assume you have the .ply files of all the segmented objects in a single folder OBJECTS_SCENE_DIR. Fill required fields in the [script](scripts/run_on_scene.sh), following the given instructions. Now you are ready to run inference by doing:
```bash
bash scripts/run_on_scene.sh 
```

## 🙏  Acknowledgements
We adapted some codes from some awesome repositories including [superquadric_parsing](https://github.com/paschalidoud/superquadric_parsing), [CuboidAbstractionViaSeg](https://github.com/SilenKZYoung/CuboidAbstractionViaSeg), [volumentations](https://github.com/kumuji/volumentations), [LION](https://github.com/nv-tlabs/LION), [occupancy_networks](https://github.com/autonomousvision/occupancy_networks), and [convolutional_occupancy_networks](https://github.com/autonomousvision/convolutional_occupancy_networks). Thanks for making codes and data public available.

## 🤝 Contributing

We welcome contributions! Please feel free to submit issues, feature requests, or pull requests. For more specific questions or collaborations, please contact [Gabriel](mailto:gtavernini@ethz.ch) and [Elisabetta](mailto:efedele@ethz.ch).
