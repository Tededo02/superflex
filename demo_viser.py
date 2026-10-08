import argparse
from pathlib import Path
import random
import time

import numpy as np
from omegaconf import OmegaConf
import torch
import trimesh

from superflex.superflex import SuperFlex
from superflex.utils.predictions_handler import PredictionHandler
from superflex.data.normalization import normalize_points, denormalize_outdict, denormalize_points
from superflex.data.transform import rotate_around_axis


# Change this positive integer to set the number of input points for inference.
NUM_POINTS = 10000


def parse_args():
    parser = argparse.ArgumentParser(description="SuperFlex inference on CPU, Apple MPS, or CUDA.")
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints/robust"))
    parser.add_argument("--checkpoint-file", default="ckpt.pt")
    parser.add_argument("--input", type=Path, default=Path("examples/lamp.ply"))
    parser.add_argument("--device", choices=("cpu", "mps", "cuda", "auto"), default="cpu")
    parser.add_argument("--z-up", action="store_true", help="Input uses Z as the vertical axis.")
    parser.add_argument("--no-normalize", action="store_true")
    parser.add_argument("--resolution", type=int, default=30, help="Mesh extraction resolution (at least 3).")
    parser.add_argument("--output", type=Path, default=Path("outputs/test.npz"))
    parser.add_argument("--mesh-output", type=Path, help="Optionally export the predicted mesh, e.g. outputs/test.glb.")
    parser.add_argument("--no-viewer", action="store_true", help="Save predictions and exit without starting Viser.")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--benchmark-runs", type=int, default=0, help="Measure steady-state inference after two warmup runs.")
    args = parser.parse_args()
    if args.resolution < 3:
        parser.error("--resolution must be at least 3")
    if args.benchmark_runs < 0:
        parser.error("--benchmark-runs cannot be negative")
    return args


def select_device(requested):
    if requested == "auto":
        if torch.cuda.is_available():
            requested = "cuda"
        elif torch.backends.mps.is_available():
            requested = "mps"
        else:
            requested = "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Use --device cpu or --device mps on an Apple Silicon Mac.")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("Apple MPS is unavailable in this process. Check your PyTorch/macOS setup or use --device cpu.")
    return torch.device(requested)


def load_points(path, seed):
    if not isinstance(NUM_POINTS, int) or NUM_POINTS <= 0:
        raise ValueError("NUM_POINTS must be a positive integer.")
    if not path.is_file():
        raise FileNotFoundError(f"Input not found: {path}")
    geometry = trimesh.load(path, process=False)
    if isinstance(geometry, trimesh.Scene):
        geometry = geometry.to_geometry()
    if not hasattr(geometry, "vertices"):
        raise ValueError(f"Input must contain a point cloud or mesh: {path}")
    vertices = np.asarray(geometry.vertices)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) == 0:
        raise ValueError("Input must contain a nonempty array of XYZ coordinates.")
    if not np.isfinite(vertices).all():
        raise ValueError("Input contains nonfinite XYZ coordinates. Clean the point cloud first.")
    if isinstance(geometry, trimesh.Trimesh) and len(geometry.faces):
        points, _ = trimesh.sample.sample_surface(geometry, NUM_POINTS, seed=seed)
    elif len(vertices) != NUM_POINTS:
        indices = np.random.default_rng(seed).choice(len(vertices), NUM_POINTS, replace=len(vertices) < NUM_POINTS)
        points = vertices[indices]
    else:
        points = vertices
    return points


def synchronize(device):
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize(device)


def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = select_device(args.device)
    print(f"Using device: {device} (PyTorch {torch.__version__})", flush=True)

    checkpoint_path = args.checkpoint_dir / args.checkpoint_file
    config_path = args.checkpoint_dir / "config.yaml"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration not found: {config_path}")
    config = OmegaConf.load(config_path)
    # Load on CPU first to avoid copying training state onto the GPU.
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = SuperFlex(config.superflex)
    model.load_state_dict(checkpoint["model_state_dict"])
    del checkpoint
    model = model.to(device).eval()
    print(f"Loaded checkpoint: {checkpoint_path}", flush=True)

    points = load_points(args.input, args.seed)
    if not args.no_normalize:
        points, translation, scale = normalize_points(points)
    else:
        translation, scale = np.zeros(3), 1.0
    if args.z_up:
        points = rotate_around_axis(points, axis=(1, 0, 0), angle=-np.pi / 2, center_point=np.zeros(3))
    points = torch.from_numpy(np.asarray(points, dtype=np.float32)).unsqueeze(0).to(device)

    synchronize(device)
    start = time.perf_counter()
    with torch.inference_mode():
        outdict = model(points)
        synchronize(device)
        elapsed = time.perf_counter() - start
        if args.benchmark_runs:
            for _ in range(2):
                model(points)
            synchronize(device)
            timings = []
            for _ in range(args.benchmark_runs):
                benchmark_start = time.perf_counter()
                model(points)
                synchronize(device)
                timings.append(time.perf_counter() - benchmark_start)
            print(f"Steady-state inference: {np.median(timings):.3f} s median over {args.benchmark_runs} runs (2 warmups)", flush=True)
        outdict = {key: value.cpu() if isinstance(value, torch.Tensor) else value for key, value in outdict.items()}
        outdict = denormalize_outdict(outdict, np.array([translation]), np.array([scale]), args.z_up)
        points = denormalize_points(points.cpu(), np.array([translation]), np.array([scale]), args.z_up)
    print(f"Inference on {points.shape[1]} points: {elapsed:.3f} s", flush=True)

    handler = PredictionHandler.from_outdict(outdict, points, [args.input.stem])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    handler.save_npz(str(args.output))
    print(f"Saved predictions: {args.output}", flush=True)
    if args.no_viewer and args.mesh_output is None:
        return
    mesh = handler.get_meshes(resolution=args.resolution)[0]
    if mesh is None:
        raise RuntimeError("Mesh extraction failed; predictions were saved successfully.")
    if args.mesh_output is not None:
        args.mesh_output.parent.mkdir(parents=True, exist_ok=True)
        mesh.export(str(args.mesh_output))
        print(f"Saved mesh: {args.mesh_output}", flush=True)
    if args.no_viewer:
        return

    import viser

    pc = handler.get_segmented_pcs()[0]
    server = viser.ViserServer(host="127.0.0.1", port=args.port)
    server.scene.add_mesh_trimesh("superquadrics", mesh=mesh, visible=True)
    server.scene.add_point_cloud(
        name="/segmented_pointcloud",
        points=np.asarray(pc.points),
        colors=np.asarray(pc.colors),
        point_size=0.005,
    )
    server.scene.set_up_direction([0.0, 0.0, 1.0] if args.z_up else [0.0, 1.0, 0.0])

    @server.on_client_connect
    def on_connect(client: viser.ClientHandle):
        center = pc.points.mean(axis=0)
        extent = max(float(np.ptp(pc.points, axis=0).max()), 0.1)
        client.camera.position = tuple(center + extent * np.array([0.8, 0.8, 0.8]))
        client.camera.look_at = tuple(center)

    print("Press Ctrl+C to stop the viewer.", flush=True)
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        pass
    finally:
        server.stop()


if __name__ == "__main__":
    main()
