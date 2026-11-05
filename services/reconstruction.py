import os
import sys
import torch
import trimesh
import numpy as np
from pathlib import Path
from typing import List

def generate_3d_mesh(image_paths: List[str], site_id: str) -> str:
    """
    Takes a list of multi-angle street view images, processes them through DUSt3R,
    and exports a clean .glb 3D point cloud.
    """
    output_dir = Path("static/assets/models")
    output_dir.mkdir(parents=True, exist_ok=True)
    mesh_path = output_dir / f"{site_id}.glb"

    # 1. Dynamically inject the dust3r path
    project_root = Path(__file__).resolve().parent.parent
    dust3r_repo_path = project_root / "dust3r"
    if str(dust3r_repo_path) not in sys.path:
        sys.path.append(str(dust3r_repo_path))

    # 2. Force the import at runtime 
    try:
        from dust3r.inference import inference
        from dust3r.model import AsymmetricCroCo3DStereo
        from dust3r.image_pairs import make_pairs
        from dust3r.utils.image import load_images
        from dust3r.cloud_opt import global_aligner, GlobalAlignerMode
    except Exception as e:
        raise RuntimeError(f"DUSt3R Import Failed: {e}")

    print(f"[3D Engine] Loading {len(image_paths)} images for reconstruction...", flush=True)
    
    # 3. Initialize DUSt3R Model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = "naver/DUSt3R_ViTLarge_BaseDecoder_512_dpt"
    model = AsymmetricCroCo3DStereo.from_pretrained(model_name).to(device)
    
    # 4. Prepare Images and Pairs
    imgs = load_images(image_paths, size=768)
    pairs = make_pairs(imgs, scene_graph="complete", prefilter=None, symmetrize=True)
    
    # 5. Run Inference
    print("[3D Engine] Running DUSt3R inference...", flush=True)
    output = inference(pairs, model, device, batch_size=2)
    
    # 6. Global Alignment using the correct optimizer method
    print("[3D Engine] Aligning 3D geometry...", flush=True)
    aligner = global_aligner(output, device=device, mode=GlobalAlignerMode.PointCloudOptimizer)
    aligner.compute_global_alignment(init="mst", niter=300, schedule="linear", lr=0.01)
    
    # 7. Extract 3D Data
    print("[3D Engine] Flattening and merging multi-view point clouds...", flush=True)
    pts3d = aligner.get_pts3d()
    colors = aligner.imgs
    masks = aligner.get_masks()
    
    pts_list = []
    color_list = []
    
    # Helper to safely convert PyTorch Tensors OR NumPy arrays to standard arrays
    def to_numpy(data):
        return data.detach().cpu().numpy() if hasattr(data, 'detach') else np.array(data)
    
    # Loop through each image's resulting data, flatten them, and apply the valid mask
    for i in range(len(pts3d)):
        p = to_numpy(pts3d[i]).reshape(-1, 3)
        c = to_numpy(colors[i]).reshape(-1, 3)
        m = to_numpy(masks[i]).reshape(-1)
        
        # Convert mask to strict boolean (DUSt3R confidences are sometimes floats)
        valid_mask = m > 0.25 if m.dtype.kind in 'fc' else m.astype(bool)
        
        pts_list.append(p[valid_mask])
        color_list.append(c[valid_mask])
        
    pts_flat = np.concatenate(pts_list, axis=0)
    colors_flat = np.concatenate(color_list, axis=0)
    
    # Trimesh expects RGB colors in [0, 255] integer format
    if colors_flat.max() <= 1.0:
        colors_flat = np.clip(colors_flat, 0, 1) * 255
    colors_flat = colors_flat.astype(np.uint8)
    
    print("[3D Engine] Exporting point cloud to .glb format...", flush=True)
    
    # 8. Convert to Trimesh PointCloud and export directly as a .glb
    pc = trimesh.PointCloud(pts_flat, colors=colors_flat)
    pc.export(file_obj=str(mesh_path), file_type="glb")
    
    print(f"[3D Engine] ✅ 3D Point Cloud saved to {mesh_path}", flush=True)
    
    return f"/static/assets/models/{site_id}.glb"