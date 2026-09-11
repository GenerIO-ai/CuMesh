# examples/uv_unwrap.py

import torch
import trimesh
import cumesh
import utils
from PIL import Image
from datetime import datetime
from time import perf_counter


# Select the identifier for the mesh to unwrap
IDENTIFIER = "example-03"


if __name__ == "__main__":
    start_time = datetime.now()
    start_counter = perf_counter()
    print(f"Start time: {start_time:%Y-%m-%d %H:%M:%S}")

    mesh = utils.load_mesh(f"{IDENTIFIER}.glb")

    vertices = torch.from_numpy(mesh.vertices).float()
    faces = torch.from_numpy(mesh.faces).int()
    print(f"Original mesh: {vertices.shape[0]} vertices, {faces.shape[0]} faces")

    vertices = vertices.cuda()
    faces = faces.cuda()
    
    mesh = cumesh.CuMesh()
    mesh.init(vertices, faces)

    new_vertices, new_faces, uv, chart_texture = mesh.uv_unwrap(
        verbose=True,
        debug_charts=True,
    )

    print(f"Packed UV mesh: {new_vertices.shape[0]} vertices, {new_faces.shape[0]} faces")

    image_texture=Image.fromarray(chart_texture.detach().cpu().numpy())
    visual = trimesh.visual.texture.TextureVisuals(
        uv=uv.detach().cpu().numpy(),
        image=image_texture,
    )
    new_mesh = trimesh.Trimesh(
        vertices=new_vertices.cpu().numpy(), 
        faces=new_faces.cpu().numpy(), 
        visual=visual,
        process=False 
    )
    utils.save_mesh(new_mesh, f"{IDENTIFIER}-unwrapped.glb")

    end_time = datetime.now()
    total_time = perf_counter() - start_counter
    print(f"End time: {end_time:%Y-%m-%d %H:%M:%S}")
    print(f"Total time: {total_time:.2f} seconds")
