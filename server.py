from typing import Union

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, validator
    
import os
import time

from estimater import *
from datareader import *

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
    
scorer = ScorePredictor()
refiner = PoseRefinePredictor()
glctx = dr.RasterizeCudaContext()

@app.post("/")
async def estimate_pose(
    request: Request, 
    mesh_obj: UploadFile = File(...),
    mesh_mtl: UploadFile = File(...),
    mesh_texture: UploadFile = File(...),
    rgb: UploadFile = File(...),
    depth: UploadFile = File(...),
    mask: UploadFile = File(...),
    cam_K: str = Form(...),
):
    client_host = request.client.host
    user_agent = request.headers.get("user-agent", "unknown")
    referer = request.headers.get("referer", "none")
    
    _mesh_obj = await mesh_obj.read()
    _mesh_mtl = await mesh_mtl.read()
    _mesh_texture = await mesh_texture.read()
    _rgb = await rgb.read()
    _depth = await depth.read()
    _mask = await mask.read()
    _cam_K=json.loads(cam_K)
    
    # Create a directory with the current timestamp
    timestamp = str(int(time.time() * 1000))
    readable_time = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime(int(timestamp) // 1000))
    output_dir = os.path.join(f"{os.getcwd()}/data/", f"{readable_time}")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "mesh"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "scene", "rgb"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "scene", "depth"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "scene", "masks"), exist_ok=True)
    
    # Save the files to the output directory
    with open(os.path.join(output_dir, "request_meta.txt"), "w") as f:
        f.write(f"Client IP: {client_host}\n")
        f.write(f"User-Agent: {user_agent}\n")
        f.write(f"Referer: {referer}\n")
    with open(os.path.join(output_dir, "mesh", f"{mesh_obj.filename}"), "wb") as f:
        f.write(_mesh_obj)
    with open(os.path.join(output_dir, "mesh", f"{mesh_mtl.filename}"), "wb") as f:
        f.write(_mesh_mtl)
    with open(os.path.join(output_dir, "mesh", f"{mesh_texture.filename}"), "wb") as f:
        f.write(_mesh_texture)
    with open(os.path.join(output_dir,"scene" , "rgb", f"{timestamp}.png"), "wb") as f:
        f.write(_rgb)
    with open(os.path.join(output_dir, "scene", "depth", f"{timestamp}.png"), "wb") as f:
        f.write(_depth)
    with open(os.path.join(output_dir, "scene", "masks", f"{timestamp}.png"), "wb") as f:
        f.write(_mask)
    
    with open(os.path.join(output_dir, "scene", "cam_K.txt"), "w") as f:
        f.write(f"{_cam_K[0][0]} {_cam_K[0][1]} {_cam_K[0][2]}\n")
        f.write(f"{_cam_K[1][0]} {_cam_K[1][1]} {_cam_K[1][2]}\n")
        f.write(f"{_cam_K[2][0]} {_cam_K[2][1]} {_cam_K[2][2]}\n")
    
    print("Saved files to", output_dir)

    mesh_file = os.path.join(output_dir, "mesh", f"{mesh_obj.filename}")
    scene_dir = os.path.join(output_dir, "scene")
    est_refine_iter = 5
    debug = 0

    set_logging_format()
    set_seed(0)

    mesh = trimesh.load(mesh_file)

    to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
    bbox = np.stack([-extents/2, extents/2], axis=0).reshape(2,3)

    est = FoundationPose(model_pts=mesh.vertices, model_normals=mesh.vertex_normals, mesh=mesh, scorer=scorer, refiner=refiner, debug_dir=output_dir, debug=debug, glctx=glctx)
    logging.info("estimator initialization done")
    reader = YcbineoatReader(video_dir=scene_dir, shorter_side=None, zfar=np.inf)
    
    for i in range(len(reader.color_files)):
        logging.info(f'i:{i}')
        color = reader.get_color(i)
        depth = reader.get_depth(i)
        if i==0:
            mask = reader.get_mask(0).astype(bool)
            pose = est.register(K=reader.K, rgb=color, depth=depth, ob_mask=mask, iteration=est_refine_iter)
        else:
            raise HTTPException(status_code=500, detail=str("Only one frame is supported for now"))
        
        os.makedirs(f'{output_dir}/ob_in_cam', exist_ok=True)
        np.savetxt(f'{output_dir}/ob_in_cam/{reader.id_strs[i]}.txt', pose.reshape(4,4))

        center_pose = pose@np.linalg.inv(to_origin)
        vis = draw_posed_3d_box(reader.K, img=color, ob_in_cam=center_pose, bbox=bbox)
        vis = draw_xyz_axis(color, ob_in_cam=center_pose, scale=0.04, K=reader.K, thickness=3, transparency=0, is_input_rgb=True)
        os.makedirs(f'{output_dir}/track_vis', exist_ok=True)
        imageio.imwrite(f'{output_dir}/track_vis/{reader.id_strs[i]}.png', vis)
    
    return {"pose": pose.reshape(4,4).tolist()}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app",
                host="0.0.0.0",
                port=80,
                reload=False)