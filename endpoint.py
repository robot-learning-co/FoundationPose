from typing import Union

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, validator

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

class Request(BaseModel):
    mesh_file: str
    scene_dir: str
    output_dir: str
    
scorer = ScorePredictor()
refiner = PoseRefinePredictor()
glctx = dr.RasterizeCudaContext()


@app.post("/pose")
def estimate_pose(payload: Request):
    mesh_file = payload.mesh_file
    scene_dir = payload.scene_dir
    est_refine_iter = 5
    debug = 0
    output_dir = payload.output_dir

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
    
    return {"pose": pose.reshape(4,4).tolist(), "vis": f'{output_dir}/track_vis/{reader.id_strs[i]}.png'}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("endpoint:app",
                host="0.0.0.0",
                port=8000,
                reload=False)