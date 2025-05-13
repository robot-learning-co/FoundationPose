import requests
import numpy as np
import trimesh

# Prepare the matrix

matrix = np.array([[606.7363891601562, 0.0, 317.734130859375], 
                   [0.0, 606.6710205078125, 242.07131958007812], 
                   [0.0, 0.0, 1.0]])
matrix_json = matrix.tolist()


rgb_path = '/home/ubuntu/trlc/data/images/1747013940674/rgb/1747013940674.png'
depth_path = '/home/ubuntu/trlc/data/images/1747013940674/depth/1747013940674.png'
mask_path = '/home/ubuntu/trlc/data/images/1747013940674/masks/1747013940674.png'

mesh_obj_path = '/home/ubuntu/trlc/data/meshes/lego/LEGO_DUPLO_.obj'
mesh_mtl_path = '/home/ubuntu/trlc/data/meshes/lego/LEGO_DUPLO_.obj.mtl'
mesh_texture_path = '/home/ubuntu/trlc/data/meshes/lego/texture.png'

files = {
    'mesh_obj': open(mesh_obj_path, 'rb'),
    'mesh_mtl': open(mesh_mtl_path, 'rb'),
    'mesh_texture': open(mesh_texture_path, 'rb'),
    'rgb': open(rgb_path, 'rb'),
    'depth': open(depth_path, 'rb'),
    'mask': open(mask_path, 'rb'),
}
data = {
    'cam_K': str(matrix_json)  # or json.dumps(matrix_json)
}

response = requests.post(
    'http://localhost:8000/pose',
    files=files,
    data=data
)

print(response.json())
# print(response.json())