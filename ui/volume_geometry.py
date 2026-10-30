"""Physical coordinates for voxel slice planes; handles anisotropy and shear."""
import numpy as np

def slice_plane_coordinates(shape,affine,axis,index,step=1):
    other=[value for value in range(3) if value!=axis]
    axes=[np.arange(0,shape[value],step,dtype=float) for value in other]
    first,second=np.meshgrid(*axes,indexing="ij")
    coordinates=[np.full(first.shape,index,dtype=float) for _ in range(3)]
    coordinates[other[0]]=first;coordinates[other[1]]=second
    voxel=np.stack(coordinates,axis=-1)
    world=voxel@np.asarray(affine)[:3,:3].T+np.asarray(affine)[:3,3]
    return tuple(world[...,value] for value in range(3))

def volume_world_bounds(shape,affine):
    corners=np.array(np.meshgrid(*[(0,length-1) for length in shape],indexing="ij")).reshape(3,-1).T
    world=corners@np.asarray(affine)[:3,:3].T+np.asarray(affine)[:3,3]
    return tuple(value for axis in range(3) for value in (world[:,axis].min(),world[:,axis].max()))

def slice_plane_pose(shape,affine,axis,index):
    center=(np.asarray(shape,dtype=float)-1)/2
    center[axis]=index
    affine=np.asarray(affine,dtype=float)
    origin=affine[:3,:3]@center+affine[:3,3]
    normal=np.linalg.inv(affine[:3,:3]).T[:,axis]
    normal/=np.linalg.norm(normal)
    return tuple(origin),tuple(normal)


def volume_camera_pose(shape, affine):
    """View the scan from its superior/right/anterior side; transform no data.

    Use the superior voxel direction as camera-up. Orthogonalize only the
    camera basis, never the physical slice geometry (which may contain shear).
    """
    affine = np.asarray(affine, dtype=float)
    right = affine[:3, 0].copy()
    right /= np.linalg.norm(right)
    up = affine[:3, 2] - right*np.dot(right, affine[:3, 2])
    up /= np.linalg.norm(up)
    anterior = np.cross(up, right)
    if np.dot(anterior, affine[:3, 1]) < 0:
        anterior *= -1
    direction = right + anterior + .65*up
    direction /= np.linalg.norm(direction)
    center = (affine @ np.r_[(np.asarray(shape)-1)/2, 1])[:3]
    bounds = np.asarray(volume_world_bounds(shape, affine)).reshape(3, 2)
    distance = max(1., float(np.linalg.norm(bounds[:, 1]-bounds[:, 0])))*1.8
    return tuple(center + distance*direction), tuple(center), tuple(up)
