import pickle
import torch

def load_pose_data(path):
    import pdb; pdb.set_trace()
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data

def reprojection_layer(self, poses_3d, cams, root):
    # Takes a 3D pose as input, and reprojects the pose into 2D pose
    # poses_3d: b x n_joints*3
    # cams: b x 6
    num_joints = poses_3d.size(1) // 3

    poses_3d = poses_3d.reshape(-1, num_joints, 3).permute(0, 2, 1)
    cams = cams.reshape(-1, 2, 3)

    poses_2d = torch.matmul(cams, poses_3d).permute(0, 2, 1)

    return poses_2d.reshape(-1, num_joints*2)

def project_to_2d(X, camera_params, root):
    """
    Project 3D points to 2D using only linear parameters (focal length and principal point).

    Arguments:
    X -- 3D points in *camera space* to transform (N, *, 3)
    camera_params -- intrinsic parameteres (N, 2+2+3+2=9)
    """
    X = X.view(-1, 17, 3) + root
    assert X.shape[-1] == 3
    assert len(camera_params.shape) == 2
    assert camera_params.shape[-1] == 9
    assert X.shape[0] == camera_params.shape[0]

    while len(camera_params.shape) < len(X.shape):
        camera_params = camera_params.unsqueeze(1)

    f = camera_params[..., :2]
    c = camera_params[..., 2:4]

    XX = torch.clamp(X[..., :2] / X[..., 2:], min=-1, max=1)
    XX = f*XX + c

    return XX.view(-1, 17*2)


def MPJPE(pred, gt):
    """
        pred: b x n_joints*3
        gt: b x n_joints*2
    """
    batch_size = gt.size(0)
    pred = 1000 * pred.reshape(batch_size, -1, 3)
    gt = 1000 * gt.reshape(batch_size, -1, 3)

    diff = ((pred - gt)**2).sum(-1)
    mpjpe = torch.sqrt(diff).mean()
    return mpjpe


if __name__ == "__main__":
    path = 'data/Human36M/dataset_test.pickle'
    data = load_pose_data(path)
