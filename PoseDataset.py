import numpy as np
import torch.utils.data as data
import pickle
from arguments import parse_args

class PoseDataset(data.Dataset):
    def __init__(self, opt, split='train'):

        self.split = split
        self.opt = opt
        dataset_path = 'data/dataset_{}_gt.pickle'.format(split)

        with open(dataset_path, 'rb') as f:
            self.dataset = pickle.load(f)
        f.close()

        self.cameras, self.pose_2d, self.pose_3d = self.merge_videos()

        self.n_samples = self.pose_2d.shape[0]
        self.real_ids = np.arange(self.n_samples)
        self.gen_ids = np.arange(self.n_samples)

    def merge_videos(self):
        subject_map = {'S1': 1, 'S5': 5, 'S6': 6, 'S7': 7, 'S8': 8, 'S9': 9, 'S11': 11}

        # keys = ['subject', 'cameras', 'poses_3d', 'poses_2d']

        n_vids = len(self.dataset['subject'])

        subjects = np.tile(np.asarray(subject_map[self.dataset['subject'][0]]),
                           (self.dataset['poses_2d'][0].shape[0], 1))
        cameras = np.tile(self.dataset['cameras'][0], (self.dataset['poses_2d'][0].shape[0], 1))
        poses_2d = self.dataset['poses_2d'][0]
        poses_3d = self.dataset['poses_3d'][0]

        for i in range(1, n_vids):
            vid_samples = self.dataset['poses_2d'][i].shape[0]

            subjects = np.append(subjects, np.tile(np.asarray(
                subject_map[self.dataset['subject'][i]]), (vid_samples, 1)), axis=0)
            cameras = np.append(cameras, np.tile(self.dataset['cameras'][i], (vid_samples, 1)), axis=0)
            poses_2d = np.append(poses_2d, self.dataset['poses_2d'][i], axis=0)
            poses_3d = np.append(poses_3d, self.dataset['poses_3d'][i], axis=0)

        return cameras, poses_2d, poses_3d

    def __getitem__(self, idx):
        real_id = self.real_ids[idx]
        gen_id = self.gen_ids[idx]

        pose_2d = self.pose_2d[gen_id].reshape(-1)

        pose_3d_rel_gen = self.pose_3d[gen_id].copy()
        root = pose_3d_rel_gen[0, :].copy().reshape(1, -1)
        pose_3d_rel_gen[0, :] = 0
        pose_3d_rel_gen = pose_3d_rel_gen.reshape(-1)

        pose_3d_rel_real = self.pose_3d[real_id].copy()
        pose_3d_rel_real[0, :] = 0
        pose_3d_rel_real = pose_3d_rel_real.reshape(-1)

        camera_param_gen = self.cameras[gen_id].reshape(-1)

        if self.split == 'train':
            return pose_2d, pose_3d_rel_real, camera_param_gen, root
        else:
            return pose_2d, pose_3d_rel_gen, camera_param_gen


    def __len__(self):
        return self.n_samples

    def init_epoch(self):
        np.random.shuffle(self.real_ids)
        np.random.shuffle(self.gen_ids)

# if __name__ == "__main__":
#     import torch
#     opt = parse_args()
#     dataset = H36M(opt, split='train')
#     import pdb; pdb.set_trace()
#     dataloader = torch.utils.data.DataLoader(dataset, batch_size=4)
#     for i, (a, b, c, d) in enumerate(dataloader):
#         pdb.set_trace()
#         a = 10
