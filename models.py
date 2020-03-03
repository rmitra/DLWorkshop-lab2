import torch.nn as nn

class ResBlock(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.l1 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.l2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.leaky_relu = nn.LeakyReLU(0.1)

    def forward(self, x):
        y = self.l1(x)
        y = self.leaky_relu(y)
        y = self.l2(y)
        z = x + y
        z = self.leaky_relu(z)
        return z

class Lifter(nn.Module):
    def __init__(self, num_joints, hidden_dim):
        super().__init__()
        self.num_joints = num_joints
        self.hidden_dim = hidden_dim

        # Shared features between the Pose and Camera blocks
        self.shared_feats = nn.Sequential(
                            nn.Linear(2*self.num_joints, self.hidden_dim),
                            nn.LeakyReLU(0.1),
                            ResBlock(self.hidden_dim)
                            )


        # The next blocks are meant for 3D pose regression
        self.pose_reg = nn.Sequential(
                        ResBlock(self.hidden_dim),
                        ResBlock(self.hidden_dim),
                        nn.Linear(self.hidden_dim, self.hidden_dim),
                        nn.LeakyReLU(0.1),
                        nn.Linear(self.hidden_dim, 3*self.num_joints)
                        )

        # The next blocks are meant for camera regression
        self.cam_reg = nn.Sequential(
                        ResBlock(self.hidden_dim),
                        ResBlock(self.hidden_dim),
                        nn.Linear(self.hidden_dim, 6)
                        )


    def forward(self, input, cams):
        # input: n x n_joints*2
        # cams: n x 2 x 3
        backbone = self.shared_feats(input)

        poses_3d = self.pose_reg(backbone)
        # cams = self.cam_reg(backbone)

        return poses_3d

class Critic(nn.Module):
    def __init__(self, num_joints, hidden_dim):
        super().__init__()
        self.num_joints = num_joints
        self.hidden_dim = hidden_dim

        self.pose_critic = nn.Sequential(
                            nn.Linear(3*self.num_joints, self.hidden_dim),
                            nn.LeakyReLU(0.1),
                            ResBlock(self.hidden_dim),
                            nn.Linear(self.hidden_dim, self.hidden_dim),
                            nn.LeakyReLU(0.1),
                            nn.Linear(self.hidden_dim, self.hidden_dim),
                            nn.LeakyReLU(0.1),
                            nn.Linear(self.hidden_dim, 1),
                            nn.Sigmoid()
                            )

    def forward(self, input):
        out = self.pose_critic(input)
        return out
