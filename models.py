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

        # We wrap the Lifter architecture inside a Sequential container
        self.pose_reg = nn.Sequential(
                            nn.Linear(2*self.num_joints, self.hidden_dim),
                            nn.LeakyReLU(0.1),
                            ResBlock(self.hidden_dim)
                            ResBlock(self.hidden_dim),
                            ResBlock(self.hidden_dim),
                            nn.Linear(self.hidden_dim, self.hidden_dim),
                            nn.LeakyReLU(0.1),
                            nn.Linear(self.hidden_dim, 3*self.num_joints)
                        )


    def forward(self, input):
        # input: b x n_joints*2
        poses_3d = self.pose_reg(input)

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
