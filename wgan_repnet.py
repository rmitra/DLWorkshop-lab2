"""
Implementation of the Paper from Wandt and Rosenhahn
"RepNet: Weakly Supervised Training of an Adversarial Reprojection Network for 3D Human Pose Estimation"

This training script trains a neural network similar to the paper.
Except some minor improvements that are documented in the code this is the original implementation.

For further information contact Bastian Wandt at wandt@tnt.uni-hannover.de
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
import torch.nn as nn

from h36m import H36M
from models import Lifter
from models import Critic
from utils import MPJPE
from utils import project_to_2d
from arguments import parse_args
from tqdm import tqdm


def train(epoch, lifter, critic, lifter_optimizer, critic_optimizer,
                                rep_loss, crit_loss, train_loader):

    # train
    rep_loss_total, lifter_critic_loss_total = 0, 0
    discriminator_loss_total = 0
    train_loader.dataset.init_epoch()

    for i, (input_2d, real_poses_3d, cameras, root) in enumerate(tqdm(train_loader, ascii=True)):
        # Constant Tensors to be used as labels for losses
        valid = torch.ones((real_poses_3d.size(0), 1)).float().cuda()
        fake = torch.zeros((input_2d.size(0), 1)).float().cuda()
        input_2d = input_2d.cuda()
        real_poses_3d = real_poses_3d.cuda()
        cameras = cameras.cuda()
        root = root.cuda()
        poses_3d = lifter(input_2d, cameras)

        # ---------------- Training the Critic ------------------

        real_validity = critic(real_poses_3d)
        fake_validity = critic(poses_3d.detach())
        critic_loss = -torch.mean(real_validity) + torch.mean(fake_validity)

        # Flush the existing gradients and backpropagate the loss
        critic_optimizer.zero_grad()
        critic_loss.backward()
        critic_optimizer.step()

        # ------------- Training the Generator -----------------

        # Adversarial/Critic Loss
        lifter_critic_loss = -torch.mean(critic(poses_3d))

        # Reprojection loss
        poses_rep = project_to_2d(poses_3d, cameras, root)
        reprojection_loss = rep_loss(poses_rep, input_2d)

        # Combined loss
        lifter_loss = reprojection_loss + lifter_critic_loss

        # Flush the existing gradients and backpropagate the loss
        lifter_optimizer.zero_grad()
        lifter_loss.backward()
        lifter_optimizer.step()



        # --------------- Book Keeping --------------------------
        rep_loss_total += reprojection_loss.cpu().item()
        lifter_critic_loss_total += lifter_critic_loss.cpu().item()
        discriminator_loss_total += critic_loss.cpu().item()

    print(f'epoch: {epoch}, rep_loss: {rep_loss_total / (i+1) :.4f}, \
            critic_loss = {lifter_critic_loss_total / (i+1) :.4f}, \
            discriminator_critic_loss = {discriminator_loss_total / (i+1) :.2f}')



def val(epoch, lifter, test_loader):
    # Setting the model's mode to eval mode
    lifter.eval()

    mpjpe = 0
    for i, (input_2d, gt_3d, cameras) in enumerate(test_loader):
        input_2d = input_2d.cuda()
        gt_3d = gt_3d.cuda()
        cameras = cameras.cuda()

        pred_3d = lifter(input_2d, cameras)

        # Calculate the Mean Per Joint Position Error
        mpjpe += MPJPE(pred_3d, gt_3d)

    print(f"The MPJPE for epoch {epoch} is {mpjpe/i}")

    # Resetting the model back to training mode
    lifter.train()


def main(cfg):
    # Initializing the 2D-to-3D lifting module
    lifter = Lifter(num_joints=17, hidden_dim=1000)
    lifter.cuda()
    lifter_optimizer = torch.optim.Adam(lifter.parameters(),
                                        lr=cfg.learning_rate,
                                        betas=(0.5, 0.9)
                                        )

    # Initializing the adversarial/critic module
    critic = Critic(num_joints=17, hidden_dim=100)
    critic.cuda()
    critic_optimizer = torch.optim.Adam(critic.parameters(),
                                        lr=cfg.learning_rate,
                                        betas=(0.5, 0.9)
                                        )

    # Data Loaders
    train_dataset = H36M(cfg, split='train')
    train_loader = torch.utils.data.DataLoader(
                                train_dataset,
                                batch_size = cfg.batch_size,
                                shuffle = True,
                                num_workers = 4
                                )
    test_dataset = H36M(cfg, split='test')
    test_loader = torch.utils.data.DataLoader(
                                test_dataset,
                                batch_size = cfg.batch_size,
                                shuffle = True,
                                num_workers = 4
                                )

    rep_loss = nn.MSELoss().cuda()
    crit_loss = nn.BCELoss().cuda()

    # Enter the training loop
    for epoch in range(cfg.epochs):
        train(epoch, lifter, critic, lifter_optimizer, critic_optimizer,
                            rep_loss, crit_loss, train_loader)

        if (epoch+1) % 1 == 0:
            val(epoch, lifter, test_loader)


if __name__ == "__main__":
    cfg = parse_args()
    main(cfg)
