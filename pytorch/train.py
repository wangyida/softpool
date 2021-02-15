import open3d as o3d
import argparse
import random
import numpy as np
import torch
import torch.optim as optim
import sys
from dataset import *
from model import *
from utils import *
import os
import json
import time, datetime
import visdom
from time import time
sys.path.append("./emd/")
import emd_module as emd
sys.path.append("./chamfer/")
import dist_chamfer as cd
from extensions.gridding_loss import GriddingLoss
gridding_loss = GriddingLoss(scales=[64, 128], alphas=[0.5, 0.5])
# gridding_loss = GriddingLoss(scales=[128], alphas=[0.5])

parser = argparse.ArgumentParser()
parser.add_argument(
    '--batchSize', type=int, default=8, help='input batch size')
parser.add_argument(
    '--workers', type=int, help='number of data loading workers', default=12)
parser.add_argument(
    '--nepoch', type=int, default=750, help='number of epochs to train for')
parser.add_argument(
    '--model', type=str, default='', help='optional reload model path')
parser.add_argument(
    '--num_points', type=int, default=8192, help='number of points')
parser.add_argument(
    '--n_regions', type=int, default=16, help='number of surface elements')
parser.add_argument(
    '--env', type=str, default="SoftPoolNet", help='visdom environment')
parser.add_argument(
    '--dataset', type=str, default="shapenet", help='dataset for evaluation')
parser.add_argument(
    '--methods',
    nargs='+',
    default=['softpool', 'msn', 'folding', 'grnet'],
    help='a list of methods')
parser.add_argument('--savepath', type=str, default='', help='path for saving')

opt = parser.parse_args()
print(opt)


class FullModel(nn.Module):
    def __init__(self, model):
        super(FullModel, self).__init__()
        self.model = model
        self.EMD = emd.emdModule()
        self.CD = cd.chamferDist()

    def forward(self, parts, gt, part_seg, gt_seg, eps, iters):
        output1, output2, output3, output4, output5, output6 = self.model(
            parts, part_seg)
        loss_points = torch.zeros(1).cuda()
        loss_others = torch.zeros(1).cuda()
        if output1:
            dist1, dist2, _, _ = self.CD(output1[0], gt)
            cd1 = torch.mean(dist1, 1) + torch.mean(dist2, 1)
            dist1, dist2, _, _ = self.CD(output1[1], gt)
            cd1 += torch.mean(dist1, 1) + torch.mean(dist2, 1)
            """
            if self.model.n_regions == 1:
                dist1, dist2, _, _ = self.CD(
                    output1[0][:, output1[0].shape[1] // 2:, :], part)
                cd1 += torch.mean(dist1, 1) + torch.mean(dist2, 1)
                dist1, dist2, _, _ = self.CD(
                    output1[1][:, output1[1].shape[1] // 2:, :], part)
                cd1 += torch.mean(dist1, 1) + torch.mean(dist2, 1)
            """
            dist1, dist2, _, _ = self.CD(output1[2], gt)
            cd1 += torch.mean(dist1, 1) + torch.mean(dist2, 1)
            dist1, dist2, _, _ = self.CD(output1[3], gt)
            cd1 += torch.mean(dist1, 1) + torch.mean(dist2, 1)
            cd1 = cd1.mean()
            loss_points += cd1
            loss_others += 0.1 * output1[5]
        else:
            cd1 = []

        if output2:
            dist1, dist2, _, _ = self.CD(output2[0], gt)
            cd2 = torch.mean(dist1, 1) + torch.mean(dist2, 1)
            dist1, dist2, _, _ = self.CD(output2[1], gt)
            cd2 += torch.mean(dist1, 1) + torch.mean(dist2, 1)
            cd2 = cd2.mean()
            loss_points += cd2
            loss_others += 0.1 * output2[2]
        else:
            cd2 = []

        if output3:
            dist1, dist2, _, _ = self.CD(output3[0], gt)
            cd3 = torch.mean(dist1, 1) + torch.mean(dist2, 1)
            cd3 = cd3.mean()
            loss_points += cd3
        else:
            cd3 = []

        if output4:
            dist1, dist2, _, _ = self.CD(output4[0], gt)
            cd4 = torch.mean(dist1, 1) + torch.mean(dist2, 1)

            grid_loss = gridding_loss(output4[0], gt)
            cd4 += grid_loss

            dist1, dist2, idx1, _ = self.CD(output4[1], gt)
            cd4 += torch.mean(dist1, 1) + torch.mean(dist2, 1)

            SM = torch.nn.Softmax(dim=-1)
            sem_feat = SM(output4[3][:, :, :]).float()
            labels_gt = torch.gather(gt_seg[:, :, 0], dim=1, index=idx1.long())
            sem_gt = torch.nn.functional.one_hot(
                labels_gt.to(torch.int64), 12).float()
            loss_sem_fine = torch.mean(-torch.sum(
                0.97 * sem_gt * torch.log(1e-6 + sem_feat) +
                (1 - 0.97) * (1 - sem_gt) * torch.log(1e-6 + 1 - sem_feat),
                dim=-1))
            cd4 += 0.01 * loss_sem_fine

            dist1, dist2, idx1, _ = self.CD(output4[2], gt)
            cd4 += torch.mean(dist1, 1) + torch.mean(dist2, 1)
            sem_feat = SM(output4[4][:, :, :]).float()
            labels_gt = torch.gather(gt_seg[:, :, 0], dim=1, index=idx1.long())
            sem_gt = torch.nn.functional.one_hot(
                labels_gt.to(torch.int64), 12).float()
            loss_sem_coar = torch.mean(-torch.sum(
                0.97 * sem_gt * torch.log(1e-6 + sem_feat) +
                (1 - 0.97) * (1 - sem_gt) * torch.log(1e-6 + 1 - sem_feat),
                dim=-1))
            cd4 += 0.01 * loss_sem_coar
            cd4 = cd4.mean()
            loss_points += cd4
        else:
            cd4 = []

        if output5:
            dist1, dist2, _, _ = self.CD(output5[0], gt)
            cd5 = torch.mean(dist1, 1) + torch.mean(dist2, 1)
            cd5 = cd5.mean()
            loss_points += cd5
        else:
            cd5 = []

        if output6:
            dist1, dist2, _, _ = self.CD(output6[0], gt)
            cd6 = torch.mean(dist1, 1) + torch.mean(dist2, 1)
            dist1, dist2, _, _ = self.CD(output6[1], gt)
            cd6 += torch.mean(dist1, 1) + torch.mean(dist2, 1)
            cd6 = cd6.mean()
            loss_points += cd6
        else:
            cd6 = []

        return output1, output2, output3, output4, output5, output6, cd1, cd2, cd3, cd4, cd5, cd6, loss_points, loss_others


# vis = visdom.Visdom(port = 8097, env=opt.env) # set your port
now = datetime.datetime.now()
save_path = opt.savepath  # now.isoformat()
if not os.path.exists('./log/'):
    os.mkdir('./log/')
dir_name = os.path.join('log', save_path)
if not os.path.exists(dir_name):
    os.mkdir(dir_name)
logname = os.path.join(dir_name, 'log.txt')
os.system('cp ./train.py %s' % dir_name)
os.system('cp ./dataset.py %s' % dir_name)
os.system('cp ./model.py %s' % dir_name)

opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
best_val_loss = 10

dataset = ShapeNet(
    train=True, npoints=opt.num_points, dataset_name=opt.dataset)
dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=opt.batchSize,
    shuffle=True,
    num_workers=int(opt.workers))
dataset_test = ShapeNet(train=False, npoints=opt.num_points)
dataloader_test = torch.utils.data.DataLoader(
    dataset_test,
    batch_size=opt.batchSize,
    shuffle=False,
    num_workers=int(opt.workers))

len_dataset = len(dataset)
print("Train set size: ", len_dataset)
network = Network(
    num_points=opt.num_points,
    n_regions=opt.n_regions,
    model_lists=opt.methods)
network = torch.nn.DataParallel(FullModel(network))
network.cuda()
# network.module.model.apply(weights_init)  #initialization of the weight

if opt.model != '':
    network.module.model.load_state_dict(torch.load(opt.model))
    print("Previous weight loaded ")

lrate = 1e-4
optimizer = optim.Adam(
    network.module.model.parameters(),
    lr=lrate,
    weight_decay=0,
    betas=(.9, .999))

train_loss = AverageValueMeter()
val_loss = AverageValueMeter()
with open(logname, 'a') as f:  #open and append
    f.write(str(network.module.model) + '\n')

train_curve = []
val_curve = []
labels_generated_points = torch.Tensor(
    range(1,
          (opt.n_regions + 1) * (opt.num_points // opt.n_regions) + 1)).view(
              opt.num_points // opt.n_regions, (opt.n_regions + 1)).transpose(
                  0, 1)
labels_generated_points = (labels_generated_points) % (opt.n_regions + 1)
labels_generated_points = labels_generated_points.contiguous().view(-1)

for epoch in range(opt.nepoch):
    #TRAIN MODE
    # train_loss.reset()
    network.module.model.train()

    # learning rate schedule
    if epoch == 20:
        optimizer = optim.Adam(
            network.module.model.parameters(), lr=lrate / 10.0)
    if epoch == 40:
        optimizer = optim.Adam(
            network.module.model.parameters(), lr=lrate / 100.0)

    for i, data in enumerate(dataloader, 0):
        optimizer.zero_grad()
        id, part, gt, part_seg, gt_seg = data
        part = part.float().cuda()
        part_seg = part_seg.float().cuda()
        gt = gt.float().cuda()
        gt_seg = gt_seg.float().cuda()
        output1, output2, output3, output4, output5, output6, cd1, cd2, cd3, cd4, cd5, cd6, loss_points, loss_others = network(
            part.transpose(2, 1), gt, part_seg, gt_seg, 0.005, 50)

        loss_all = loss_points + loss_others
        loss_all.backward()
        # train_loss.update(cd4.mean().item())
        optimizer.step()

        if i % 10 == 0:
            idx = random.randint(0, part.size()[0] - 1)
        if i % 300 == 0:
            print('saving net...')
            torch.save(network.module.model.state_dict(),
                       '%s/network.pth' % (dir_name))

        print(
            opt.env +
            ' train [%d: %d/%d]  cd1: %.2f cd2: %.2f cd3: %.2f cd4: %.2f cd5: %.2f cd6: %.2f'
            % (epoch, i, len_dataset / opt.batchSize,
               cd1.item() * 1e4 if cd1 else 0, cd2.item() * 1e4 if cd2 else 0,
               cd3.item() * 1e4 if cd3 else 0, cd4.item() * 1e4 if cd4 else 0,
               cd5.item() * 1e4 if cd5 else 0, cd6.item() * 1e4 if cd6 else 0))
    # train_curve.append(train_loss.avg)

    # VALIDATION
    if epoch % 200 == 199:
        val_loss.reset()
        network.module.model.eval()
        with torch.no_grad():
            for i, data in enumerate(dataloader_test, 0):
                id, part, gt, gt_seg = data
                part = part.float().cuda()
                part_seg = part_seg.float().cuda()
                gt = gt.float().cuda()
                gt_seg = gt_seg.float().cuda()
                output1, output2, output3, output4, output5, output6, cd1, cd2, cd3, cd4, cd5, cd6, _ = network(
                    part.transpose(2, 1), gt, part_seg, gt_seg, 0.004, 3000)
                # val_loss.update(cd4.mean().item())
                idx = random.randint(0, part.size()[0] - 1)
                print(
                    opt.env +
                    ' val [%d: %d/%d]  cd1: %.2f cd2: %.2f cd3: %.2f cd4: %.2f'
                    % (epoch, i, len_dataset / opt.batchSize,
                       cd1.item() if cd1 else 0, cd2.item() if cd2 else 0,
                       cd3.item() if cd3 else 0, cd4.item() if cd4 else 0))
