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
gridding_loss = GriddingLoss(scales=[64, 128], alphas=[0.1, 0.5])

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
        output1, output2, output3, output4, out_seg, input_chosen, loss_trans, expansion_penalty = self.model(
            parts, part_seg)
        """
        for i in range(16):
            out_seg[i] = out_seg[i].transpose(1, 2).contiguous()
            gt_seg = gt[(torch.abs((i-gt_seg[:, :, 0]*11))<0.1).nonzero()]

        """
        dist1, dist2, _, _ = self.CD(output1[0], gt)
        emd1 = torch.mean(dist1, 1) + torch.mean(dist2, 1)
        dist1, dist2, _, _ = self.CD(output1[1], part)
        emd1 += torch.mean(dist1, 1) + torch.mean(dist2, 1)
        dist1, dist2, _, _ = self.CD(output1[2], gt)
        emd1 += torch.mean(dist1, 1) + torch.mean(dist2, 1)

        dist1, dist2, _, _ = self.CD(output2[0], gt)
        emd2 = torch.mean(dist1, 1) + torch.mean(dist2, 1)
        dist1, dist2, _, _ = self.CD(output2[1], gt)
        emd2 += torch.mean(dist1, 1) + torch.mean(dist2, 1)

        dist1, dist2, _, _ = self.CD(output3, gt)
        emd3 = torch.mean(dist1, 1) + torch.mean(dist2, 1)

        dist1, dist2, _, _ = self.CD(output4[0], gt)
        emd4 = torch.mean(dist1, 1) + torch.mean(dist2, 1)

        grid_loss = gridding_loss(output4[0], gt)
        emd4 += grid_loss

        dist1, dist2, idx1, _ = self.CD(output4[1], gt)
        emd4 += torch.mean(dist1, 1) + torch.mean(dist2, 1)
        """
        import ipdb; ipdb.set_trace()
        for i in range(np.shape(gt)[0]):
            sem_feat = tf.nn.softmax(coarse[i, :, 3:], -1)
            sem_gt = torch.nn.functional.one_hot(gt_seg[i, :, 0][idx1[i].long()].to(torch.int64), 12)
            loss_sem_coarse = torch.reduce_mean(-torch.reduce_sum(
                0.97 * sem_gt * torch.log(1e-6 + sem_feat) +
                (1 - 0.97) * (1 - sem_gt) * torch.log(1e-6 + 1 - sem_feat), [1]))
            emd4 += 0.01 * loss_sem_coarse
        """
        """
        gt_seg = gt_seg[:,:,0]
        size = list(gt_seg.size())
        gt_seg = torch.gather(gt_seg, dim=1, index=indexes.long()).view(-1)
        ones = torch.sparse.torch.eye(16).cuda()
        gt_seg = ones.index_select(0, gt_seg.long())
        size.append(16)
        gt_seg = gt_seg.view(*size)
        enp = -torch.mean(torch.sum((gt_seg) * torch.log(out_seg+0.01), dim=2))-torch.mean(torch.sum((1-gt_seg) * torch.log(1-out_seg+0.01), dim=2))
        emd1 += enp
        """

        return output1, output2, output3, output4, input_chosen, emd1, emd2, emd3, emd4, loss_trans, expansion_penalty


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

network = Network(num_points=opt.num_points, n_regions=opt.n_regions)
network = torch.nn.DataParallel(FullModel(network))
network.cuda()
network.module.model.apply(weights_init)  #initialization of the weight

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
    train_loss.reset()
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
        output1, output2, output3, output4, input_chosen, emd1, emd2, emd3, emd4, l_trans, expansion_penalty = network(
            part.transpose(2, 1), gt, part_seg, gt_seg, 0.005, 50)
        """
        output1, output2, output3, output4, input_chosen, emd1, emd2, emd3, emd4, expansion_penalty = network(
            part, full_regions, seg.contiguous(), 0.005, 50)
        """
        """
        loss_net = emd1.mean() + expansion_penalty.mean() * 0.1 + emd2.mean(
        ) + emd3.mean() + emd4.mean()
        """
        loss_net = emd1.mean() + emd2.mean() + emd3.mean() + emd4.mean(
        ) + expansion_penalty.mean() * 0.1

        loss_net.backward()
        train_loss.update(emd2.mean().item())
        optimizer.step()

        if i % 10 == 0:
            idx = random.randint(0, part.size()[0] - 1)
        if i % 300 == 0:
            print('saving net...')
            torch.save(network.module.model.state_dict(),
                       '%s/network.pth' % (dir_name))

        print(opt.env +
              ' train [%d: %d/%d]  emd1: %f emd2: %f emd3: %f emd4: %f' %
              (epoch, i, len_dataset / opt.batchSize, emd1.mean().item(),
               emd2.mean().item(), emd3.mean().item(), emd4.mean().item()))
    train_curve.append(train_loss.avg)

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
                output1, output2, output3, output4, input_chosen, emd1, emd2, emd3, emd4, l_trans = network(
                    part.transpose(2, 1), gt, part_seg, gt_seg, 0.004, 3000)
                val_loss.update(emd2.mean().item())
                idx = random.randint(0, part.size()[0] - 1)
                print(opt.env +
                      ' val [%d: %d/%d]  emd1: %f emd2: %f emd3: %f emd4: %f' %
                      (epoch, i, len_dataset / opt.batchSize,
                       emd1.mean().item(), emd2.mean().item(),
                       emd3.mean().item(), emd4.mean().item()))

    val_curve.append(val_loss.avg)

    log_table = {
        "train_loss": train_loss.avg,
        "val_loss": val_loss.avg,
        "epoch": epoch,
        "lr": lrate,
        "bestval": best_val_loss,
    }
    with open(logname, 'a') as f:
        f.write('json_stats: ' + json.dumps(log_table) + '\n')
