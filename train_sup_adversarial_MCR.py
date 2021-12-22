import argparse
import os

import numpy as np
import torch
from torch.utils.data import DataLoader
from augmentloader import AugmentLoader
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

import train_func as tf
from loss import MaximalCodingRateReduction, MCR2_binary_classwise
import utils

# Set random seeds and deterministic pytorch for reproducibility
manualSeed = 100
torch.manual_seed(manualSeed)  # pytorch random seed
np.random.seed(manualSeed)  # numpy random seed
torch.backends.cudnn.deterministic = True

parser = argparse.ArgumentParser(description='Supervised Learning')
parser.add_argument('--arch', type=str, default='lenet',
                    help='architecture for deep neural network (default: resnet18)')
parser.add_argument('--fd', type=int, default=128,
                    help='dimension of feature dimension (default: 128)')
parser.add_argument('--data', type=str, default='mnist',
                    help='dataset for training (default: CIFAR10)')
parser.add_argument('--epo', type=int, default=200,
                    help='number of epochs for training (default: 800)')
parser.add_argument('--bs', type=int, default=2000,
                    help='input batch size for training (default: 1000)')
parser.add_argument('--lr', type=float, default=0.0001,
                    help='learning rate (default: 0.001)')
parser.add_argument('--mom', type=float, default=0.9,
                    help='momentum (default: 0.9)')
parser.add_argument('--wd', type=float, default=5e-4,
                    help='weight decay (default: 5e-4)')
parser.add_argument('--gam1', type=float, default=1.,
                    help='gamma1 for tuning empirical loss (default: 1.)')
parser.add_argument('--gam2', type=float, default=1.,
                    help='gamma2 for tuning empirical loss (default: 1.)')
parser.add_argument('--eps', type=float, default=0.5,
                    help='eps squared (default: 0.5)')
parser.add_argument('--corrupt', type=str, default="default",
                    help='corruption mode. See corrupt.py for details. (default: default)')
parser.add_argument('--lcr', type=float, default=0.,
                    help='label corruption ratio (default: 0)')
parser.add_argument('--lcs', type=int, default=10,
                    help='label corruption seed for index randomization (default: 10)')
parser.add_argument('--tail', type=str, default='',
                    help='extra information to add to folder name')
parser.add_argument('--transform', type=str, default='default_mnist',
                    help='transform applied to trainset (default: default') # just to_tensor
parser.add_argument('--save_dir', type=str, default='./saved_models/adv_train/',
                    help='base directory for saving PyTorch model. (default: ./saved_models/)')
parser.add_argument('--data_dir', type=str, default='./data/',
                    help='base directory for saving PyTorch model. (default: ./data/)')
parser.add_argument('--pretrain_dir', type=str, default=None,
                    help='load pretrained checkpoint for assigning labels')
parser.add_argument('--pretrain_epo', type=int, default=None,
                    help='load pretrained epoch for assigning labels')
parser.add_argument('--optim', type=str, default='SGD',
                    help='optimizer')
parser.add_argument('--save_freq', type=int, default=10,
                    help='save frequency')
parser.add_argument('--perturb_steps', type=int, default=10,
                    help='perturb_steps')
parser.add_argument('--step_size', type=float, default=0.04,
                    help='step_size')
parser.add_argument('--epsilon', type=float, default=0.3,
                    help='L-inf perturb size')

args = parser.parse_args()


## Pipelines Setup
model_dir = os.path.join(args.save_dir,
               'adv_train_MCR_eps{}_ps{}_ss{}_{}+{}_{}_epo{}_bs{}_lr{}_mom{}_wd{}_gam1{}_gam2{}_eps{}_lcr{}_optim{}{}'.format(
                    args.epsilon, args.perturb_steps, args.step_size, args.arch, args.fd, args.data, args.epo, args.bs, args.lr, args.mom, 
                    args.wd, args.gam1, args.gam2, args.eps, args.lcr, args.optim, args.tail))

"""Initialize folder and .csv logger."""
# utils.init_pipeline(model_dir)
# project folder
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
    os.makedirs(os.path.join(model_dir, 'checkpoints'))
    os.makedirs(os.path.join(model_dir, 'figures'))
    os.makedirs(os.path.join(model_dir, 'plabels'))

headers = ["epoch", "step", "loss_mcr", "discrimn_loss_mcr", "compress_loss_mcr", 
    "discrimn_loss_loss_b_mcr",  "compress_loss_loss_b_mcr"]
utils.create_csv(model_dir, 'losses.csv', headers)
print("project dir: {}".format(model_dir))
## Prepare for Training
if args.pretrain_dir is not None:
    net, _ = tf.load_checkpoint(args.pretrain_dir, args.pretrain_epo)
    utils.update_params(model_dir, args.pretrain_dir)
else:
    net = tf.load_architectures(args.arch, args.fd)
transforms = tf.load_transforms(args.transform)
trainset = tf.load_trainset(args.data, transforms, path=args.data_dir)
trainset = tf.corrupt_labels(args.corrupt)(trainset, args.lcr, args.lcs)

#trainset = tf.get_dataset_with_specific_class(trainset, [0,1,2,3,4])

trainloader = DataLoader(trainset, batch_size=args.bs, drop_last=True, num_workers=4)

criterion = MaximalCodingRateReduction(gam1=args.gam1, gam2=args.gam2, eps=args.eps)
criterion_binary = MCR2_binary_classwise(gam1=args.gam1, gam2=args.gam2, eps=args.eps)
if args.optim == 'SGD':
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.mom, weight_decay=args.wd)
elif args.optim == 'Adam':
    optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.wd)

scheduler = lr_scheduler.MultiStepLR(optimizer, [200, 400, 600], gamma=0.1)
utils.save_params(model_dir, vars(args))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

## Training
for epoch in range(args.epo):
    for step, (x_nat, batch_lbls) in enumerate(trainloader):
        x_nat = x_nat.to(DEVICE)
        x_adv = x_nat.clone().detach() + 0.001 * torch.randn(x_nat.shape).to(DEVICE).detach()
        
        loss_binary_mcr, _, _ = criterion_binary(net(x_nat), net(x_adv), batch_lbls, num_classes=10)
        print('\n')
        print((x_adv-x_nat).abs().max(), loss_binary_mcr.item())
        for _ in range(args.perturb_steps):
            x_adv.requires_grad_()
            with torch.enable_grad():
                loss_binary_mcr, _, _ = criterion_binary(net(x_nat), net(x_adv), batch_lbls, num_classes=10)
            grad = torch.autograd.grad(loss_binary_mcr, [x_adv])[0]
            x_adv = x_adv.detach() - args.step_size * torch.sign(grad.detach())
            x_adv = torch.min(torch.max(x_adv, x_nat - args.epsilon), x_nat + args.epsilon)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)
        print((x_adv-x_nat).abs().max(), loss_binary_mcr.item())

        features = net(x_nat.to(DEVICE))
        loss_mcr, loss_mcr_empi, loss_mcr_theo = criterion(features, batch_lbls, num_classes=10)
        loss_binary_mcr, loss_binary_mcr_expand, loss_binary_mcr_compress = criterion_binary(net(x_nat), net(x_adv), batch_lbls, num_classes=10)
        loss_total = loss_mcr - loss_binary_mcr
        
        print('Epoch:',epoch, 'Step:', step)
        print('loss mcr:', loss_mcr.item())
        print('loss mcr expand:', loss_mcr_empi[0])
        print('loss mcr compress:', loss_mcr_empi[1])
        print('loss binary mcr:', loss_binary_mcr.item())
        print('loss binary mcr expand:', loss_binary_mcr_expand)
        print('loss binary mcr compress:', loss_binary_mcr_compress)

        optimizer.zero_grad()
        loss_total.backward()
        optimizer.step()
        # utils.save_state(model_dir, epoch, step, loss_mcr.item(), *loss_empi, *loss_theo)
        utils.save_state(model_dir, epoch, step, loss_mcr.item(), loss_mcr_empi[0], loss_mcr_empi[1], loss_binary_mcr.item(), loss_binary_mcr_expand, loss_binary_mcr_compress)
    scheduler.step()
    if (epoch + 1) % args.save_freq == 0:
        utils.save_ckpt(model_dir, net, epoch)
print("training complete.")
