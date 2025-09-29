import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from model import *
from data_loader import *
import matplotlib.pyplot as plt
from tqdm import tqdm
import IPython
import gc
import matplotlib
from torchvision import transforms
from load_corrupted_data import CIFAR10, CIFAR100
matplotlib.rcParams.update({'errorbar.capsize': 5})
# thêm argument về dataset
import argparse

parser = argparse.ArgumentParser(description='Meta Learning with PyTorch')
parser.add_argument('--dataset', default='cifar10', type=str,
                    help='dataset (cifar10 [default] or cifar100)')
parser.add_argument('--corruption_prob', type=float, default=0.4,
                    help='label noise')
parser.add_argument('--corruption_type', '-ctype', type=str, default='unif',
                    help='Type of corruption ("unif" or "flip" or "flip2").')
parser.add_argument('--num_meta', type=int, default=1000)
parser.add_argument('--epochs', default=120, type=int,
                    help='number of total epochs to run')
parser.add_argument('--iters', default=60000, type=int,
                    help='number of total iters to run')
parser.add_argument('--start-epoch', default=0, type=int,
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--batch_size', '--batch-size', default=100, type=int,
                    help='mini-batch size (default: 100)')
parser.add_argument('--lr', '--learning-rate', default=1e-1, type=float,
                    help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--nesterov', default=True, type=bool, help='nesterov momentum')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                    help='weight decay (default: 5e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    help='print frequency (default: 10)')
parser.add_argument('--layers', default=28, type=int,
                    help='total number of layers (default: 28)')
parser.add_argument('--widen-factor', default=10, type=int,
                    help='widen factor (default: 10)')
parser.add_argument('--droprate', default=0, type=float,
                    help='dropout probability (default: 0.0)')
parser.add_argument('--no-augment', dest='augment', action='store_false',
                    help='whether to use standard augmentation (default: True)')
parser.add_argument('--resume', default='', type=str,
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--name', default='WideResNet-28-10', type=str,
                    help='name of experiment')
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--prefetch', type=int, default=0, help='Pre-fetching threads.')

args = parser.parse_args()
hyperparameters = {
    'lr' : 1e-3,
    'momentum' : 0.9,
    'batch_size' : 100,
    'num_iterations' : 8000,
}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def build_dataset():
    normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                     std=[x / 255.0 for x in [63.0, 62.1, 66.7]])

    train_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])

    if args.dataset == 'cifar10':
        train_data_meta = CIFAR10(
            root='../data', train=True, meta=True, num_meta=args.num_meta, corruption_prob=args.corruption_prob,
            corruption_type=args.corruption_type, transform=train_transform, download=True)
        train_data = CIFAR10(
            root='../data', train=True, meta=False, num_meta=args.num_meta, corruption_prob=args.corruption_prob,
            corruption_type=args.corruption_type, transform=train_transform, download=True, seed=args.seed)
        test_data = CIFAR10(root='../data', train=False, transform=test_transform, download=True)


    elif args.dataset == 'cifar100':
        train_data_meta = CIFAR100(
            root='../data', train=True, meta=True, num_meta=args.num_meta, corruption_prob=args.corruption_prob,
            corruption_type=args.corruption_type, transform=train_transform, download=True)
        train_data = CIFAR100(
            root='../data', train=True, meta=False, num_meta=args.num_meta, corruption_prob=args.corruption_prob,
            corruption_type=args.corruption_type, transform=train_transform, download=True, seed=args.seed)
        test_data = CIFAR100(root='../data', train=False, transform=test_transform, download=True)


    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True,
        num_workers=args.prefetch, pin_memory=True)
    train_meta_loader = torch.utils.data.DataLoader(
        train_data_meta, batch_size=args.batch_size, shuffle=True,
        num_workers=args.prefetch, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, shuffle=False,
                                              num_workers=args.prefetch, pin_memory=True)

    return train_loader, train_meta_loader, test_loader



def to_var(x, requires_grad=True):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, requires_grad=requires_grad)

def build_model():
    net = ResNet32(args.dataset == 'cifar10' and 10 or 100)

    if torch.cuda.is_available():
        net.cuda()
        torch.backends.cudnn.benchmark=True

    opt = torch.optim.SGD(net.params(),lr=hyperparameters["lr"])
    
    return net, opt
def test(model, test_loader):
    model.eval()
    correct = 0
    test_loss = 0

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            test_loss +=F.cross_entropy(outputs, targets).item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.4f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        accuracy))

    return accuracy
def train_lre(train_loader,train_meta_loader,model, vnet,optimizer_model,optimizer_vnet,epoch):
    print('Training at epoch {}'.format(epoch)) 
    net_losses = []
    meta_losses_clean = []
    net_l = 0
    smoothing_alpha = 0.9 
    accuracy_log = []
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        model.train()
        inputs, targets = to_var(inputs), to_var(targets)

        meta_net = ResNet32(args.dataset == 'cifar10' and 10 or 100)
        meta_net.load_state_dict(model.state_dict())
        if torch.cuda.is_available():
            meta_net.cuda()
        
        y_f_hat = meta_net(inputs)
        cost = F.cross_entropy(y_f_hat, targets, reduction='none')
        eps = to_var(torch.ones(cost.size(0), 1))
        l_f_meta = torch.sum(cost.view(-1,1) * eps)
        meta_net.zero_grad()
        grads = torch.autograd.grad(l_f_meta, (meta_net.params()), create_graph=True)
        meta_net.update_params(lr_inner=hyperparameters['lr'], source_params=grads)
        try:
            inputs_val, targets_val = next(train_meta_loader_iter)
        except StopIteration:
            train_meta_loader_iter = iter(train_meta_loader)
            inputs_val, targets_val = next(train_meta_loader_iter)
        y_g_hat = meta_net(inputs_val)
        l_g_meta = F.cross_entropy(y_g_hat, targets_val)
        l_g_meta = torch.sum(l_g_meta)
        grad_eps = torch.autograd.grad(l_g_meta, eps, only_inputs=True)[0]
        w_tilde = torch.clamp(-grad_eps, min=0)
        norm_c = torch.sum(w_tilde)

        if norm_c != 0:
            w = w_tilde / norm_c
        else:
            w = w_tilde
        
        y_f_hat = model(inputs)
        cost = F.cross_entropy(y_f_hat, targets, reduction='none')
        l_f = torch.sum(cost.view(-1,1) * w)
        optimizer_model.zero_grad()
        l_f.backward()
        optimizer_model.step()

        meta_l = smoothing_alpha * meta_l + (1 - smoothing_alpha) * l_g_meta.item() if batch_idx > 0 else l_g_meta.item()

        meta_losses_clean.append(meta_l/(1-smoothing_alpha**(batch_idx+1)))

        net_l = smoothing_alpha * net_l + (1 - smoothing_alpha) * l_f.item() if batch_idx > 0 else l_f.item()
        net_losses.append(net_l/(1-smoothing_alpha**(batch_idx+1)))

        if batch_idx % 50 == 0:
            print('Epoch: [{}/{}], Iteration: [{}/{}] \t Net Loss: {:.4f} \t Meta Loss: {:.4f}'.format(
                epoch, args.epochs, batch_idx+1,
                len(train_loader), net_losses[-1], meta_losses_clean[-1]))
            accuracy = test(model)
            accuracy_log.append(accuracy)
            print('Val accuracy: {:.2f}%'.format(accuracy))
            model.train()
            gc.collect()
            torch.cuda.empty_cache()

model, optimizer_model = build_model()

train_loader, train_meta_loader, test_loader = build_dataset()

def main():
    vnet = None
    optimizer_vnet = None
    best_acc = 0
    for epoch in range(args.start_epoch, args.epochs):
        train_lre(train_loader,train_meta_loader,model, vnet,optimizer_model,optimizer_vnet,epoch)
        acc = test(model, test_loader)
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), 'best_model.pth')
            print('Model saved!')
    print('Best accuracy: {:.2f}%'.format(best_acc))