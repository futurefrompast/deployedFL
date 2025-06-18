import os
import sys
import copy
import argparse
import time
import signal
import random
import wandb
from datetime import datetime
from collections import OrderedDict
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split, Subset

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from lib.models.swin import SwinTransformer


def communication(server, clients, alpha_server=0.9, alpha_client=0.9, mode='full'):
    # Server: Freeze BN params
    new_server_dict = copy.deepcopy(OrderedDict(server.meta_named_parameters()))
    new_client_dict = [copy.deepcopy(OrderedDict(clients[i].meta_named_parameters())) for i in range(len(clients))]

    for name in new_server_dict.keys():
        if name.find('norm') >= 0:
            if mode == "low":
                if name.find('layers.0') >= 0:
                    param_avg = alpha_server * new_server_dict[name] + (1. - alpha_server) * sum([new_client_dict[i][name] for i in range(len(clients))]) / float(len(clients))
                    new_server_dict[name] = param_avg
                    for i in range(len(clients)):
                        new_client_dict[i][name] = alpha_client * new_client_dict[i][name] + (1. - alpha_client) * param_avg
            elif mode == "high":
                if name.find('layers.2.blocks.4') >= 0 or name.find("layers.2.blocks.5") >= 0:
                    param_avg = alpha_server * new_server_dict[name] + (1. - alpha_server) * sum([new_client_dict[i][name] for i in range(len(clients))]) / float(len(clients))
                    new_server_dict[name] = param_avg
                    for i in range(len(clients)):
                        new_client_dict[i][name] = alpha_client * new_client_dict[i][name] + (1. - alpha_client) * param_avg
            else:
                device_server = new_server_dict[name].device
                param_avg = alpha_server * new_server_dict[name] + (1. - alpha_server) * sum([new_client_dict[i][name].to(device_server) for i in range(len(clients))]) / float(len(clients))
                new_server_dict[name] = param_avg
                for i in range(len(clients)):
                    device_client = new_client_dict[i][name].device
                    new_client_dict[i][name] = alpha_client * new_client_dict[i][name] + (1. - alpha_client) * param_avg.to(device_client)

    server.load_state_dict(new_server_dict, strict=False)
    for i, client in enumerate(clients):
        client.load_state_dict(new_client_dict[i], strict=False)


def train(args, model_server, model_clients,
            optimizer_server, optimizer_client, loss_function, server_loader, client_loader, epoch=0, stop_iter=500):

    start = time.time()
    model_server.train()
    model_clients.eval()

    device_server = next(model_server.parameters()).device

    n_clients = len(model_clients)
    for batch_index, (images, labels) in enumerate(server_loader):

        if batch_index == stop_iter:
            break

        images = images.to(device_server)
        labels = labels.to(device_server)

        optimizer_server.zero_grad()
        outputs_server = model_server(images)

        loss_server = loss_function(outputs_server, labels)
        loss_server.backward()
        optimizer_server.step()

        loss_client_list = []
        if batch_index % args.round_iter == 0:
            for i in range(n_clients):
                images_client, labels_client = next(iter(client_loader[i]))

                device = next(model_clients[i].parameters()).device
                images_client = images_client.to(device)
                labels_client = labels_client.to(device)

                optimizer_client[i].zero_grad()
                outputs_client = model_clients[i](images_client)

                loss_client = loss_function(outputs_client, labels_client)
                loss_client.backward()
                optimizer_client[i].step()

                loss_client_list.append(loss_client)

            # Communication
            communication(model_server, model_clients, alpha_server=args.alpha_server, alpha_client=args.alpha_client, mode=args.mode)

        n_iter = (epoch - 1) * len(server_loader) + batch_index + 1

        if batch_index % 100 == 0:
            print('Training Epoch: {epoch} [{trained_samples}/{total_samples}]\tLoss(s): {:0.4f}\t, Loss(c): {} LR: {:0.6f}'.format(
                loss_server.item(),
                [l.item() for l in loss_client_list],
                optimizer_server.param_groups[0]['lr'],
                epoch=epoch,
                trained_samples=batch_index * args.b + len(images),
                total_samples=len(server_loader.dataset)
            ))
    finish = time.time()


@torch.no_grad()
def eval_training(args, model, loss_function, data_loader, epoch=0, name="server"):

    start = time.time()
    model.eval()

    test_loss = 0.0 # cost function error
    correct = 0.0

    device = next(model.parameters()).device

    for (images, labels) in data_loader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = loss_function(outputs, labels)

        test_loss += loss.item()
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum()

    finish = time.time()

    print('Test set: Epoch: {}, Loss(s): {:.4f}, Accuracy(s): {:.4f} (Time consumed:{:.2f}s)'.format(
        epoch,
        test_loss / len(data_loader.dataset),
        correct.float() / len(data_loader.dataset),
        finish - start
    ))

    return correct.float().item() / len(data_loader.dataset)


def main(args):

    random_seed = 0
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed) # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)

    #wandb.init(project=f"FEDAVG-{args.data_name}-swin-finer-corr")
    wandb.init(project=f"test-{args.data_name}")
    wandb.run.name = args.wandb_name
    wandb.run.save()

    wandb.config.update({
        "learning_rate": args.lr_server,
        "batch_size": args.b
        })

    # Models
    model_server = SwinTransformer(
                        embed_dim=96,
                        depths=[2, 2, 6, 2],
                        num_heads=[3, 6, 12, 24],
                        window_size=7,
                        )

    n_clients = len(args.corrupt_name_train)
    model_clients = nn.ModuleList([copy.deepcopy(model_server) for _ in range(n_clients)])

    model_server = model_server.to('cuda:0')
    for i in range(n_clients):
        model_clients[i] = model_clients[i].to(f'cuda:{i+1}')

    # Load pretrained
    if args.weights is not None:
        weights_path = args.weights
        print("Load network from {}".format(weights_path))
        ckpt = torch.load(weights_path, weights_only=True)["state_dict"]

        new_ckpt = OrderedDict()
        for n, v in ckpt.items():
            name = n.replace("module.","")
            new_ckpt[name] = v

        model_server.load_state_dict(new_ckpt)
        for i in range(n_clients):
            model_clients[i].load_state_dict(new_ckpt)

    # Datasets and DataLoader
    transform_server_train = transforms.Compose([
        transforms.TrivialAugmentWide(interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    transform_client_train = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    orig_train_path = os.path.join(args.data_root, args.data_name, "train")
    if args.corrupt_name_train is None: 
        args.corrupt_name_train = []
    corr_train_path = []
    for c_name in args.corrupt_name_train:
        corr_train_path.append(
            os.path.join(args.data_root, "domain", args.data_name + "_c", c_name, str(args.level))
        )

    server_dataset = torchvision.datasets.ImageFolder(
        orig_train_path,
        transform=transform_server_train
        )
    server_dataset = Subset(server_dataset, list(range(1, len(server_dataset), 2)))

    client_dataset = []
    for c_train_path in corr_train_path:
        dset = torchvision.datasets.ImageFolder(
                c_train_path,
                transform=transform_client_train
                )
        subset_indices = list(range(0, len(dset), 2))
        dset = Subset(dset, subset_indices)
        client_dataset.append(dset)

    server_loader = DataLoader(
        server_dataset, shuffle=True, num_workers=4, batch_size=args.b, pin_memory=True)
    client_loader = [DataLoader(
        client_dataset[i], shuffle=True, num_workers=1, batch_size=args.b_client, pin_memory=True) for i in range(n_clients)]

    transform_test = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    orig_test_path = os.path.join(args.data_root, args.data_name, "val")
    if args.corrupt_name_test is None: 
        args.corrupt_name_test = []
    corr_test_path = []
    for c_name in args.corrupt_name_test:
        corr_test_path.append(
            os.path.join(args.data_root, "domain", args.data_name + "_c", c_name, str(args.level))
        )

    test_orig_dataset = torchvision.datasets.ImageFolder(
        orig_test_path,
        transform=transform_test,
    )
    test_orig_loader = DataLoader(
        test_orig_dataset, shuffle=True, num_workers=1, batch_size=100)

    test_corr_loader = []
    for c_path in corr_test_path:
        dset = torchvision.datasets.ImageFolder(
            c_path,
            transform=transform_test,
        )
        subset_indices = list(range(1, len(dset), 2))
        dset = Subset(dset, subset_indices)
        loader = DataLoader(
            dset, shuffle=True, num_workers=1, batch_size=100)
        test_corr_loader.append(loader)

    # Server: Freeze BN params
    server_params = model_server.meta_parameters()

    client_params = [list()] * n_clients
    for i in range(n_clients):
        for name, param in model_clients[i].meta_named_parameters():
            if name.find('norm') >= 0:
                if args.mode == "low":
                    if name.find('layers.0') >= 0:
                        param.requires_grad = True
                        client_params[i].append(param)
                elif args.mode == "high":
                    if name.find('layers.2.blocks.4') >= 0 or name.find("layers.2.blocks.5") >= 0:
                        param.requires_grad = True
                        client_params[i].append(param)
                else:
                    param.requires_grad = True
                    client_params[i].append(param)
            else:
                param.requires_grad = False

    loss_function = nn.CrossEntropyLoss()
    optimizer_server = optim.AdamW(server_params, lr=args.lr_server, weight_decay=1e-1)
    #optimizer_client = [optim.AdamW(client_params[i], lr=args.lr_client, weight_decay=1e-1) for i in range(n_clients)]
    #optimizer_server = optim.SGD(server_params, lr=args.lr, momentum=0.9, weight_decay=5e-4)
    optimizer_client = [optim.SGD(client_params[i], lr=args.lr_client, momentum=0.9, weight_decay=5e-4) for i in range(n_clients)]

    iter_per_epoch = len(server_loader)

    best_acc = 0.0
    for epoch in range(1, args.epochs+1):

        acc_server = eval_training(args, model_server, loss_function, test_orig_loader, epoch, name="server")
        wandb.log({"Test server orig acc": acc_server})

        acc_server_avg = 0
        for i in range(n_clients):
            acc_server = eval_training(args, model_server, loss_function, test_corr_loader[i], epoch, name="server_{}".format(args.corrupt_name_test[i]))
            acc_server_avg += acc_server / float(n_clients)
        wandb.log({"Test server corruption acc": acc_server_avg})

        acc_clients_avg = 0
        for i in range(n_clients):
            acc_clients = eval_training(args, model_clients[i], loss_function, test_corr_loader[i], epoch, name="client_{}".format(args.corrupt_name_test[i]))
            acc_clients_avg += acc_clients / float(n_clients)
        wandb.log({"Test clients avg acc": acc_clients_avg})

        train(args, model_server, model_clients,
                optimizer_server, optimizer_client, loss_function, server_loader, client_loader, epoch, stop_iter=500)

    wandb.finish()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-data_root', type=str, required=True, help='data_root')
    parser.add_argument('-data_name', type=str, help='<Required> Set flag', required=True)
    parser.add_argument('-corrupt_name_train', nargs='+', help='<Required> Set flag', required=False)
    parser.add_argument('-corrupt_name_test', nargs='+', help='<Required> Set flag', required=False)
    parser.add_argument('-net', type=str, required=True, help='net type')
    parser.add_argument('-b', type=int, default=128, help='batch size for dataloader')
    parser.add_argument('-b_client', type=int, default=128, help='batch size for dataloader')
    parser.add_argument('-lr_server', type=float, default=0.1, help='initial learning rate')
    parser.add_argument('-lr_client', type=float, default=0.1, help='initial learning rate')
    parser.add_argument('-alpha_server', type=float, default=0.9, help='initial learning rate')
    parser.add_argument('-alpha_client', type=float, default=0.9, help='initial learning rate')
    parser.add_argument('-weights', type=str, required=False, help='pretrained weights')
    parser.add_argument('-wandb_name', type=str, required=True, help='optimizer type')
    parser.add_argument('-round_iter', type=int, required=True, help='optimizer type')
    parser.add_argument('-level', type=int, required=True, help='corruption level')
    parser.add_argument('-mode', type=str, required=False, help='full, low, high')
    parser.add_argument('-epochs', type=int, default=200, help='total epochs')
    args = parser.parse_args()

    main(args)

