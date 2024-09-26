import torch
from torchvision import datasets, transforms
from torch.utils.tensorboard import SummaryWriter

import copy
import numpy as np

from src.devices import CLIENT, SERVER
from src.model import *
from utils.options import arg_parser
from utils.sampling import DatasetSplit, iid, dirichlet_noniid

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def setmodel(args):
    if args.model == 'mlp': ### mnist or fashion
        modelarch = Supervised(args)

    elif args.model == 'cnn' and args.num_channels == 1: ### mnist or fahsion
        modelarch = Supervised_CNN(args)

    elif args.model == 'cnn' and args.num_channels == 3: # for svhn and cifar
        modelarch = Supervised_CNN3(args)


    return modelarch

def setup(dataset, args):
    if args.LDS:
        client_idcs = dirichlet_noniid(dataset, args.num_users)
    else:
        client_idcs = iid(dataset, args.num_users)
    modelarch = setmodel(args)

    modelweights = modelarch.state_dict()
    Clients = [CLIENT(i, args, modelweights) for i in range(args.num_users)]
    [c.set_data(dataset, client_idcs[i]) for i, c in enumerate(Clients)]

    Server = SERVER(args, modelweights)

    return np.array(Clients), Server, modelarch


def settag(args, more=''):

    tag = 'CrowdFed-' + args.model + '-R=' + str(args.rounds) + '-beta=' + str(args.beta)
    if args.num_public != 100:
        tag = tag + '-public=' + str(args.num_public)
    if args.LDS:
        tag = tag + '+LDS'
    if args.LPS:
        tag = tag + '+LPS'
    if args.match == 1:
        tag = tag + '+Cate'
    if args.twomatch:
        tag = tag + "+Sam-Cate"
    return tag

if __name__ == '__main__':
    args = arg_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() else 'cpu')
    if args.all_clients:
        args.frac = 1.0

    args.beta = 0.5
    args.rounds = 200
    args.dataset = 'mnist'
    args.model = 'cnn'
    args.LPS = True
    args.flag = False
    args.match = 0
    args.twomatch = True

    if args.dataset == 'mnist':
        args.input_size = 28 * 28
        trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        dataset = datasets.MNIST('./data/', train=True, download=True, transform=trans_mnist)
        dataset_test = datasets.MNIST('./data/', train=False, download=True, transform=trans_mnist)

    elif args.dataset == 'cifar':
        args.num_channels = 3
        args.input_size = 32 * 32
        trans_cifar = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataset = datasets.CIFAR10('./data/', train=True, download=True, transform=trans_cifar)
        dataset_test = datasets.CIFAR10('./data/', train=False, download=True, transform=trans_cifar)


    elif args.dataset == 'fashion_mnist':
        args.input_size = 28 * 28
        trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        dataset = datasets.FashionMNIST('./data/', train=True, download=True, transform=trans_mnist)
        dataset_test = datasets.FashionMNIST('./data/', train=False, download=True, transform=trans_mnist)

    elif args.dataset == 'svhn':
        args.input_size = 32 * 32
        args.num_channels = 3
        trans_svhn = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataset = datasets.SVHN('./data/', split='train', download=True, transform=trans_svhn)
        dataset_test = datasets.SVHN('./data/', split='test', download=True, transform=trans_svhn)

    print(args)

    set_seed(args.seed)
    Clients, Server, modelarch = setup(dataset, args)

    tag = settag(args)
    writer = SummaryWriter('runs/{0}/{1}'.format(args.dataset, tag))
    idxs = np.random.choice(np.arange(len(dataset_test)), args.num_public, replace=False)
    Server.set_data(dataset_test, idxs)
    stored_acc, stored_loss = [], []

    for r in range(args.rounds):
        Server.pretrain(modelarch)
        Server.select(r, Clients)
        loss_l = []
        for c in Server.selected_clients:
            if r > 30 and args.twomatch:
                c.set_match(1)
            c.download_model(Server.get_model)
            c.download_embeddings(Server.embedding_pool)
            loss = c.local_train(modelarch)
            loss_l.append(loss[0])

        Server.aggregate()
        print("Training Loss: {0:.3}".format(np.mean(loss_l)))

        test_acc, test_loss = [], []
        for i, c in enumerate(Clients): # all client test
            c.download_model(Server.get_model)
            a,l = c.local_test(modelarch)
            test_acc.append(a), test_loss.append(l)

        stored_acc.append(test_acc), stored_loss.append(test_loss)
        writer.add_scalar("Acc", np.mean(test_acc), r)
        writer.add_scalar("Loss", np.mean(test_loss), r)

        print("Round {0}: avg {1:.4} %/ std {2:.4}".format(r, np.mean(test_acc), np.std(test_acc)))

    np.save("Results/{0}/{1}-acc.npy".format(args.dataset,tag), np.array(stored_acc))
    np.save("Results/{0}/{1}-loss.npy".format(args.dataset, tag), np.array(stored_loss))


