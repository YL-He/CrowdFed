import argparse

def arg_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--rounds', type=int, default=200, help="total number of communication rounds")
    parser.add_argument('--num_users', type=int, default=50, help="number of users: K")
    parser.add_argument('--frac', type=float, default=0.3, help="fraction of clients: C")
    parser.add_argument('--local_ep', type=int, default=5, help="number of local epochs: E")
    parser.add_argument('--local_bs', type=int, default=20, help="local batch size: B")
    parser.add_argument('--lr', type=float, default=0.001, help="client learning rate")
    parser.add_argument('--momentum', type=float, default=0.5, help="SGD momentum (default: 0.5)")
    parser.add_argument('--hidden_size', type=int, default=256, help="size of embedding")

    parser.add_argument('--dataset', type=str, default='mnist', help="name of dataset")
    parser.add_argument('--model', type=str, default='mlp', help='model name (mlp/cnn)')
    parser.add_argument('--num_classes', type=int, default=10, help="number of classes")
    parser.add_argument('--num_channels', type=int, default=1, help="number of channels of images")
    parser.add_argument('--beta', type=float, default=0.5, help="fraction of unlabeled data")
    parser.add_argument('--num_public', type=int, default=100, help='number of the server data')
    parser.add_argument('--seed', type=int, default=2023, help='random seed (default: 2023)')
    parser.add_argument('--gpu', type=int, default=0, help="GPU ID, -1 for CPU")
    parser.add_argument('--flag', action='store_true', default=False, help='unlabel')
    parser.add_argument('--all_clients', action='store_true', default=False, help='aggregation over all clients')
    parser.add_argument('--LDS', action='store_true', default=False, help='use dirichlet_noniid')
    parser.add_argument('--LPS', action='store_true', default=False, help='client relabel')


    args = parser.parse_args()

    return args