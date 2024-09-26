import math
import torch
from torch import nn,autograd
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from torch.autograd import Variable


class Encoder(nn.Module):
    def __init__(self, N):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(N, N, bias=0),
            # nn.ReLU()
        )

    def forward(self, x):
        return self.encoder(x)

class Decoder(nn.Module):
    def __init__(self,N):
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.Linear(N, N, bias=0),
            # nn.ReLU()
        )
    def forward(self, x):
        return self.decoder(x)

class Extractor(nn.Module):
    def __init__(self,args):
        super(Extractor,self).__init__()
        self.input_size = args.input_size
        self.extractor = nn.Sequential(
            nn.Linear(args.input_size, 256),
            nn.ReLU(),
            nn.Linear(256, args.hidden_size),
            nn.ReLU()
        )

    def forward(self,x):
        x = x.view(-1, self.input_size)
        return self.extractor(x)

class Reconstructor(nn.Module):
    def __init__(self,args):
        super(Reconstructor,self).__init__()
        self.reconstructor = nn.Sequential(
            nn.Linear(args.hidden_size, 256),
            nn.ReLU(),
            nn.Linear(256, args.input_size),
            nn.Sigmoid()
        )

    def forward(self,x):
        return self.reconstructor(x)

class Unsupervised(nn.Module):
    def __init__(self,args):
        super(Unsupervised, self).__init__()
        self.extractor = Extractor(args)
        self.reconstructor = Reconstructor(args)

    def forward(self,x):
        h = self.extractor(x)
        x_ = self.reconstructor(h)
        return h, x_

class Classifier(nn.Module):
    def __init__(self,args):
        super(Classifier,self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(args.hidden_size, 128),
            nn.ReLU(True),
            nn.Linear(128,64),
            nn.ReLU(True),
            nn.Linear(64,args.num_classes)
        )
    def forward(self,x):
        return self.classifier(x)

class Supervised(nn.Module):
    def __init__(self,args):
        super(Supervised,self).__init__()
        self.extractor = Extractor(args)
        # self.reconstructor = Reconstructor(args)
        self.classifier = Classifier(args)

    def forward(self, X):
        x_data = X
        h = self.extractor(x_data)
        # x_ = self.reconstructor(h)
        y_ = self.classifier(h)
        return y_, h, None

##-----------------CNN------------------------##

class Extractor_CNN(nn.Module):
    def __init__(self,args):
        super(Extractor_CNN,self).__init__()
        self.input_size = args.input_size
        self.extractor = nn.Sequential(
            nn.Conv2d(args.num_channels, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 8, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )

    def forward(self,x):
        return self.extractor(x)

class Reconstructor_CNN(nn.Module):
    def __init__(self,args):
        super(Reconstructor_CNN,self).__init__()
        self.reconstructor = nn.Sequential(
            nn.ConvTranspose2d(8, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, args.num_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self,x):
        return self.reconstructor(x)

class Classifier_CNN(nn.Module):
    def __init__(self,args):
        super(Classifier_CNN,self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(args.hidden_size, 128),
            nn.ReLU(True),
            nn.Linear(128,64),
            nn.ReLU(True),
            nn.Linear(64,args.num_classes)
        )
    def forward(self,x):
        x = torch.flatten(x,1)
        return self.classifier(x)

class Supervised_CNN(nn.Module):
    def __init__(self,args):
        super(Supervised_CNN,self).__init__()
        self.extractor = Extractor_CNN(args)
        # self.reconstructor = Reconstructor_CNN(args)

        args.hidden_size = 8*7*7 # for mnist
        if args.dataset in ['cifar', 'svhn']:
            args.hidden_size = 8*8*8
        self.classifier = Classifier_CNN(args)

    def forward(self, X):
        x_data = X
        h = self.extractor(x_data)
        # x_ = self.reconstructor(h)
        y_ = self.classifier(h)
        return y_, torch.flatten(h,1), None


## -------CNN 3 for SVHN and CIFAR

class Extractor_CNN3(nn.Module):
    def __init__(self,args):
        super(Extractor_CNN3,self).__init__()
        self.input_size = args.input_size
        self.extractor = nn.Sequential(
            nn.Conv2d(args.num_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

    def forward(self,x):
        return self.extractor(x)

class Classifier_CNN3(nn.Module):
    def __init__(self,args):
        super(Classifier_CNN3,self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(64 * 8 * 8, 128),
            nn.ReLU(),
            nn.Linear(128,args.num_classes)
        )
    def forward(self,x):
        x = x.view(-1, 64 * 8 * 8)
        return self.classifier(x)

class Supervised_CNN3(nn.Module):
    def __init__(self,args):
        super(Supervised_CNN3,self).__init__()
        self.extractor = Extractor_CNN3(args)
        self.classifier = Classifier_CNN3(args)

    def forward(self, x):
        h = self.extractor(x)
        # x_ = self.reconstructor(h)
        y_ = self.classifier(h)
        return y_, torch.flatten(h,1), None