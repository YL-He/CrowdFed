import torch
import copy
import numpy as np
import random
from torch import nn, autograd
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset, ConcatDataset


from utils.sampling import DatasetSplit
from src.strategy import FedAvg, Entropy, Normalized_entropy
from src.model import Encoder, Decoder



def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x/np.sum(e_x,axis=0)

class EMBEDDING_POOL(object): # On CPU
    def __init__(self,N):
        self.num = N
        self.embeddings = [ [] for _ in range(N)]
        self.loss = [[] for _ in range(N)]
        self.class_emb = []

    def store(self,Labels, Embeddings, Loss): # e.g. labels.data.cpu()
        for i,e,l in zip(Labels, Embeddings, Loss):
            self.embeddings[i].append(e.numpy())
            self.loss[i].append(l.numpy())

    def update_class(self, tmp): # tmp: outputs.data.cpu()[0], padding for []
        for i in range(self.num):
            if not self.loss[i]:
                self.embeddings[i].append(np.zeros_like(tmp))
                self.loss[i].append(.0)
        weights = [softmax(-np.array(i)) for i in self.loss]
        for i,e in enumerate(self.embeddings):
            self.class_emb.append(np.average(e,axis=0,weights=weights[i])) 

    def reset(self):
        self.embeddings = [ [] for _ in range(self.num)]
        self.loss = [[] for _ in range(self.num)]
        self.class_emb = []

    def SampleMatch(self, targets):
        # cosine similarity
        target_class = []
        class_idxs = []
        for target in targets:
            cos_sim = np.array([e.dot(target)/(np.linalg.norm(e)*np.linalg.norm(target))
                            if e.dot(target) else 0
                            for e in self.class_emb
                            ])
            target_class.append(self.class_emb[np.argmax(cos_sim)])
            class_idxs.append(np.argmax(cos_sim))
        return target_class, class_idxs

    def CategoryMatch(self, targets):
        h, local_labels = targets
        class_idxs = np.array([self.L[i] for i in local_labels])
        idxs = np.argwhere(class_idxs == -1).reshape(-1)
        if len(idxs) > 0:
            _, samples = self.SampleMatch(h[idxs])
            class_idxs[idxs] = samples
        class_idxs.tolist()
        return None, class_idxs

    def CategorySet(self, selected_embs):
        localclass_Vote = np.zeros((self.num, self.num))
        localclass_H = np.array([1e4 for _ in range(self.num)])
        flag, used = np.ones(self.num), np.zeros(self.num)
        for c, targets in enumerate(selected_embs):
            if len(targets)>0:
                votes = np.array(self.SampleMatch(targets)[1])
                sort_votes = [len(np.argwhere(votes == i)) for i in range(self.num)]
                ent = Normalized_entropy(sort_votes)
                localclass_H[c] = ent
                localclass_Vote[c] = np.array(sort_votes)
            else:
                ...
        L = -np.ones(self.num) # L[1]=7, local 1 ==> global 7
        for i in range(self.num):
            l_ = np.argmin(flag*localclass_H)
            if l_ > 1e3:
                continue
            g_ = np.argmax(localclass_Vote[l_])
            if used[g_] == 0:
                L[l_] = g_
                flag[l_] = 1e4
                used[l_] = 1
            else:
                # print("error: Global {0} has been used.".format(g_))
                ...

        self.L = L


    def find(self, targets): # find the most similar class
        if len(targets)==2:
            return self.CategoryMatch(targets)
        else:
            return self.SampleMatch(targets)



class CLIENT:

    def __init__(self,client_id, args, ModelWeights=None):
        self.id = client_id
        self.seed = client_id
        self.args = args
        N = self.args.num_classes
        self.model = copy.deepcopy(ModelWeights) # E, C

        # Encoder
        self.encoder = Encoder(N)

        # local data
        self.Udata = None       # Unlabel data
        self.Ldata = None       # label data for train
        self.Tdata = None       # label data for test

        # local label
        label = np.arange(N)
        if self.args.LPS:
            np.random.seed(self.seed)
            np.random.shuffle(label)
        self.relabel = label # init for local label


        self.local_embedding = EMBEDDING_POOL(N)
        self.global_embedding = None

        self.flag = False

        self.match = args.match  # 0 for sample, 1 for category


    @property
    def get_model(self):
        return copy.deepcopy(self.model)

    def download_model(self,w):
        self.model = copy.deepcopy(w)

    def download_embeddings(self, emb_pool):
        self.global_embedding = emb_pool

    @property
    def report(self):
        return {'id': self.id,
                'num': len(self.Ldata),
                'test_num':len(self.T_idxs)}

    def set_lr(self,lr):
        self.args.lr = lr
    def set_match(self,match):
        self.match = match
    def set_flag(self,flag):
        self.flag = flag

    def set_data(self, dataset, idxs):
        np.random.shuffle(idxs)
        self.U_idxs, self.L_idxs, self.T_idxs = np.split(idxs,[int(self.args.beta*len(idxs)), int(0.9*len(idxs))])

        self.Udata = DataLoader(DatasetSplit(dataset, self.U_idxs, self.relabel), batch_size=self.args.local_bs, shuffle=True)
        self.L_dataset = DatasetSplit(dataset, self.L_idxs, self.relabel)
        self.Ldata = DataLoader(DatasetSplit(dataset, self.L_idxs, self.relabel), batch_size=self.args.local_bs, shuffle=True)
        self.Tdata = DataLoader(DatasetSplit(dataset, self.T_idxs, self.relabel), batch_size=self.args.local_bs, shuffle=True)

        self.class_idcs = [[] for _ in range(self.args.num_classes)]
        for i, sample in enumerate(self.L_dataset):
            self.class_idcs[sample[1]].append(i)

    def random_sample(self, net, per_n=20):
        selected_idcs = [np.random.choice(self.class_idcs[i], min(per_n, len(self.class_idcs[i]))) for i in range(self.args.num_classes)]

        selected_embs = [[] for i in range(self.args.num_classes)]
        net.load_state_dict(self.model)
        net.to(self.args.device)
        net.eval()

        with torch.no_grad():
            for local_l, idxs in enumerate(selected_idcs):
                if len(idxs)>0:
                    images = torch.stack([self.L_dataset[idx][0] for idx in idxs]).to(self.args.device)
                    y_, h, x_ = net(images)
                    selected_embs[local_l] = h.data.cpu()

        return selected_embs


    def change_label(self, dataset, new_label):
        self.Ldata = DataLoader(DatasetSplit(dataset, self.L_idxs, new_label), batch_size=self.args.local_bs, shuffle=True)
        # self.Tdata = DataLoader(DatasetSplit(dataset, self.T_idxs, new_label), batch_size=self.args.local_bs,
        #                         shuffle=True)


    def local_train(self,net):

        if self.match:
            selected_embs = self.random_sample(net)
            self.global_embedding.CategorySet(selected_embs)


        net.load_state_dict(self.model)
        net.to(self.args.device)
        net.train()
        self.encoder.to(self.args.device)
        self.encoder.train()
        #
        optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=self.args.momentum)
        optimizer_enc = torch.optim.SGD(self.encoder.parameters(), lr=self.args.lr)
        L_c = torch.nn.CrossEntropyLoss(reduction='none') # Classification loss
        L_r = torch.nn.MSELoss(reduction='none') #Reconstruction loss
        L = torch.nn.CrossEntropyLoss()
        L_ED = torch.nn.MSELoss()


        epoch_loss_label = []
        epoch = self.args.local_ep

        for iter in range(epoch):
            batch_loss_label,batch_loss_unlabel = [], []
            for batch_idx, (images, labels) in enumerate(self.Ldata):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                optimizer.zero_grad()

                y_, h, x_ = net(images)

                optimizer_enc.zero_grad()
                local_labels = self.encoder(y_)
                loss_E_each = L_c(local_labels, labels)
                loss_enc = torch.sum(loss_E_each)
                loss_enc.backward(retain_graph=True)
                optimizer_enc.step()

                if self.match:
                    global_labels = torch.Tensor(np.array(self.global_embedding.find( (h.data.cpu(), labels.data.cpu()) )[1])).long().to(self.args.device)
                else:
                    global_labels = torch.Tensor(np.array(self.global_embedding.find(h.data.cpu())[1])).long().to(self.args.device)
                # loss_R_each = torch.mean(L_r(torch.flatten(images, 1), torch.flatten(x_, 1)), 1)
                loss_C_each = L_c(y_, global_labels)
                # loss = torch.sum(loss_C_each) + torch.mean(loss_R_each)
                loss = torch.sum(loss_C_each)
                loss.backward()

                # loss_each = loss_C_each + loss_R_each
                # self.local_embedding.store(global_labels.data.cpu(), h.data.cpu(), loss_C_each.data.cpu()) # store global label
                self.local_embedding.store(labels.data.cpu(), h.data.cpu(), loss_E_each.data.cpu())

                optimizer.step()

                batch_loss_label.append(loss.item())

            self.local_embedding.update_class(h.data.cpu()[0])

            if self.flag:
                for batch_idx, (images, labels) in enumerate(self.Udata):
                    images = images.to(self.args.device)
                    optimizer.zero_grad()
                    y_, h, x_ = net(images)
                    optimizer_enc.zero_grad()
                    local_labels = self.encoder(y_)

                    pesudo_labels = torch.Tensor(np.array(self.local_embedding.find(h.data.cpu())[1])).long().to(self.args.device)

                    loss_E_each = L_c(local_labels, pesudo_labels)
                    loss_enc = torch.sum(loss_E_each)
                    loss_enc.backward(retain_graph=True)
                    optimizer_enc.step()

                    h_class = torch.Tensor(np.array(self.global_embedding.find(h.data.cpu())[1])).long().to(self.args.device)
                    # h_class = torch.Tensor(np.array(self.global_embedding.find(h.data.cpu())[1])).long().to(self.args.device)
                    # loss_each = 1 - F.cosine_similarity(h,h_class)
                    # loss_R_each = L_r(torch.flatten(images,1), torch.flatten(x_,1))
                    # loss = torch.sum(loss_each) + torch.mean(loss_R_each)
                    loss = torch.sum(L_c(y_, h_class))
                    loss.backward()
                    optimizer.step()

                    batch_loss_unlabel.append(loss.item())

            self.local_embedding.reset()

            # epoch_loss_unlabel.append(sum(batch_loss_unlabel)/len(batch_loss_unlabel))
            epoch_loss_label.append(sum(batch_loss_label)/len(batch_loss_label))

        self.model = copy.deepcopy(net.state_dict())
        return epoch_loss_label

    def local_test(self,net,data_loader=None):
        if not data_loader:
            data_loader = self.Tdata

        net.load_state_dict(self.model)
        net.to(self.args.device)
        net.eval()
        self.encoder.to(self.args.device)
        self.encoder.eval()

        test_loss, correct = 0, 0

        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(data_loader):
                images, labels = images.to(self.args.device), labels.to(self.args.device)

                y_, h, x_ = net(images)
                local_labels = self.encoder(y_)


                test_loss += F.cross_entropy(local_labels, labels).item()
                y_pred = local_labels.data.max(1, keepdim=True)[1]
                correct += y_pred.eq(labels.data.view_as(y_pred)).long().cpu().sum()

            test_loss /= len(data_loader.dataset)
            accuracy = 100.00*correct /len(data_loader.dataset)

        return accuracy, test_loss


class SERVER:
    def __init__(self, args,ModelWeights=None):
        self.args = args
        N = self.args.num_classes
        self.model = ModelWeights
        self.Ldata = None
        self.Tdata = None
        self.relabel = np.arange(N)
        self.embedding_pool = EMBEDDING_POOL(N)

    @property
    def get_model(self):
        return copy.deepcopy(self.model)

    def download_model(self, w):
        self.model = copy.deepcopy(w)

    def set_data(self, dataset, idxs):
        self.Ldata = DataLoader(DatasetSplit(dataset, idxs, self.relabel), batch_size=self.args.local_bs, shuffle=True)
        self.Tdata = DataLoader(dataset, batch_size=self.args.local_bs, shuffle=True)

    def pretrain(self,net):
        net.load_state_dict(self.model)
        net.to(self.args.device)
        net.train()

        L_c = torch.nn.CrossEntropyLoss(reduction='none')  # Classification loss

        optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=self.args.momentum)
        epoch = self.args.local_ep
        store_emb = False
        for iter in range(epoch):
            if iter == epoch - 1:
                store_emb = True
                self.embedding_pool.reset()

            for batch_idx, (images, labels) in enumerate(self.Ldata):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                optimizer.zero_grad()

                y_, h, x_ = net(images)
                loss_C_each = L_c(y_,labels)

                if store_emb:
                    self.embedding_pool.store(labels.data.cpu(),h.data.cpu(),loss_C_each.data.cpu())

                loss = torch.sum(loss_C_each)
                loss.backward()

                optimizer.step()
            if store_emb:
                self.embedding_pool.update_class(h.data.cpu()[0])
                store_emb = False

        self.model = copy.deepcopy(net.state_dict())

    def test(self, net, data_loader=None):
        if not data_loader:
            data_loader = self.Tdata

        net.load_state_dict(self.model)
        net.to(self.args.device)
        net.eval()

        test_loss, correct = 0, 0

        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(data_loader):
                images, labels = images.to(self.args.device), labels.to(self.args.device)

                local_labels, h, x_ = net(images)

                test_loss += F.cross_entropy(local_labels, labels).item()
                y_pred = local_labels.data.max(1, keepdim=True)[1]
                correct += y_pred.eq(labels.data.view_as(y_pred)).long().cpu().sum()

            test_loss /= len(data_loader.dataset)
            accuracy = 100.00*correct /len(data_loader.dataset)

        return accuracy, test_loss


    def select(self, round, pre_clients, min_num=2):

        np.random.seed(round)
        client_num = max(min_num, int(np.rint(len(pre_clients)*self.args.frac)))
        self.selected_clients = np.random.choice(pre_clients, client_num, replace=False)

    def aggregate(self):
        models = []
        weights = []
        for c in self.selected_clients:
            models.append(c.get_model)
            weights.append(c.report['num'])
        self.model = FedAvg(models, weights=True, num_user=weights)



