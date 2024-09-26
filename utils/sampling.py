from torch.utils.data import DataLoader, Dataset, Subset, ConcatDataset
import numpy as np

class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs, relabel):
        self.dataset = dataset
        self.idxs = list(idxs)
        self.relabel = relabel

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        label = self.relabel[label]
        return image, label

def iid(dataset,n_clients):
    num = int(len(dataset) / n_clients)
    idcs = np.arange(len(dataset))
    np.random.shuffle(idcs)

    client_idcs = [idcs[i*num:(i+1)*num] for i in range(n_clients)]

    return client_idcs

## Dirichlet
def dirichlet_noniid(dataset, n_clients):

    alpha = 1.0
    try:
        train_labels = np.array(dataset.targets)
    except Exception:
        train_labels = np.array(dataset.labels) # for svhn
    n_classes = train_labels.max() + 1
    label_distribution = np.random.dirichlet([alpha] * n_clients, n_classes)

    class_idcs = [np.argwhere(train_labels == y).flatten() for y in range(n_classes)]
    client_idcs = [[] for _ in range(n_clients)]

    for c, fracs in zip(class_idcs, label_distribution):
        for i, idcs in enumerate(np.split(c, (np.cumsum(fracs)[:-1] * len(c)).astype(int))):
            client_idcs[i] += [idcs]

    client_idcs = [np.concatenate(idcs) for idcs in client_idcs]

    return client_idcs

def pathological_noniid(dataset,n_clients,shards_per_client=2):
    train_labels = np.array(dataset.targets)

    n_classes = train_labels.max() + 1
    class_idcs = np.concatenate([np.argwhere(train_labels == y).flatten() for y in range(n_classes)])


    n_shards = shards_per_client * n_clients

    num = int(len(class_idcs)/n_shards)
    shards = [class_idcs[i*num:(i+1)*num] for i in range(n_shards)]

    np.random.shuffle(shards)

    client_idcs = [np.concatenate(shards[i*shards_per_client:(i+1)*shards_per_client])
                   for i in range(n_clients)]

    return client_idcs

def dirichlet_sampling(dataset,clients,modelarch,shared_num=50,num=20):

    labels = np.array(dataset.targets)
    n_classes = labels.max() + 1
    class_idcs = np.array([np.argwhere(labels == y).flatten() for y in range(n_classes)])
    # idxs = np.concatenate(class_idcs[l])

    class_dataset = [DataLoader(DatasetSplit(dataset, c), batch_size=20, shuffle=True)
                     for c in class_idcs]
    sample_pool = []
    for client in clients:
        acc_ = np.zeros(n_classes)
        for c,d in enumerate(class_dataset):
            a, l = client.local_test(modelarch, d)
            acc_[c] = a.item()
        acc_ = np.abs(acc_ - acc_.mean())
        sample_num = np.random.multinomial(num, np.random.dirichlet(acc_))
        sample_pool.append(np.concatenate([np.random.choice(c,i,replace=False) for i,c in zip(sample_num,class_idcs)]))

    sample_pool = np.concatenate(np.array(sample_pool))
    sample_share = np.random.choice(sample_pool,shared_num,replace=False)

    return DatasetSplit(dataset, sample_share)