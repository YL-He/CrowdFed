import copy
import torch
import numpy as np

def FedAvg(w,weights=False,num_user=[]):
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        tmp = torch.zeros_like(w[0][k],dtype = torch.float32)
        if weights:
            for i in range(len(w)):
                tmp += w[i][k] * num_user[i]
            tmp = torch.true_divide(tmp,np.sum(num_user))
        else:
            for i in range(len(w)):
                tmp += w[i][k]
            tmp = torch.true_divide(tmp,len(w))
        w_avg[k].copy_(tmp)
    return w_avg


def Entropy(vector):
    total = np.sum(vector)
    probabilities = vector / total

    mask = probabilities > 0
    probabilities = probabilities[mask]

    entropy = -np.sum(probabilities * np.log2(probabilities))
    return entropy

def Normalized_entropy(vector):
    """
    计算标准化熵
    :param vector: np.ndarray, 每个元素表示每类的数量
    :return: float, 标准化熵值
    """
    k = len(vector)  # 类别的数量
    max_entropy = np.log2(k)  # 最大熵（均匀分布时）
    entropy = Entropy(vector)
    normalized_entropy = entropy / max_entropy
    return normalized_entropy
