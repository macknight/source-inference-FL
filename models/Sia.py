import numpy as np
import torch
import sys
from torch import nn
from torch.utils.data import DataLoader, Dataset


class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)#idxs already a python list

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label


class SIA(object):
    def __init__(self, args, w_locals=None, dataset=None, dict_sia_users=None):
        self.args = args
        self.w_locals = w_locals
        self.dataset = dataset
        self.dict_sia_users = dict_sia_users

    def attack(self, net):
        correct_total = 0
        len_set = 0
        for idx in self.dict_sia_users:
            #idx=0,1,...,9            
            dataset_local = DataLoader(DatasetSplit(self.dataset, self.dict_sia_users[idx]),
                                       batch_size=self.args.local_bs, shuffle=False)
            y_loss_all = []
            # evaluate the selected training data on each local model
            for local in self.dict_sia_users:
                y_loss_party = []
                idx_tensor = torch.tensor(idx)
                net.load_state_dict(self.w_locals[local])
                net.eval()
                for id, (data, target) in enumerate(dataset_local):
                    if self.args.gpu != -1:
                        data, target = data.to('cpu'), target.to('cpu') # data.cuda(), target.cuda()
                        idx_tensor = idx_tensor.to('cpu') # idx_tensor.cuda()
                    log_prob = net(data)
                    # {args.local_bs}=12
                    # print(data.shape) #torch.Size([12, 60])
                    # print(target.shape) #torch.Size([12])
                    # print(log_prob.shape) #torch.Size([12, 10])
                    # prediction loss based attack: get the prediction loss of the target training sample
                    loss = nn.CrossEntropyLoss(reduction='none')
                    y_loss = loss(log_prob, target)
                    # print(y_loss.shape) #torch.Size([12]),.....,torch.Size([12]),torch.Size([4])
                    y_loss_party.append(y_loss.cpu().detach().numpy()) # turn torch into numpy array
                # print(len(y_loss_party)) # size=9
                y_loss_party = np.concatenate(y_loss_party).reshape(-1)
                # print(len(y_loss_party)) # size=100
                y_loss_all.append(y_loss_party)

            #y_loss_all has {args.num_users} elements, each element has 100 floating numbers
            y_loss_all = torch.tensor(y_loss_all).to(self.args.device) # turn python list to pytorch tensor
            index_of_min_loss = y_loss_all.min(0, keepdim=True)[1] # [0] return the minimum values; [1] return the indexes corresponding to the minimum values
            # t_son1 = [1,2,3]
            # t_son2 = [3,2,1]
            # t_parent = [t_son1, t_son2]
            # t_parent = torch.tensor(t_parent).to(self.args.device) 
            # index_of_min_t = t_parent.min(0, keepdim=True)[1]
            # print(index_of_min_t) #tensor([[0, 0, 1]])
            # len(dataset_local.dataset)=100
            # print(idx_tensor) => [idx]
            correct_local = index_of_min_loss.eq(
                idx_tensor.repeat_interleave(len(dataset_local.dataset))).long().cpu().sum() # result of eq is a boolean tensor, and needs to convert to tensor with Long values and turned on CPU to use sum
            # correct_local is the correct guess count for that 100 smaple, min=0; max = 100
            correct_total += correct_local
            len_set += len(dataset_local.dataset)

        # calculate source inference attack accuracy
        accuracy_sia = 100.00 * correct_total / len_set
        print(
            '\nPrediction loss based source inference attack accuracy: {}/{} ({:.2f}%)\n'.format(correct_total, len_set,
                                                                                                 accuracy_sia))
        return accuracy_sia
