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
                    # print(log_prob)
                    # tensor([[ 0.7444, -1.3788, -1.5015, -1.5817, 10.9929, -1.5876, -1.4226, -1.5035, -1.5850, -1.2491],
                    #         [ 4.2696, -1.5763, -1.6291, -1.6188,  8.2942, -1.7788, -1.5152, -1.4952, -1.8125, -1.5821],
                    #         [ 9.7028, -1.5080, -1.5516, -1.4804,  2.9473, -1.6133, -1.5931, -1.4809, -1.6065, -1.6130],
                    #         [ 7.5150, -1.6249, -1.5921, -1.6777,  5.4376, -1.8360, -1.6683, -1.6002, -1.6524, -1.7037],
                    #         [ 6.0079, -1.3055, -1.3941, -1.3381,  4.8255, -1.3195, -1.3623, -1.4024, -1.4178, -1.3107],
                    #         [ 3.2032, -1.3913, -1.6080, -1.4136,  8.7698, -1.6155, -1.5149, -1.5468, -1.5419, -1.4217],
                    #         [ 5.1713, -1.4082, -1.3500, -1.4401,  5.9176, -1.3834, -1.3678, -1.4180, -1.4512, -1.2441],
                    #         [-1.5387, -1.5982, -1.7975, -1.8339, 14.6218, -1.7132, -1.5005, -1.6010, -1.7261, -1.4833],
                    #         [ 6.1873, -1.4759, -1.4862, -1.4411,  5.2951, -1.5557, -1.4743, -1.4827, -1.4966, -1.4380],
                    #         [ 6.4237, -1.4271, -1.5463, -1.4450,  5.3560, -1.4959, -1.4793, -1.4797, -1.5276, -1.4450],
                    #         [ 9.4976, -1.1674, -1.4183, -1.2892,  1.3706, -1.2697, -1.4172, -1.3922, -1.3102, -1.2461],
                    #         [ 5.1815, -1.2578, -1.4478, -1.3773,  5.9090, -1.3577, -1.3509, -1.4131, -1.4302, -1.2786]], grad_fn=<AddmmBackward>)
                    # print(target) #tensor([6, 9, 0, 0, 3, 8, 6, 6, 8, 8, 6, 6])
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
        print('Prediction loss based source inference attack accuracy: {}/{} ({:.2f}%)'.format(correct_total, len_set,
                                                                                                 accuracy_sia))
        return accuracy_sia
