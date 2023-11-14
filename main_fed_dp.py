import copy
import numpy as np
import torch
import pydp as dp  # 导入 PyDP 模块
from pydp.algorithms.laplacian import BoundedSum

from models.Fed import FedAvg
from models.Nets import MLP, Mnistcnn
from models.Sia import SIA
from models.Update import LocalUpdate
from models.test import test_fun
from utils.dataset import get_dataset, exp_details
from utils.options import args_parser
from utils.differential_privacy import add_laplace_noise

if __name__ == '__main__':
    # 解析参数
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    print(f'args.device:', {args.device})
    
    # 加载数据集并为用户分割数据
    dataset_train, dataset_test, dict_party_user, dict_sample_user, dict_simulation_user = get_dataset(args)

    # 建立模型
    if args.model == 'cnn' and args.dataset == 'MNIST':
        net_glob = Mnistcnn(args=args).to(args.device)
    elif args.model == 'mlp':
        len_in = 1
        dataset_train = dataset_train.dataset
        dataset_test = dataset_test.dataset
        img_size = dataset_train[0][0].shape
        for x in img_size:
            len_in *= x
        net_glob = MLP(dim_in=len_in, dim_hidden=200, dim_out=args.num_classes).to(args.device)
    else:
        exit('Error: unrecognized model')
    empty_net = net_glob
    print('Model architecture:')
    print(net_glob)
    net_glob.train()  # 将神经网络模型 net_glob 设置为训练模式

    # 复制权重
    w_glob = net_glob.state_dict()

    # 训练
    if args.all_clients:
        print("Aggregation over all clients")
        w_locals = [w_glob for i in range(args.num_users)]

    best_att_acc = 0
    for iter in range(args.epochs):
        loss_locals = []
        if not args.all_clients:
            w_locals = []
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)

        for idx in idxs_users:
            local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_party_user[idx])

            w, loss = local.train(net=copy.deepcopy(net_glob).to(args.device))

            if args.all_clients:
                w_locals[idx] = copy.deepcopy(w)
            else:
                w_locals.append(copy.deepcopy(w))
            loss_locals.append(copy.deepcopy(loss))

        # implement the source inference attack
        SIA_attack = SIA(args=args, w_locals=w_locals, dataset=dataset_train, dict_sia_users=dict_sample_user)
        attack_acc = SIA_attack.attack(net=empty_net.to('cpu'))#args.device
        best_att_acc = max(best_att_acc, attack_acc)

        # update global weights
        privacy_budget = 0.4  # 设置差分隐私隐私预算，根据需要调整
        #DP add noise
        minimum = []
        maximum = []
        for w_local in w_locals:
            for key, value in w_local.items():
                original_shape = value.shape
                flat_list = value.view(-1).tolist()
                minimum.append(min(flat_list))
                maximum.append(max(flat_list))

        for w_local in w_locals:
            for key, value in w_local.items():
                original_shape = value.shape
                flat_list = value.view(-1).tolist()
                # print(f'flat_list:{flat_list}')
                noised_list = add_laplace_noise(flat_list, args.epsilon, min(minimum), max(maximum))
                # print(f'noised_list:{noised_list}')
                reshaped_tensor = torch.tensor(noised_list).view(original_shape)
                w_local[key] = reshaped_tensor

        #server aggregation
        w_glob = FedAvg(w_locals)

        # copy weight to net_glob
        net_glob.load_state_dict(w_glob)
        acc_train, loss_train_ = test_fun(net_glob, dataset_train, args)
        # print loss
        loss_avg = sum(loss_locals) / len(loss_locals)
        print('Epoch Round {:3d}, Average training loss {:.3f}'.format(iter, loss_avg))
        #end of each epoch of args.epochs

    # 测试
    net_glob.eval()
    acc_train, loss_train_ = test_fun(net_glob, dataset_train, args)
    acc_test, loss_test = test_fun(net_glob, dataset_test, args)
    # 实验设置
    exp_details(args)



    print('Experimental result summary:')
    print("Training accuracy of the joint model: {:.2f}".format(acc_train))
    print("Testing accuracy of the joint model: {:.2f}".format(acc_test))
    
    print('Random guess baseline of source inference : {:.2f}'.format(1.0/args.num_users*100))
    print('Highest prediction loss based source inference accuracy: {:.2f}'.format(best_att_acc))
