import copy
import numpy as np
import torch
# from openfhe import *
import tenseal as ts

from models.Fed import FedAvg
from models.Nets import MLP, Mnistcnn
from models.Sia import SIA
from models.Update import LocalUpdate
from models.test import test_fun
from utils.dataset import get_dataset, exp_details
from utils.options import args_parser

#define a function called "decrypt"
def decrypt(enc):
    return enc.decrypt().tolist()

if __name__ == '__main__':
    # parse args
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    print(f'args.device:', {args.device})
    # load dataset and split data for users
    dataset_train, dataset_test, dict_party_user, dict_sample_user = get_dataset(args)

    # build model
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
    net_glob.train()#Set the neural network model net _ glob to enter the training mode

    # copy weights
    w_glob = net_glob.state_dict()
    print('1')

    # Setup TenSEAL context
    context = ts.context(
                ts.SCHEME_TYPE.CKKS,
                poly_modulus_degree=16384,
                coeff_mod_bit_sizes=[60, 40, 40, 40, 60]
            )
    context.generate_galois_keys()
    context.global_scale = 2**40
    print('2')
    #encrypt the global weights
    encrypted_w_glob = {}
    for key, value in w_glob.items():
        encrypted_w_glob[key] = ts.ckks_tensor(context, value)
    print('3')

    # training
    if args.all_clients:
        print("Aggregation over all clients")
        w_locals = [w_glob for i in range(args.num_users)]
        encrypted_w_locals = [encrypted_w_glob for i in range(args.num_users)]
    print('4')

    best_att_acc = 0
    for iter in range(args.epochs):
        loss_locals = []
        if not args.all_clients:
            w_locals = []
            encrypted_w_locals = []
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)

        for idx in idxs_users:
            local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_party_user[idx])

            local_net = copy.deepcopy(net_glob).to(args.device)
            local_net.load_state_dict({key: decrypt(value) for key, value in encrypted_w_glob.items()})
            w, loss = local.train(net=local_net)

            if args.all_clients:
                w_locals[idx] = copy.deepcopy(w)
                encrypted_w_locals[idx] = {key: ts.ckks_tensor(context, value) for key, value in w_locals[idx].items()}
            else:
                w_locals.append(copy.deepcopy(w))
                encrypted_w_locals.append({key: ts.ckks_tensor(context, value) for key, value in w_locals[idx].items()})
            loss_locals.append(copy.deepcopy(loss))


        # implement the source inference attack
        ## SIA_attack = SIA(args=args, w_locals=w_locals, dataset=dataset_train, dict_sia_users=dict_sample_user)
        ## attack_acc = SIA_attack.attack(net=empty_net.to('cpu'))#args.device
        ## best_att_acc = max(best_att_acc, attack_acc)

        # update global weights
        # encrypted_w_glob = {}
        for key in w_locals[0].keys(): #Perform homomorphic addition of encrypted weight parameters with the same keys for each client.
            sum_encrypted_weights = encrypted_w_locals[0]
            for encrypted_w_local in encrypted_w_locals[1:]:
                sum_encrypted_weights = sum_encrypted_weights + encrypted_w_local
            avg_encrypted_weight = sum_encrypted_weights * (len(encrypted_w_locals))  #calculate average
            encrypted_w_glob[key] = avg_encrypted_weight #Put the average encrypted weight parameters to the global encrypted weight parameters.

        # At this point, encrypted_w_glob contains the encrypted global average weight parameters.
        # Decrypt the global weight parameters.
        w_glob = {key: decrypt(value) for key, value in encrypted_w_glob.items()}

        # copy weight to net_glob
        net_glob.load_state_dict(w_glob)
        acc_train, loss_train_ = test_fun(net_glob, dataset_train, args)
        # print loss
        loss_avg = sum(loss_locals) / len(loss_locals)
        print('Epoch Round {:3d}, Average training loss {:.3f}'.format(iter, loss_avg))

    print('5')
    # testing
    net_glob.eval()
    acc_train, loss_train_ = test_fun(net_glob, dataset_train, args)
    acc_test, loss_test = test_fun(net_glob, dataset_test, args)
    # experiment setting
    exp_details(args)


    print('Experimental result summary:')
    print("Training accuracy of the joint model: {:.2f}".format(acc_train))
    print("Testing accuracy of the joint model: {:.2f}".format(acc_test))
    
    print('Random guess baseline of source inference : {:.2f}'.format(1.0/args.num_users*100))
    print('Highest prediction loss based source inference accuracy: {:.2f}'.format(best_att_acc))
