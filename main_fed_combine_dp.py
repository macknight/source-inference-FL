import copy
import numpy as np
import torch
import pydp as dp
import sys
import random
import tenseal as ts
import time

from pydp.algorithms.laplacian import BoundedSum
from models.Fed import FedAvg
from models.Nets import MLP, Mnistcnn, model_dict_to_list, list_to_model_dict
from models.Sia import SIA
from models.SimulationAttack import SimulationAttack
from models.Update import LocalUpdate
from models.test import test_fun, averaged_test_fun
from utils.dataset import get_dataset, exp_details
from utils.options import args_parser
from utils.sampling import split_params, generate_full_param
from utils.differential_privacy import add_laplace_noise

def process(args):
    # Load the data set and split the data for the user
    dataset_train, dataset_test, dict_party_user, dict_sample_user, dict_simulation_user = get_dataset(args)

    # setup model
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
    
    empty_net = copy.deepcopy(net_glob).to(args.device)
    # print('Model architecture:')
    # print(net_glob)
    net_glob.train()  # switch to training mode

    m = max(int(args.frac * args.num_users), 1)
    idxs_users = np.random.choice(range(args.num_users), m, replace=False)
    idxs_users.sort()
    # print(f'idxs_users={idxs_users}')

    # initial settings
    execution_time = 0
    # train
    print("traning\n")
    best_att_acc = 0
    att_acc_list = []
    for iter in range(args.epochs):
        print('+++')
        #server attack simulation:
        Simulation_attack = SimulationAttack(args=args, model_dict=net_glob.state_dict(), dataset=dataset_train, dict_simulation_users=dict_simulation_user)
        encrypted_index = Simulation_attack.attack(net=empty_net.to('cpu'))#args.device
        
        print(f'len(encrypted_index):{len(encrypted_index)}')
        #prepare weights and biases``
        loss_locals = []
        w_locals = []
        # w_params_all = []
        w_params_need_encrypted = []
        w_params_SIA_guessed = []
        w_params_non_encrypted = []

        print(f'Epoch Round {iter} Start, local train')
        start_time = time.time()
        #<<CLIENTS>>
        for idx in idxs_users:
            local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_party_user[idx])
            w, loss = local.train(net=copy.deepcopy(net_glob).to(args.device)) #bug fixed: net_glob is changed in SimulationAttack
            loss_locals.append(copy.deepcopy(loss))
            tmp_w = copy.deepcopy(w)
            plain_param = model_dict_to_list(tmp_w)

            tmp_need_encrypted, tmp_non_encrypted, tmp_SIA_guessed = split_params(plain_param, encrypted_index) # => three plain parts

            w_params_need_encrypted.append(tmp_need_encrypted) #DP
            w_params_non_encrypted.append(tmp_non_encrypted) #plaintext

        #record time
        end_time = time.time()
        execution_time += end_time - start_time
        print(f'Operation time: {end_time - start_time}')

        start_time = time.time()
        #<<DP_NOISE>>
        w_params_encrypted = []
        if args.encrypt_percent != 0:
            epsilon = args.epsilon
            #get min & max
            minimum = []
            maximum = []
            for i in range(len(w_params_need_encrypted)):
                minimum.append(min(w_params_need_encrypted[i]))
                maximum.append(max(w_params_need_encrypted[i]))
            #add noise
            for i in range(len(w_params_need_encrypted)):
                w_param_noised = add_laplace_noise(w_params_need_encrypted[i], epsilon, min(minimum), max(maximum))
                w_params_SIA_guessed.append(w_param_noised) #part1
                w_params_encrypted.append(w_param_noised) #part1

        #record time
        end_time = time.time()
        execution_time += end_time - start_time
        print(f'Operation time: {end_time - start_time}')
        
        ## formal SIA attack toward: w_param_obfuscated is for attackers
        for i in range(len(idxs_users)):
            w_param_SIA_guessed = []
            if args.encrypt_percent != 0:
               w_param_SIA_guessed = w_params_SIA_guessed[i]
            w_param_obfuscated = generate_full_param(encrypted_index, w_param_SIA_guessed, w_params_non_encrypted[i])
            w_locals.append(list_to_model_dict(empty_net.state_dict(), w_param_obfuscated))
        SIA_attack = SIA(args=args, w_locals=w_locals, dataset=dataset_train, dict_sia_users=dict_sample_user)
        attack_acc = SIA_attack.attack(net=empty_net.to('cpu'))#args.device
        att_acc_list.append(attack_acc)
        best_att_acc = max(best_att_acc, attack_acc)

        start_time = time.time()
        #<<SERVER>> aggregation
        #<DP>
        avg_w_params_encrypted = [sum(values) / len(values) for values in zip(*w_params_encrypted)]
        #<Plaintext>
        avg_w_params_non_encrypted = [sum(values) / len(values) for values in zip(*w_params_non_encrypted)]

        # <AGGREGATION> averaged HE ciphertext + averaged DP-noised plaintext => averaged param => w_glob
        param_glob = generate_full_param(encrypted_index, avg_w_params_encrypted, avg_w_params_non_encrypted) ##########################

        #param_glob => w_glob
        w_glob = list_to_model_dict(empty_net.state_dict(), param_glob)

        # copy weight to net_glob, which is sent to clients
        net_glob.load_state_dict(w_glob)

        #record time
        end_time = time.time()
        execution_time += end_time - start_time
        print(f'Operation time: {end_time - start_time}')

        acc_train, loss_train_ = test_fun(net_glob, dataset_train, args)
        # print loss
        loss_avg = sum(loss_locals) / len(loss_locals)
        print(f'Epoch Round {iter} End, Average training loss {loss_avg}')
        print('---\n')
        #end of Epoch

    # test
    net_glob.eval()

    acc_train, loss_train = test_fun(net_glob, dataset_train, args)
    acc_test, loss_test = test_fun(net_glob, dataset_test, args)
    # Experimental setting
    exp_details(args)

    print('Experimental result summary:')
    print(f'Execution time: {execution_time}')
    print(f'args.epsilon:', {args.epsilon})
    print(f'args.encrypt_percent:{args.encrypt_percent}')
    
    print("Training accuracy of the joint model: {:.2f}".format(acc_train))
    print("Testing accuracy of the joint model: {:.2f}".format(acc_test))
    
    print('Random guess baseline of source inference : {:.2f}'.format(1.0/args.num_users*100))
    print('Highest prediction loss based source inference accuracy: {:.2f}'.format(best_att_acc))
    print('Average prediction loss based source inference accuracy: {:.2f}'.format(sum(att_acc_list) / len(att_acc_list) if att_acc_list else 0))


if __name__ == '__main__':
    # parse
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    print(f'args.device:', {args.device})
    
    sys.stdout = open(f'main_fed_combine_dp.txt', 'w')

    epsilons = [1.48]
    for epsilon in epsilons:
        args.epsilon = epsilon
        print(f'epsilon={args.epsilon}===========================\n')

        ratios = [0, 0.2, 0.4, 0.6, 0.8, 1]
        for ratio in ratios:
            args.encrypt_percent = ratio
            print(f'encrypt_percent={args.encrypt_percent}-------\n')
            process(args)
            print(f'-------encrypt_percent={args.encrypt_percent}\n')
        
        print(f'===========================epsilon={args.epsilon}\n')
    
    sys.stdout.close()