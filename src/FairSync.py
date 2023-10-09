#coding:utf-8
import argparse
import json
import math
import os
import random
import shutil
import sys
import time
from collections import defaultdict

from tqdm import tqdm, trange

import numpy as np

import faiss
import tensorflow as tf
from data_iterator import DataIterator
from utils import *
from tensorboardX import SummaryWriter

import torch
import torch.optim

import pandas as pd

best_metric = 0



def evaluate_full_mmf(sess, test_data, model, model_path, batch_size, item_cate_map,
                  p_num, provider_matrix=None, save=True, coef=None, minimum_exposure=10, topN=20, lr=1e-2):
    items = np.array(list(range(1, len(item_cate_map))))
    item_num, p_emb = provider_matrix.shape
    pid = []
    for iid in items:
        pid.append(item_cate_map[iid])
    pid = np.array(pid)
    p_features = provider_matrix[pid]
    p_features = np.concatenate(([np.zeros(p_emb)], p_features), axis=0)
    p_features = p_features.astype(np.float32)

    item_embs = model.output_item(sess)
    revised_embs = np.concatenate((item_embs,p_features),axis=-1)

    res = faiss.StandardGpuResources()
    flat_config = faiss.GpuIndexFlatConfig()
    flat_config.device = 0

    try:
        gpu_index = faiss.GpuIndexFlatIP(res, args.embedding_dim + p_emb, flat_config)
        gpu_index.add(revised_embs)

    except Exception as e:
        print(e)

    total = 0
    total_recall = 0.0
    total_ndcg = 0.0
    total_hitrate = 0
    total_map = 0.0
    total_diversity = 0.0

    exposures_list = np.zeros(p_num)
    mu = torch.zeros(p_num, device='cuda', requires_grad=True)
    torch_vector = torch.tensor(provider_matrix, device='cuda')
    if dataset == 'book':
        test_len = 60367
    else:
        test_len = 97678
        #test_len = 10000
    optimizer = torch.optim.Adam([mu, ], lr=lr)

    print("**************************")
    assert minimum_exposure <= topN * test_len / p_num
    index = 0
    print("testing....")
    for src, tgt in tqdm(test_data):
        index += 1
        if index > test_len:
            break
        #mu = mu.detach().numpy()

        #query = torch.matmul(mu,torch_vector).cpu().detach().numpy()
        query = mu.cpu().detach().numpy()
        loss = 0.0
        nick_id, item_id, hist_item, hist_mask = prepare_data(src, tgt)

        user_embs = model.output_user(sess, [hist_item, hist_mask])
        batch_size = len(user_embs)

        if len(user_embs.shape) == 2:
            query = np.tile(query,(batch_size,1))
            user_embs = np.concatenate((user_embs,-query),axis=-1)
            #print(user_embs.shape)
            D, I = gpu_index.search(user_embs, topN)
            for i, iid_list in enumerate(item_id):
                recall = 0
                dcg = 0.0
                true_item_set = set(iid_list)
                for no, iid in enumerate(I[i]):

                    pid = item_cate_map[iid]
                    exposures_list[pid] += 1
                    loss = loss + D[i][no] - mu[pid]
                    if iid in true_item_set:
                        recall += 1
                        dcg += 1.0 / math.log(no + 2, 2)
                idcg = 0.0
                for no in range(recall):
                    idcg += 1.0 / math.log(no + 2, 2)
                total_recall += recall * 1.0 / len(iid_list)
                if recall > 0:
                    total_ndcg += dcg / idcg
                    total_hitrate += 1
                if not save:
                    total_diversity += compute_diversity(I[i], item_cate_map)
        else:
            ni = user_embs.shape[1]
            query = np.tile(np.array(query),reps=(batch_size,ni,1))
            user_embs = np.concatenate((user_embs, -query), axis=-1)
            user_embs = np.reshape(user_embs, [-1, user_embs.shape[-1]])
            D, I = gpu_index.search(user_embs, topN)

            for i, iid_list in enumerate(item_id):
                recall = 0
                dcg = 0.0
                item_list_set = set()
                item_cor_list = []

                item_list = list(zip(np.reshape(I[i * ni:(i + 1) * ni], -1), np.reshape(D[i * ni:(i + 1) * ni], -1)))
                item_list.sort(key=lambda x: x[1], reverse=True)
                for j in range(len(item_list)):
                    if item_list[j][0] not in item_list_set and item_list[j][0] != 0:
                        item_list_set.add(item_list[j][0])
                        item_cor_list.append(item_list[j][0])
                        if len(item_list_set) >= topN:
                            break

                true_item_set = set(iid_list)
                for no, iid in enumerate(item_cor_list):
                    pid = item_cate_map[iid]
                    loss = loss - mu[pid]
                    exposures_list[pid] += 1
                    if iid in true_item_set:
                        recall += 1
                        dcg += 1.0 / math.log(no + 2, 2)
                idcg = 0.0
                for no in range(recall):
                    idcg += 1.0 / math.log(no + 2, 2)
                total_recall += recall * 1.0 / len(iid_list)
                if recall > 0:
                    total_ndcg += dcg / idcg
                    total_hitrate += 1
                if not save:
                    total_diversity += compute_diversity(list(item_list_set), item_cate_map)

        loss += (torch.sum(minimum_exposure * mu) + torch.max(mu) * (test_len*topN-minimum_exposure*p_num))/test_len
        optimizer.zero_grad()
        #loss =
        loss.backward()
        optimizer.step()
        total += len(item_id)

    recall = total_recall / total
    ndcg = total_ndcg / total
    hitrate = total_hitrate * 1.0 / total
    diversity = total_diversity * 1.0 / total
    #print("index:",index)
    ESP = np.sum(exposures_list >= minimum_exposure).astype(np.float) / p_num

    # if save:
    #     return {'recall': recall, 'ndcg': ndcg, 'hitrate': hitrate, 'mmf': mmf}
    return {'recall': recall, 'ndcg': ndcg, 'hitrate': hitrate, 'diversity': diversity, 'ESP': ESP}

def test_mmf(
        test_file,
        cate_file,
        item_count,
        dataset="book",
        batch_size=128,
        maxlen=100,
        model_type='DNN',
        lr=0.001,
        minimum_exposure=10,
        topN = 20,
        mmf_lr = 1e-2,
        eval_batch_size = 1,
):
    para_name = '_'.join([dataset, model_type, 'b' + str(batch_size), 'lr' + str(lr), 'd' + str(64),
                          'len' + str(maxlen)])
    exp_name = para_name + '_' + "test"
    # exp_name = "test"
    best_model_path = "best_model/" + exp_name + '/'
    gpu_options = tf.GPUOptions(allow_growth=True)
    model = get_model(model_type, item_count, batch_size, maxlen)
    item_cate_map, p_num = load_item_cate(cate_file)
    p_matrix = np.eye(p_num).astype(np.float32)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        model.restore(sess, best_model_path)

        test_data = DataIterator(test_file, eval_batch_size, maxlen, train_flag=2)
        result = evaluate_full_mmf(sess, test_data, model, best_model_path, eval_batch_size, item_cate_map, p_num,
                                    p_matrix,
                                    save=False, coef=args.coef, minimum_exposure=minimum_exposure, topN=topN, lr=mmf_lr)
        print(result)
        #return times

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--p', type=str, default='test', help='test_mmf | test')
    parser.add_argument('--dataset', type=str, default='book', help='book | taobao')
    parser.add_argument('--random_seed', type=int, default=19)
    parser.add_argument('--embedding_dim', type=int, default=64)
    parser.add_argument('--hidden_size', type=int, default=64)
    parser.add_argument('--num_interest', type=int, default=4)
    parser.add_argument('--model_type', type=str, default='ComiRec-DR', help='DNN | GRU4REC | MIND | ComiRec-DR | ComiRec-SA')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='')
    parser.add_argument('--max_iter', type=int, default=1000, help='(k)')
    parser.add_argument('--patience', type=int, default=200)
    parser.add_argument('--coef', default=None)
    parser.add_argument('--minimum_exposure', type=float, default=200, help='minimum exposures for each cate')
    parser.add_argument('--topN', type=int, default=50)
    parser.add_argument('--FairSync_lr', type=float, default=1e-2)
    parser.add_argument('--eval_batch_size', type=int, default=1)

    print(sys.argv)
    args = parser.parse_args()


    train_name = 'train'
    valid_name = 'valid'
    test_name = 'test'

    if args.dataset == 'taobao':
        path = './data/taobao_data/'
        item_count = 1708531
        batch_size = 256
        maxlen = 50
        test_iter = 500
    elif args.dataset == 'book':
        path = './data/book_data/'
        item_count = 367983
        batch_size = 128
        maxlen = 20
        test_iter = 1000
    
    train_file = path + args.dataset + '_train.txt'
    valid_file = path + args.dataset + '_valid.txt'
    test_file = path + args.dataset + '_test.txt'
    cate_file = path + args.dataset + '_item_cate.txt'
    dataset = args.dataset

    args.p = 'test_mmf'

    result = {}
    #for dataset in ['book', 'taobao']:
    for dataset in ['book']:
        args.dataset = dataset
        train_name = 'train'
        valid_name = 'valid'
        test_name = 'test'

        if args.dataset == 'taobao':
            path = './data/taobao_data/'
            item_count = 1708531
            batch_size = 256
            maxlen = 50
            test_iter = 500
        elif args.dataset == 'book':
            path = './data/book_data/'
            item_count = 367983
            batch_size = 128
            maxlen = 20
            test_iter = 1000

        train_file = path + args.dataset + '_train.txt'
        valid_file = path + args.dataset + '_valid.txt'
        test_file = path + args.dataset + '_test.txt'
        cate_file = path + args.dataset + '_item_cate.txt'


        SEED = args.random_seed

        tf.set_random_seed(SEED)
        np.random.seed(SEED)
        random.seed(SEED)
        test_mmf(test_file=test_file, cate_file=cate_file, item_count=item_count, dataset=dataset, batch_size=batch_size, maxlen=maxlen,
   model_type=args.model_type, lr=args.learning_rate, minimum_exposure=args.minimum_exposure,
                           topN=args.topN, mmf_lr=args.FairSync_lr, eval_batch_size=args.eval_batch_size)