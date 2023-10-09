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
from model import *
from utils import *
from tensorboardX import SummaryWriter

import torch
import torch.optim

import pandas as pd

best_metric = 0



def evaluate_full(sess, test_data, model, model_path, batch_size, item_cate_map, minimum_exposure_list, topN,
                  p_num, save=True, coef=None):
    #topN = args.topN

    item_embs = model.output_item(sess)
    #print(type(item_embs))
    #print(item_embs.shape)

    res = faiss.StandardGpuResources()
    flat_config = faiss.GpuIndexFlatConfig()
    flat_config.device = 0

    try:
        gpu_index = faiss.GpuIndexFlatIP(res, args.embedding_dim, flat_config)
        gpu_index.add(item_embs)
    except Exception as e:
        return {}

    total = 0
    total_recall = 0.0
    total_ndcg = 0.0
    total_hitrate = 0
    total_map = 0.0
    total_diversity = 0.0


    test_len = 1000000
    exposures_list = np.zeros(p_num)
    index = 0
    for src, tgt in tqdm(test_data):
        index += 1
        if index > test_len:
            break
        nick_id, item_id, hist_item, hist_mask = prepare_data(src, tgt)

        user_embs = model.output_user(sess, [hist_item, hist_mask])

        if len(user_embs.shape) == 2:
            D, I = gpu_index.search(user_embs, topN)
            for i, iid_list in enumerate(item_id):
                recall = 0
                dcg = 0.0
                true_item_set = set(iid_list)
                for no, iid in enumerate(I[i]):
                    if iid != 0:
                        exposures_list[item_cate_map[iid]] += 1
                    if iid in true_item_set:
                        recall += 1
                        dcg += 1.0 / math.log(no+2, 2)
                idcg = 0.0
                for no in range(recall):
                    idcg += 1.0 / math.log(no+2, 2)
                total_recall += recall * 1.0 / len(iid_list)
                if recall > 0:
                    total_ndcg += dcg / idcg
                    total_hitrate += 1
                if not save:
                    total_diversity += compute_diversity(I[i], item_cate_map)
        else:
            ni = user_embs.shape[1]
            user_embs = np.reshape(user_embs, [-1, user_embs.shape[-1]])
            times = time.time()
            D, I = gpu_index.search(user_embs, topN)
            consume_time = time.time() - times
            return consume_time
            for i, iid_list in enumerate(item_id):
                recall = 0
                dcg = 0.0
                item_list_set = set()
                item_cor_list = []

                item_list = list(zip(np.reshape(I[i*ni:(i+1)*ni], -1), np.reshape(D[i*ni:(i+1)*ni], -1)))
                item_list.sort(key=lambda x:x[1], reverse=True)
                for j in range(len(item_list)):
                    if item_list[j][0] not in item_list_set and item_list[j][0] != 0:
                        item_list_set.add(item_list[j][0])
                        item_cor_list.append(item_list[j][0])
                        if len(item_list_set) >= topN:
                            break

                true_item_set = set(iid_list)
                for no, iid in enumerate(item_cor_list):
                    exposures_list[item_cate_map[iid]] += 1
                    if iid in true_item_set:
                        recall += 1
                        dcg += 1.0 / math.log(no+2, 2)
                idcg = 0.0
                for no in range(recall):
                    idcg += 1.0 / math.log(no+2, 2)
                total_recall += recall * 1.0 / len(iid_list)
                if recall > 0:
                    total_ndcg += dcg / idcg
                    total_hitrate += 1
                if not save:
                    total_diversity += compute_diversity(list(item_list_set), item_cate_map)
        
        total += len(item_id)
    
    recall = total_recall / total
    ndcg = total_ndcg / total
    hitrate = total_hitrate * 1.0 / total
    diversity = total_diversity * 1.0 / total

    metrics = {'recall': recall, 'ndcg': ndcg, 'hitrate': hitrate, 'diversity': diversity}
    for topk in minimum_exposure_list:
        metrics.update({"mmf@"+str(topk):np.sum(exposures_list>=topk).astype(np.float)/p_num})
    #print(mmf_list)
    print("test_len:",index)
    # if save:
    #     return {'recall': recall, 'ndcg': ndcg, 'hitrate': hitrate}.update(mmf_list)


    return metrics


def test(
        test_file,
        cate_file,
        item_count,
        dataset = "book",
        batch_size = 128,
        maxlen = 100,
        model_type = 'DNN',
        lr = 0.001,
        minimum_exposure=10,
        topN = 20,
        eval_batch_size = 1,
):
    #exp_name = get_exp_name(dataset, model_type, batch_size, lr, maxlen, save=False)
    para_name = '_'.join([dataset, model_type, 'b' + str(batch_size), 'lr' + str(lr), 'd' + str(64),
                          'len' + str(maxlen)])
    exp_name = para_name + '_' + "test"
    #exp_name = "test"
    best_model_path = "best_model/" + exp_name + '/'
    gpu_options = tf.GPUOptions(allow_growth=True)
    model = get_model(model_type, item_count, batch_size, maxlen)
    item_cate_map, p_num = load_item_cate(cate_file)

    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        model.restore(sess, best_model_path)
        
        #test_data = DataIterator(test_file, batch_size, maxlen, train_flag=2)
        test_data = DataIterator(test_file, eval_batch_size, maxlen, train_flag=2)
        minimum_exposure_list = [5,10, 20,30, 50, 100, 150, 200]
        metrics = evaluate_full(sess, test_data, model, best_model_path, batch_size,
                                item_cate_map, minimum_exposure_list, topN, p_num, save=False, coef=None)
        #print(', '.join(['test ' + key + ': %.6f' % value for key, value in metrics.items()]))
        return metrics


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--p', type=str, default='test', help='test_mmf | test')
    parser.add_argument('--dataset', type=str, default='book', help='book | taobao')
    parser.add_argument('--random_seed', type=int, default=19)
    parser.add_argument('--embedding_dim', type=int, default=64)
    parser.add_argument('--hidden_size', type=int, default=64)
    parser.add_argument('--num_interest', type=int, default=4)
    parser.add_argument('--model_type', type=str, default='DNN', help='DNN | GRU4REC | MIND | ComiRec-DR | ComiRec-SA')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='')
    parser.add_argument('--max_iter', type=int, default=1000, help='(k)')
    parser.add_argument('--patience', type=int, default=200)
    parser.add_argument('--coef', default=None)
    parser.add_argument('--minimum_exposure', type=float, default=10, help='minimum exposures for each cate')
    parser.add_argument('--topN', type=int, default=50)
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

    args.p = 'test'
    result = {}
    times = []

    for topN in [20,50]:
        for eval_batch_size in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]:
        #for model_type in ["DNN", "GRU4REC", "MIND", "ComiRec-DR", "ComiRec-SA"]:
            for model_type in ["ComiRec-DR"]:
                SEED = args.random_seed

                tf.set_random_seed(SEED)
                np.random.seed(SEED)
                random.seed(SEED)
                print("model_type: %s top N: %d"%(model_type,topN))
                args.model_type = model_type
                args.topN = topN
                metric = test(test_file=test_file, cate_file=cate_file, item_count=item_count, dataset=dataset, batch_size=batch_size,
                          maxlen=maxlen, model_type=args.model_type, lr=args.learning_rate,
                              minimum_exposure=args.minimum_exposure, topN=args.topN, eval_batch_size=eval_batch_size)
                # print("==================")
                # result[model_type+"_"+"topN@"+str(topN)] = metric
                times.append(metric)
                tf.compat.v1.reset_default_graph()
    print(times)
    exit(0)
    # with open("result/{}_basemodel.json".format(args.dataset), "w") as json_file:
    #     json.dump(result, json_file)
    # print(result)

    # if args.p == 'test':
    #     test(test_file=test_file, cate_file=cate_file, item_count=item_count, dataset=dataset, batch_size=batch_size,
    #          maxlen=maxlen, model_type=args.model_type, lr=args.learning_rate, minimum_exposure=args.minimum_exposure)
    # elif args.p == 'test_mmf':
    #     test_mmf(test_file=test_file, cate_file=cate_file, item_count=item_count, dataset=dataset, batch_size=batch_size, maxlen=maxlen,
    #            model_type=args.model_type, lr=args.learning_rate,p_embedding=args.p_embedding, minimum_exposure=args.minimum_exposure)
    # else:
    #     print('do nothing...')
