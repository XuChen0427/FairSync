import pandas as pd
import numpy as np
import os
from model import *

def build_map(df,col_name):
    key = df[col_name].unique().tolist()
    m = dict(zip(key, range(len(key))))
    decode = dict(zip(range(len(key)),key))
    df[col_name] = df[col_name].apply(lambda x: m[x])
    return m, len(key)

def prepare_data(src, target):
    nick_id, item_id = src
    hist_item, hist_mask = target
    return nick_id, item_id, hist_item, hist_mask

def load_item_cate(source):
    #item_cate = {}

    item2cat = pd.read_csv(source,
                           delimiter=',',
                           dtype={'item_id':int, 'cate_id': str})


    decode, p_num = build_map(item2cat, 'cate_id')
    print("p_num: ",p_num)
    item_cate = item2cat.set_index(['item_id'])['cate_id'].to_dict()
    item_cate[0] = decode['-1']

    return item_cate, p_num

def compute_diversity(item_list, item_cate_map):
    n = len(item_list)
    diversity = 0.0
    for i in range(n):
        for j in range(i+1, n):
            diversity += item_cate_map[item_list[i]] != item_cate_map[item_list[j]]
    diversity /= ((n-1) * n / 2)
    return diversity

def get_model(model_type, item_count, batch_size, maxlen, embedding_dim=64, hidden_size=64, num_interest=4):
    if model_type == 'DNN':
        model = Model_DNN(item_count, embedding_dim, hidden_size, batch_size, maxlen)
    elif model_type == 'GRU4REC':
        model = Model_GRU4REC(item_count, embedding_dim, hidden_size, batch_size, maxlen)
    elif model_type == 'MIND':
        #relu_layer = True if dataset == 'book' else False
        relu_layer = False
        model = Model_MIND(item_count, embedding_dim, hidden_size, batch_size, num_interest, maxlen, relu_layer=relu_layer)
    elif model_type == 'ComiRec-DR':
        model = Model_ComiRec_DR(item_count, embedding_dim, hidden_size, batch_size, num_interest, maxlen)
    elif model_type == 'ComiRec-SA':
        model = Model_ComiRec_SA(item_count, embedding_dim, hidden_size, batch_size, num_interest, maxlen)
    else:
        print ("Invalid model_type : %s", model_type)
        return
    return model

