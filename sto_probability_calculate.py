# -*- coding: utf-8 -*-
'''
@Date: 2022/8/26
@Time: 15:06
@Author: Wenzheng Song
@Email: gyswz123@gmail.com
'''
# -*- coding: utf-8 -*-
'''
@Date: 2022/4/19
@Time: 13:40
@Author: Wenzheng Song
@Email: gyswz123@gmail.com
'''
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn as nn

import os
import dgl
import networkx as nx
from sklearn.metrics import roc_auc_score, average_precision_score
import json
import time
from tqdm import tqdm
import numpy as np
import torch.nn.functional as F
import scipy.sparse as sp

#from dgl.nn.tensorflow.conv import SAGEConv
#change to pytorch
from dgl.nn.pytorch import SAGEConv

import itertools
from umls_get import concept_check
import biobert_encoder
import config_struct
from sto_config import ConfigStore
import sys
from torch.utils.tensorboard import SummaryWriter
import random
from datetime import datetime
from tree_tools import tree_model

import pickle

tree = tree_model.tree
# print(tree)
seed = int(time.time())
seed = 1663763962
print(seed)
np.random.seed(seed)
dgl.seed(seed)
torch.manual_seed(seed)
random.seed(seed)
s1 = time.time()
# -------------#1-config-----------------------------------
exp_name = '{}-prediction'.format('sto')
# batch_size = 150000
batch_size = 8192
device = 0
config_store = ConfigStore(exp_name)
f_console_log = open(config_store.dic_config['console_log_path'], 'w')
# sys.stdout = f_console_log
writer = SummaryWriter(config_store.dic_config['tensorboard_path'])
visual = True
# -------------#1-1data path config-----------------------------------
#dataset_name = 'select_strict'
dataset_name = 'openbiolink'

gtrain_path = r'./raw_data/{}/myGraph_all.gpickle'.format(dataset_name)
edict_path = r'./raw_data/{}/entity_dictionary_0.json'.format(dataset_name)
transl_path = r'./raw_data/{}/index_translator.json'.format(dataset_name)
reldic_path = r'./raw_data/rel_dic.json'
gtest_path = r'./raw_data/{}/myGraph_val.gpickle'.format(dataset_name)
pretrained_model = r'./raw_data/pretrained_biobert'
type_dict_path = r'./raw_data/{}/type_dict.json'.format(dataset_name)
type_name_pth = r'./raw_data/{}/type_all.json'.format(dataset_name)
# -------------#1-2hyper paremeters config-----------------------------------
graph_type = 'homo'
neg_ratio = 50
aggregation_method = 'mean'
learning_rate = 0.001
embedding_size = 64
embed_init = 'biobert_calculated'
epoches = 1000
graph_hidden_size = 128
neighbour_ratio = [25, 10]
ablation_Marg = 0
ablation_Ance = 0
ablation_RevEdge = 0
min_epoches = 50

if os.path.exists(config_store.dic_config['paras_dict_json_path']):
    print('{} exists, loading.'.format(config_store.dic_config['paras_dict_json_path']))
    with open(config_store.dic_config['paras_dict_json_path'], 'r') as fp:
        hyper_para = json.load(fp)
else:
    print('{} not exist, creating.'.format(config_store.dic_config['paras_dict_json_path']))
    hyper_para = {
        'neg_ratio': neg_ratio,
        'aggregation_method': aggregation_method,
        'learning_rate': learning_rate,
        'embedding_size': embedding_size,
        'embed_init': embed_init,
        'epoches': epoches,
        'graph_hidden_size': graph_hidden_size,
        'neighbour_ratio': neighbour_ratio,
        'graph_type': graph_type,
        'ablation_Marg' : ablation_Marg,
        'ablation_Ance' : ablation_Ance,
        'min_epoches': min_epoches,
        'ablation_RevEdge': ablation_RevEdge,
    }
    with open(config_store.dic_config['paras_dict_json_path'], 'w') as fp:
        print(json.dump(hyper_para, fp))

neg_ratio = hyper_para['neg_ratio']
aggregation_method = hyper_para['aggregation_method']
learning_rate = hyper_para['learning_rate']
embedding_size = hyper_para['embedding_size']
embed_init = hyper_para['embed_init']
epoches = hyper_para['epoches']
graph_hidden_size = hyper_para['graph_hidden_size']
neighbour_ratio = hyper_para['neighbour_ratio']
graph_type = hyper_para['graph_type']
ablation_Marg = hyper_para['ablation_Marg']
ablation_Ance = hyper_para['ablation_Ance']
ablation_RevEdge = hyper_para['ablation_RevEdge']
min_epoches = hyper_para['min_epoches']
# -------------#2-data preparation-----------------------------------
# -------------#2-1-data loading-----------------------------------
with open(edict_path, 'r') as fp:
    entity_dict = json.load(fp)

with open(transl_path, 'r') as fp:
    translator = json.load(fp)

with open(reldic_path, 'r') as fp:
    rel_dic = json.load(fp)

with open(type_dict_path, 'r') as fp:
    type_dict = json.load(fp)

with open(type_name_pth, 'r') as fp:
    type_name_list = json.load(fp)

print('loading nx graph...')
from scipy import sparse
# read_gpickle has been removed in 3.x version
#G_train = nx.read_gpickle(gtrain_path)
with open(gtrain_path, 'rb') as f:
    G_train = pickle.load(f)

# 为啥要这么写？ 2025/6/16
## 想要调研的实体的CID或者entrez_gene_id
#cid_input = 'C0004352'
## 想要调研的实体的CID或者entrez_gene_id --end
#inner_id = entity_dict[cid_input]
#I = [inner_id] * len(entity_dict.keys())
#J = list(entity_dict.values())
#V = [1] * len(entity_dict.keys())
#length_entity = len(entity_dict.keys())
#
#A = sparse.coo_matrix((V, (I, J)), shape=[length_entity, length_entity])
##G_test = nx.from_scipy_sparse_matrix(A, create_using=nx.DiGraph())
#G_test = nx.from_scipy_sparse_array(A, create_using=nx.DiGraph)
#不是很懂 2025/6/16

#G_test = nx.read_gpickle(gtest_path)
with open(gtest_path, 'rb') as f:
    G_test = pickle.load(f)

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

s2 = time.time()
print('graph loaded', str(s2 - s1))
#g_train = dgl.from_networkx(G_train, edge_attrs=['weight']).to(device)
#g_test = dgl.from_networkx(G_test, edge_attrs=['weight']).to(device)
g_train = dgl.from_networkx(G_train).to(device)
g_test = dgl.from_networkx(G_test).to(device)


if graph_type == 'homo':
    if not ablation_RevEdge == 1:
        g_train = dgl.add_reverse_edges(g_train).to(device)
        g_test = dgl.add_reverse_edges(g_test).to(device)
    #g_train.edata['weight']=torch.tensor([1]*g_train.number_of_edges()).to(device)
    #g_test.edata['weight']=torch.tensor([1]*g_test.number_of_edges()).to(device)

s3 = time.time()
type_list = torch.tensor([])
type_hexist = False
type_name_nodes = torch.tensor([])
# optimize this code

if not (ablation_Marg == 1 or ablation_Ance == 1):
    type_list = torch.tensor(list(type_dict.values()))
    g_train = g_train.to(device)
    g_train.ndata['hierarchy_type'] = type_list.to(device)

# for i in tqdm(g_train.nodes()):
#     if type_hexist:
#         tensor2add = torch.tensor(type_dict[str(int(i))]).reshape([1, -1])
#         type_list = torch.concat((type_list, tensor2add), 0)
#     else:
#         type_list = torch.tensor(type_dict[str(int(i))]).reshape([1, -1])
#         type_hexist = True
# g_train.ndata['hierarchy_type'] = type_list.to(device)
# # optimize this code end
print('dgl constructed', str(s3 - s2))
# -------------#2-2-data split-----------------------------------
u, v = g_train.edges()
u_test, v_test = g_test.edges()
train_eids = np.arange(g_train.number_of_edges())
train_eids = np.random.permutation(train_eids)
train_size = g_train.number_of_edges()
train_pos_u, train_pos_v = u[train_eids], v[train_eids]

test_eids = np.arange(g_test.number_of_edges())
test_eids = np.random.permutation(test_eids)
test_size = g_test.number_of_edges()
test_pos_u, test_pos_v = u_test[test_eids], v_test[test_eids]
print(test_pos_u, test_pos_v)


# -------------#2-3-negdata generation-----------------------------------
negative_sampler = dgl.dataloading.negative_sampler.GlobalUniform(neg_ratio)
sampler = dgl.dataloading.NeighborSampler(neighbour_ratio)
sampler = dgl.dataloading.as_edge_prediction_sampler(sampler, negative_sampler=negative_sampler)

adj = sp.coo_matrix(((np.ones(len(u))), (u.cpu().numpy(), v.cpu().numpy())), shape=[len(g_train.nodes()), len(g_train.nodes())])
csr_adj = adj.tocsr()

adj_test = sp.coo_matrix(((np.ones(len(u_test))), (u_test.cpu().numpy(), v_test.cpu().numpy())),
                         shape=[len(g_test.nodes()),len(g_test.nodes())])
csr_adj_test = adj_test.tocsr()


def neg_sampling(g, pos_adj=None, sample_ratio=None, pos_size=None, config=config_store, dataname=None):
    data_key_name = '{}_neg_uv_tuple_path'.format(dataname)
    path_exists = os.path.exists(config.dic_config[data_key_name])
    if not path_exists:
        print('{} {} not existing and start negative sampling from scratch.'.format(config.name, data_key_name))
        n_return = []
        u_neg = []
        v_neg = []
        neg_size = int(sample_ratio * pos_size)
        for i in tqdm(range(0, neg_size)):
            n_t = neg_iter(g.nodes(), pos_adj)
            while (n_t[0], n_t[1]) in n_return:
                n_t = neg_iter(g.nodes(), pos_adj)
            n_return.append((n_t[0], n_t[1]))
            u_neg.append(n_t[0])
            v_neg.append(n_t[1])
        print('{} {} start storing.'.format(config.name, data_key_name))
        torch.save((torch.tensor(u_neg), torch.tensor(v_neg)), config.dic_config[data_key_name])
        print('{} data stored at {}'.format(config.name, config.dic_config[data_key_name]))
        return torch.tensor(u_neg), torch.tensor(v_neg)
    else:
        print('{} {} existing and start loading negative sampling.'.format(config.name, data_key_name))
        neg_uv_tuple = torch.load(config.dic_config[data_key_name], map_location=torch.device(device))
        print('{} {} has loaded.'.format(config.name, data_key_name))
        return neg_uv_tuple[0], neg_uv_tuple[1]

def neg_iter(nodes, pos_adj):
    nodes = nodes.cpu().numpy()
    edge = np.random.choice(nodes, size=(2,))
    while edge[0] == edge[1] or pos_adj[edge[0], edge[1]] != 0:
        edge = np.random.choice(nodes, size=(2,))
    return edge

def dgl_neg_sampling(g, pos_adj=None, sample_ratio=None, pos_size=None, config=config_store, dataname=None):
    data_key_name = '{}_neg_uv_tuple_path'.format(dataname)
    path_exists = os.path.exists(config.dic_config[data_key_name])
    if not path_exists:
        print('{} {} not existing and start negative sampling from scratch.'.format(config.name, data_key_name))
        neg_sampler_dgl = dgl.dataloading.negative_sampler.GlobalUniform(sample_ratio)
        neg_edges_dgl = neg_sampler_dgl(g, torch.tensor(range(0, g_test.number_of_edges())))
        u_neg, v_neg = neg_edges_dgl[0], neg_edges_dgl[1]
        print('{} {} start storing.'.format(config.name, data_key_name))
        torch.save((u_neg, v_neg), config.dic_config[data_key_name])
        print('{} data stored at {}'.format(config.name, config.dic_config[data_key_name]))
        return u_neg, v_neg
    else:
        print('{} {} existing and start loading negative sampling.'.format(config.name, data_key_name))
        neg_uv_tuple = torch.load(config.dic_config[data_key_name], map_location=torch.device(device))
        print('{} {} has loaded.'.format(config.name, data_key_name))
        return neg_uv_tuple[0], neg_uv_tuple[1]

# test_neg_u, test_neg_v = neg_sampling(g_test, pos_adj=csr_adj_test, sample_ratio=1, pos_size=g_test.number_of_edges(), dataname='test')
test_neg_u, test_neg_v = dgl_neg_sampling(g_test, pos_adj=csr_adj_test, sample_ratio=1, pos_size=g_test.number_of_edges(), dataname='test')
# shuffle
neg_test_eids = np.arange(int(1 * g_test.number_of_edges()))
neg_test_eids = np.random.permutation(neg_test_eids)

print("Original neg_test_eids shape:", neg_test_eids.shape)
print("test_neg_u shape:", test_neg_u.shape)
print("g_test.number_of_edges():", g_test.number_of_edges())


print("test_neg_u.shape:", test_neg_u.shape)
print("test_neg_v.shape:", test_neg_v.shape)
print("neg_test_eids.shape:", neg_test_eids.shape)
print("neg_test_eids.max():", neg_test_eids.max())
print("neg_test_eids.min():", neg_test_eids.min())

#test_neg_u, test_neg_v = test_neg_u[neg_test_eids], test_neg_v[neg_test_eids]
# --- 安全处理负样本索引 ---
if isinstance(neg_test_eids, np.ndarray):
    neg_test_eids = torch.tensor(neg_test_eids, dtype=torch.long)

max_idx = test_neg_u.shape[0]
neg_test_eids = neg_test_eids[neg_test_eids < max_idx]  # 避免越界

test_neg_u = test_neg_u[neg_test_eids]
test_neg_v = test_neg_v[neg_test_eids]
# -------------#2-4-node features initialization-----------------------------------
# from transformers import AutoTokenizer, AutoModel
# def model_bert(nn.Module):
#     tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=pretrained_model)
#     model_bio = AutoModel.from_pretrained(pretrained_model_name_or_path=pretrained_model)
#     model_bio = model_bio.to(device)
def feature_init(initm, nodes_size, size=embedding_size):
    if initm == 'random':
        return torch.tensor(np.random.rand(nodes_size, size), dtype=torch.float32).to(device)
    if initm == 'onehot':
        u_embed = np.arange(len(g_train.nodes()))
        onehot_sp = sp.coo_matrix(((np.ones(len(g_train.nodes()))),(u_embed, u_embed)), shape=[len(g_train.nodes()), len(g_train.nodes())])
        values = onehot_sp.data
        indices = np.vstack((onehot_sp.row, onehot_sp.col))
        i = torch.LongTensor(indices)
        v = torch.FloatTensor(values)
        shape = onehot_sp.shape
        return torch.sparse.FloatTensor(i, v, torch.Size(shape)).to_dense().to(device)
    if initm == 'biobert':
        tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=pretrained_model)
        # model_bio = AutoModel.from_pretrained(pretrained_model_name_or_path=pretrained_model)
        # model_bio = model_bio.to(device)
        tensor_exist = False
        torch_return = torch.tensor([])
        print('biobert initializing...')
        with open('./raw_data/mg_data/mg_dict_store_1.json', 'r') as fp:
            mg_dict = json.load(fp)
        for i in tqdm(g_train.nodes()):
            tensor2use = biobert_encoder.token_code(translator[str(int(i))], tokenizer, mg_dict, device)
            if tensor_exist:
                torch_return = torch.cat((torch_return, tensor2use), 0)
            else:
                torch_return = tensor2use
                tensor_exist = True
        return torch_return.to(device)
    if initm == 'biobert_calculated':
        tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=pretrained_model)
        model_bio = AutoModel.from_pretrained(pretrained_model_name_or_path=pretrained_model)
        model_bio = model_bio.to(device)
        tensor_exist = False
        torch_return = torch.tensor([])
        print('biobert initializing...')
        with open('./raw_data/mg_dict_store_1.json', 'r') as fp:
            mg_dict = json.load(fp)
        for i in tqdm(g_train.nodes()):
            tensor2use = biobert_encoder.encode(translator[str(int(i))], tokenizer, model_bio, mg_dict)
            if tensor_exist:
                torch_return = torch.cat((torch_return, tensor2use), 0)
            else:
                torch_return = tensor2use
                tensor_exist = True
        torch_return.to(device)
        return torch_return


g_train = g_train.to(device)
g_test = g_test.to(device)
# change at 2025/6/10 begin
if embed_init == 'biobert_calculated':
    if not os.path.exists(config_store.dic_config['biobert_embedding_path']):
        print('biobert_embedding not existing ,creating from scratch...')
        g_train.ndata['feat'] = feature_init(embed_init, g_train.number_of_nodes())
        g_test.ndata['feat'] = feature_init(embed_init, g_test.number_of_nodes())
        torch.save(g_train.ndata['feat'], config_store.dic_config['biobert_embedding_path'])
        print('biobert_tokenembedding saved')
    else:
        print('biobert_embedding existing ,loading...')
        g_train.ndata['feat'] = torch.load(config_store.dic_config['biobert_embedding_path'], map_location=torch.device(device)).to(device)
        g_test.ndata['feat'] = g_train.ndata['feat']
        print('biobert_embedding loaded')

elif (embed_init == 'random'):
    if not os.path.exists(config_store.dic_config['random_embedding_path']):
        print('random_embedding_path not existing ,creating from scratch...')
        g_train.ndata['feat'] = feature_init(embed_init, g_train.number_of_nodes())
        g_test.ndata['feat'] = feature_init(embed_init, g_test.number_of_nodes())
        torch.save(g_train.ndata['feat'], config_store.dic_config['random_embedding_path'])
        print('random_tokenembedding saved')
    else:
        print('random_embedding_path existing ,loading...')
        g_train.ndata['feat'] = torch.load(config_store.dic_config['random_embedding_path'], map_location=torch.device(device)).to(device)
        g_test.ndata['feat'] = g_train.ndata['feat']
        print('random_embedding_path loaded')

elif (embed_init == 'onehot'): #not suggest at all
    if not os.path.exists(config_store.dic_config['onehot_embedding_path']):
        print('onehot_embedding_path not existing ,creating from scratch...')
        g_train.ndata['feat'] = feature_init(embed_init, g_train.number_of_nodes())
        g_test.ndata['feat'] = feature_init(embed_init, g_test.number_of_nodes())
        torch.save(g_train.ndata['feat'], config_store.dic_config['onehot_embedding_path'])
        print('onehot_tokenembedding saved')
    else:
        print('onehot_embedding_path existing ,loading...')
        g_train.ndata['feat'] = torch.load(config_store.dic_config['onehot_embedding_path'], map_location=torch.device(device)).to(device)
        g_test.ndata['feat'] = g_train.ndata['feat']
        print('onehot_embedding_path loaded')

else:
    print('illegal embed_init, using biobert_calculated...')
    if not os.path.exists(config_store.dic_config['biobert_embedding_path']):
        print('biobert_embedding not existing ,creating from scratch...')
        g_train.ndata['feat'] = feature_init(embed_init, g_train.number_of_nodes())
        g_test.ndata['feat'] = feature_init(embed_init, g_test.number_of_nodes())
        torch.save(g_train.ndata['feat'], config_store.dic_config['biobert_embedding_path'])
        print('biobert_tokenembedding saved')
    else:
        print('biobert_embedding existing ,loading...')
        g_train.ndata['feat'] = torch.load(config_store.dic_config['biobert_embedding_path'], map_location=torch.device(device)).to(device)
        g_test.ndata['feat'] = g_train.ndata['feat']
        print('biobert_embedding loaded')


# change at 2025/6/10 end

train_dataloader = dgl.dataloading.DataLoader(
    g_train,                                  # The graph
    torch.arange(g_train.number_of_edges()).to(device),  # The edges to iterate over
    sampler,                                # The neighbor sampler
    device=device,                          # Put the MFGs on CPU or GPU
    batch_size=batch_size,    # Batch size
    shuffle=True,       # Whether to shuffle the nodes for every epoch
    drop_last=False,    # Whether to drop the last incomplete batch
    num_workers=0       # Number of sampler processes
)
# u_embed = np.arange(len(g_train.nodes()))
# onehot_sp = sp.coo_matrix(((np.ones(len(g_train.nodes()))),(u_embed, u_embed)), shape=[len(g_train.nodes()), len(g_train.nodes())])
# g_train.ndata['feat'] = onehot_sp

# -------------#3-Graph model construction-----------------------------------
model_bio = AutoModel.from_pretrained(pretrained_model_name_or_path=pretrained_model)
model_bio = model_bio.to(device)
def nan_filter(embedding_tuple):
    output_embed = None
    for i in range(0, len(embedding_tuple)):
        mean_get = embedding_tuple[i].reshape(1, -1)
        if torch.any(torch.isnan(mean_get)):
            pass
        else:
            if output_embed == None:
                output_embed = mean_get
            else:
                output_embed = torch.concat([output_embed, mean_get], dim=0)
            # output_embed.append(mean_get)

    return output_embed


def type_embedding_translator(type_get_list, data_store = {}):
    type_embedding_outputs = None
    for i in type_get_list:
        if isinstance(i, list):
            concat_embed = None
            for j in i:
                if str(j) in data_store.keys():
                    output_embed = data_store[str(j)]
                    if concat_embed == None:
                        concat_embed = output_embed
                    concat_embed = torch.concat([concat_embed, output_embed], dim=0)
                    continue
                if j[2] == 'root':
                    # com_embed = type_embedding(onehot_func(pos_graph.ndata['hierarchy_type'].shape[1], j[2]), torch.concat((type_src,type_des)), torch.concat((id_src, id_des)), outputs)
                    src_embed = type_embedding(onehot_func(pos_graph.ndata['hierarchy_type'].shape[1], j[0]), type_src,
                                            id_src, outputs, data_store = data_store_type)
                    des_embed = type_embedding(onehot_func(pos_graph.ndata['hierarchy_type'].shape[1], j[1]), type_des,
                                            id_des, outputs, data_store = data_store_type)
                    output_embed = nan_filter((src_embed, des_embed)).mean(dim=0)
                else:
                    src_embed = type_embedding(onehot_func(pos_graph.ndata['hierarchy_type'].shape[1], j[0]), type_src,
                                            id_src, outputs, data_store = data_store_type)
                    des_embed = type_embedding(onehot_func(pos_graph.ndata['hierarchy_type'].shape[1], j[1]), type_des,
                                            id_des, outputs, data_store = data_store_type)
                    com_embed = type_embedding(onehot_func(pos_graph.ndata['hierarchy_type'].shape[1], j[2]),
                                            torch.concat((type_src, type_des)), torch.concat((id_src, id_des)),
                                            outputs, data_store = data_store_type)
                    output_embed = nan_filter((src_embed, des_embed, com_embed)).mean(dim=0)
                output_embed = output_embed.reshape(1, -1)
                data_store[str(j)] = output_embed
                if concat_embed == None:
                    concat_embed = output_embed
                concat_embed = torch.concat([concat_embed, output_embed], dim=0)
            output_embed = concat_embed.mean(dim=0)
        else:
            if i[2] == 'root':
                src_embed = type_embedding(onehot_func(pos_graph.ndata['hierarchy_type'].shape[1], i[0]), type_src,
                                        id_src, outputs, data_store = data_store_type)
                des_embed = type_embedding(onehot_func(pos_graph.ndata['hierarchy_type'].shape[1], i[1]), type_des,
                                        id_des, outputs, data_store = data_store_type)
                output_embed = nan_filter((src_embed, des_embed)).mean(dim=0)
            else:
                src_embed = type_embedding(onehot_func(pos_graph.ndata['hierarchy_type'].shape[1], i[0]), type_src,  id_src, outputs, data_store = data_store_type)
                des_embed = type_embedding(onehot_func(pos_graph.ndata['hierarchy_type'].shape[1], i[1]), type_des,
                                        id_des, outputs, data_store = data_store_type)
                com_embed = type_embedding(onehot_func(pos_graph.ndata['hierarchy_type'].shape[1], i[2]),
                                        torch.concat((type_src, type_des)), torch.concat((id_src, id_des)), outputs, data_store = data_store_type)
                output_embed = nan_filter((src_embed, des_embed, com_embed)).mean(dim=0)
        output_embed = output_embed.reshape(1, -1)
        if type_embedding_outputs == None:
            type_embedding_outputs = output_embed
        else:
            type_embedding_outputs = torch.concat([type_embedding_outputs, output_embed], dim=0)
    return type_embedding_outputs.to(device)
def find_type(embedtype, target_tensor, id_tensor,data_store={})->torch.tensor:
    '''
    get the tensor of embed type in the target_tensor
    '''
    # if str(embedtype) not in data_store.keys():
    #     index_target_tensor = torch.where((torch.mul(target_tensor, embedtype).sum(dim=-1).reshape(-1,1) == 1).all(dim=1))[0].to(device)
    #     data_store[str(embedtype)] = id_tensor[index_target_tensor]
    #     return data_store[str(embedtype)]
    # else:
    #     return data_store[str(embedtype)]
    index_target_tensor = torch.where((torch.mul(target_tensor, embedtype).sum(dim=-1).reshape(-1,1) == 1).all(dim=1))[0]
    return id_tensor[index_target_tensor]

def type_embedding(embedtype, target_tensor, id_tensor, outputs, double_inputs=False, data_store={}):
    '''
    get the mean embedding of target type in the batch
    '''
    if not double_inputs:
        index_embed = find_type(embedtype, target_tensor, id_tensor, data_store)
        select_embed = outputs[index_embed]

        return select_embed.mean(dim=0).to(device)
    else:
        index_embedsrc = find_type(embedtype, target_tensor[0], id_tensor[0], data_store)
        index_embeddst = find_type(embedtype, target_tensor[1], id_tensor[1], data_store)
        
        select_embedsrc = outputs[index_embedsrc]
        select_embeddst = outputs[index_embeddst]

        select_embed = torch.concat((select_embedsrc, select_embeddst), dim=0)
        return select_embed.mean(dim=0).to(device)
def onehot_ind_translator(ind):
    return type_name_list[ind]
def common_ancestor_sto(type_all, tree, nodea_l, nodeb_l):
    '''
    nodea_l: [[0,1,0],[1,1,0],...]
    '''
    ancestor = []
    # print(nodea_l[1], nodeb_l[1])
    for i in range(0, len(nodea_l)):
        nodea = nodea_l[i]
        nodeb = nodeb_l[i]
        # print(nodea, nodeb)
        ind_1_a = torch.where(nodea == 1)[0].tolist()
        ind_1_b = torch.where(nodeb == 1)[0].tolist()
        if len(ind_1_a) == 1 and len(ind_1_b) == 1:
            nodea = onehot_ind_translator(ind_1_a[0])
            nodeb = onehot_ind_translator(ind_1_b[0])
            nodea = tree.get_node(nodea)
            nodeb = tree.get_node(nodeb)
            if nodea.identifier == 'root':
                ancestor.append((nodea.identifier, nodeb.identifier, nodea.identifier))
                continue
            if nodeb.identifier == 'root':
                ancestor.append((nodea.identifier, nodeb.identifier, nodeb.identifier))
                continue
            if tree.is_ancestor(nodea.identifier, nodeb.identifier):
                ancestor.append((nodea.identifier, nodeb.identifier, nodea.identifier))
                continue
            if tree.is_ancestor(nodeb.identifier, nodea.identifier):
                ancestor.append((nodea.identifier, nodeb.identifier, nodeb.identifier))
                continue
            if nodea == nodeb:
                ancestor.append((nodea.identifier, nodeb.identifier, nodea.identifier))
                continue
            else:
                parent = tree.parent(nodea.identifier)
                while parent != None:
                    if tree.is_ancestor(parent.identifier, nodeb.identifier) and parent.identifier in type_all:
                        ancestor.append((nodea.identifier, nodeb.identifier, parent.identifier))
                        break
                    parent = tree.parent(parent.identifier)
        else:
            # print(ind_1_a, ind_1_b)
            ancestor_inner = []
            for ai in ind_1_a:
                for bj in ind_1_b:
                    nodea = onehot_ind_translator(ai)
                    nodeb = onehot_ind_translator(bj)
                    nodea = tree.get_node(nodea)
                    nodeb = tree.get_node(nodeb)
                    if nodea.identifier == 'root':
                        ancestor_inner.append((nodea.identifier, nodeb.identifier, nodea.identifier))
                        continue
                    if nodeb.identifier == 'root':
                        ancestor_inner.append((nodea.identifier, nodeb.identifier, nodeb.identifier))
                        continue
                    if tree.is_ancestor(nodea.identifier, nodeb.identifier):
                        ancestor_inner.append((nodea.identifier, nodeb.identifier, nodea.identifier))
                        continue
                    if tree.is_ancestor(nodeb.identifier, nodea.identifier):
                        ancestor_inner.append((nodea.identifier, nodeb.identifier, nodeb.identifier))
                        continue
                    if nodea == nodeb:
                        ancestor_inner.append((nodea.identifier, nodeb.identifier, nodea.identifier))
                        continue
                    else:
                        parent = tree.parent(nodea.identifier)
                        while parent != None:
                            if tree.is_ancestor(parent.identifier, nodeb.identifier) and parent.identifier in type_all:
                                ancestor_inner.append((nodea.identifier, nodeb.identifier, parent.identifier))
                                break
                            parent = tree.parent(parent.identifier)
            ancestor.append(ancestor_inner)
    return ancestor

def onehot_func(length, type_name=None, index=None):
    if index==None:
        x = [0] * length
        x[type_name_list.index(type_name)]=1
        return torch.tensor(x).reshape([1, -1]).to(device)
    elif type_name==None:
        x = [0] * length
        x[index]=1
        return torch.tensor(x).reshape([1, -1]).to(device)

class GraphSAGE(nn.Module):
    def __init__(self, in_feats, h_feats, aggregation):
        super(GraphSAGE, self).__init__()
        self.conv1 = SAGEConv(in_feats, h_feats, aggregation)
        self.conv2 = SAGEConv(h_feats, 32, aggregation)
        self.h_feats = h_feats

    def forward(self, mfgs, x):
        h_dst = x[:mfgs[0].num_dst_nodes()]
        size_b = torch.norm(h_dst, p=2, dim=1)
        # print('sizeb shape:', size_b.mean())
        # h_dst = h_dst/torch.norm(h_dst, p=2, dim=1).reshape(-1,1)
        # print('norm_bert:', h_dst)
        # h_dst = torch.layer_norm(h_dst, (768,))
        h = self.conv1(mfgs[0], (x, h_dst))
        # print('conv1_o:', h)
        h = h/torch.norm(h, p=2, dim=1).reshape(-1,1)
        # print('conv1_norm:', h)
        h *= size_b.mean()
        # print('conv1_sizeb:', h)
        h = F.leaky_relu(h)
        # h = torch.sigmoid(h)
        # h = torch.layer_norm(h, (128,))
        h_dst = h[:mfgs[1].num_dst_nodes()]
        size_b = torch.norm(h_dst, p=2, dim=1)
        h = self.conv2(mfgs[1], (h, h_dst))
        h = h/torch.norm(h, p=2, dim=1).reshape(-1,1)
        h *= size_b.mean()
        h = F.leaky_relu(h)
        return h

class MarginCalc(nn.Module):
    def __init__(self, input_size):
        super(MarginCalc, self).__init__()
        self.encoder = nn.TransformerEncoderLayer(d_model=input_size*3, nhead=8)
        self.mlp = nn.Linear(input_size*3,1)
    def forward(self, src, tar, emb):
        inputs = torch.concat([src, tar, emb], dim=-1)
        h = self.encoder(inputs)
        h = F.leaky_relu(h)
        h = self.mlp(h)
        h = torch.sigmoid(h)
        return h

# inference 有问题
def inference(model, graph, node_features):
    with torch.no_grad():
        nodes = torch.arange(graph.number_of_nodes())

        sampler = dgl.dataloading.NeighborSampler(neighbour_ratio)
        train_dataloader = dgl.dataloading.DataLoader(
            graph, torch.arange(graph.number_of_nodes()).to(device), sampler,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=0,
            device=device)

        result = []
        for input_nodes, output_nodes, mfgs in train_dataloader:
            # feature copy from CPU to GPU takes place here
            inputs = mfgs[0].srcdata['feat']
            # bert_rep = model_bio(inputs)
            bert_rep = inputs
            outputs = model(mfgs, bert_rep)
            result.append(outputs)

        return torch.cat(result)

def inference_bert(model, graph, node_features):
    return graph.ndata['feat']

train_pos_g = dgl.graph((train_pos_u, train_pos_v), num_nodes=g_train.number_of_nodes()).to(device)
# train_neg_g = dgl.graph((train_neg_u, train_neg_v), num_nodes=g_train.number_of_nodes()).to(device)

test_pos_g = dgl.graph((test_pos_u, test_pos_v), num_nodes=g_test.number_of_nodes()).to(device)

# tmp 2025/6/18
max_node_id = g_test.number_of_nodes()  # 1795
valid_mask = (test_neg_u < max_node_id) & (test_neg_v < max_node_id)
test_neg_u = test_neg_u[valid_mask]
test_neg_v = test_neg_v[valid_mask]
test_neg_g = dgl.graph((test_neg_u, test_neg_v), num_nodes=max_node_id).to(device)

#test_neg_g = dgl.graph((test_neg_u, test_neg_v), num_nodes=g_test.number_of_nodes()).to(device)

import dgl.function as fn


class DotPredictor(nn.Module):
    def forward(self, g, h):
        with g.local_scope():
            g.ndata['h'] = h
            # Compute a new edge feature named 'score' by a dot-product between the
            # source node feature 'h' and destination node feature 'h'.
            g.apply_edges(fn.u_dot_v('h', 'h', 'score'))
            # u_dot_v returns a 1-element vector for each edge so you need to squeeze it.
            return g.edata['score'][:, 0]
class MLPPredictor(nn.Module):
    def __init__(self, h_feats):
        super().__init__()
        self.W1 = nn.Linear(h_feats * 2, h_feats)
        self.W2 = nn.Linear(h_feats, 1)

    def apply_edges(self, edges):
        h = torch.cat([edges.src['h'], edges.dst['h']], 1)
        return {'score': self.W2(F.relu(self.W1(h))).squeeze(1)}

    def forward(self, g, h):
        with g.local_scope():
            g.ndata['h'] = h
            g.apply_edges(self.apply_edges)
            return g.edata['score']
# model = GraphSAGE(g_train.ndata['feat'].shape[1], graph_hidden_size, aggregation=aggregation_method)
model = GraphSAGE(768, graph_hidden_size, aggregation=aggregation_method)
model.to(device)
pred = MLPPredictor(32).to(device)
# pred = DotPredictor()
mar_model = MarginCalc(32)
mar_model.to(device)
mar_model_outer = MarginCalc(32)
mar_model_outer.to(device)
# def compute_loss(pos_score, neg_score):
#     scores = torch.cat([pos_score, neg_score])
#     labels = torch.cat([torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])]).to(device)
#     return F.binary_cross_entropy_with_logits(scores, labels)

# def compute_auc(pos_score, neg_score):
#     scores = torch.cat([pos_score, neg_score]).numpy()
#     labels = torch.cat(
#         [torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])]).numpy()
#     return roc_auc_score(labels, scores)

# ----------- 3. set up loss and optimizer -------------- #
# in this case, loss will in training loop
optimizer_SGD = torch.optim.SGD(itertools.chain(model.parameters(), pred.parameters(),  mar_model.parameters(), mar_model_outer.parameters()), lr=learning_rate*0.05)
optimizer_ADAM = torch.optim.Adam(itertools.chain(model.parameters(), pred.parameters(), mar_model.parameters(), mar_model_outer.parameters()), lr=learning_rate)

# ----------- 4. training -------------------------------- #
print("start training")
best_accuracy = 0
best_model_path = 'model.pt'

if not os.path.exists(config_store.dic_config['model_pt_path']):
    queue_loss = []
    loss_out = 0
    for epoch in range(epoches):
        with tqdm(train_dataloader) as tq:
            for step, (input_nodes, pos_graph, neg_graph, mfgs) in enumerate(tq):
                # feature copy from CPU to GPU takes place here
                # print(mfgs[0].srcdata)
                # a = time.time()
                data_store_type = {}
                time_s = time.time()
                inputs = mfgs[0].srcdata['feat']
                bert_rep = inputs
                outputs = model(mfgs, bert_rep)

                pos_score = pred(pos_graph, outputs)
# # cancel start
                id_src = pos_graph.edges()[0]
                id_des = pos_graph.edges()[1]
                if not (ablation_Marg == 1 or ablation_Ance == 1):
                    type_src = pos_graph.ndata['hierarchy_type'][id_src]
                    type_des = pos_graph.ndata['hierarchy_type'][id_des]
                matmul = type_src.mul(type_des)
                ind_mat = matmul.nonzero()

                # get edges whose nodes in same type
                src_ind, des_ind = id_src[ind_mat[:, 0]], id_des[ind_mat[:, 0]]
                type_ind = ind_mat[:, 1]
                marsrc = torch.unsqueeze(outputs[src_ind], 0)
                mardes = torch.unsqueeze(outputs[des_ind], 0)
                ancestor_embed = torch.zeros(marsrc.shape).to(device)

                times_record = torch.zeros(marsrc.shape).to(device)
                length_times = marsrc.shape[-1]

                # time_0 = time.time()
                # print('0 time:{}', time_0-time_s)
                for type_item_id in range(0, len(type_name_list)):
                    type_item = type_name_list[type_item_id]
                    if not (ablation_Marg == 1 or ablation_Ance == 1):
                        src_mean = type_embedding(onehot_func(pos_graph.ndata['hierarchy_type'].shape[1], type_item), type_src, id_src, outputs, data_store = data_store_type)
                        des_mean = type_embedding(onehot_func(pos_graph.ndata['hierarchy_type'].shape[1], type_item), type_des, id_des, outputs, data_store = data_store_type)
                    outputs_embed = None
                    if torch.any(torch.isnan(src_mean)) and not torch.any(torch.isnan(des_mean)):
                        outputs_embed = des_mean
                    elif torch.any(torch.isnan(des_mean)) and not torch.any(torch.isnan(src_mean)):
                        outputs_embed = src_mean
                    elif torch.any(torch.isnan(des_mean)) and torch.any(torch.isnan(src_mean)):
                        continue
                    elif not torch.any(torch.isnan(des_mean)) and not torch.any(torch.isnan(src_mean)):
                        if not (ablation_Marg == 1 or ablation_Ance == 1):
                            outputs_embed = type_embedding(onehot_func(pos_graph.ndata['hierarchy_type'].shape[1], type_item), (type_src, type_des), (id_src, id_des), outputs, double_inputs=True, data_store = data_store_type)
                    ancestor_embed[0, torch.where(type_ind==type_item_id)[0], :] += outputs_embed
                    times_record[0, torch.where(type_ind==type_item_id)[0], :] += torch.ones(length_times).to(device)
                ancestor_embed = ancestor_embed/times_record
                # print(ancestor_embed)

                margin_learned = mar_model(marsrc, mardes, ancestor_embed)
                # time_1 = time.time()
                # print('1 time:{}', time_1-time_0)
                loss_margin = nn.MarginRankingLoss(margin=0.01*float(margin_learned.max()))
                if not(ablation_Marg == 1):

                    sum_hierarchy = torch.tensor(float(0)).to(device)
                    sum_hierarchy += loss_margin(outputs[src_ind], outputs[des_ind], torch.ones(outputs[src_ind].shape).to(device))
                    # get edges whose nodes in same type --end
                if not (ablation_Marg == 1 or ablation_Ance == 1):
                    index_zero = torch.where((matmul == torch.tensor([0] * matmul.shape[1]).to(device)).all(dim=1))[0]
                    src_ind, des_ind = id_src[index_zero], id_des[index_zero]
                    marsrc = torch.unsqueeze(outputs[src_ind], 0)
                    mardes = torch.unsqueeze(outputs[des_ind], 0)

                    if not (ablation_Marg == 1 or ablation_Ance == 1):
                        src_ind_type = pos_graph.ndata['hierarchy_type'][src_ind]
                        des_ind_type = pos_graph.ndata['hierarchy_type'][des_ind]

                    type_get = common_ancestor_sto(type_name_list + ['root'], tree, src_ind_type, des_ind_type)
                    # time_2 = time.time()
                    # print('2 time:{}', time_2-time_1)
                    ancestor_embedding_outer = torch.unsqueeze(type_embedding_translator(type_get), 0)
                    margin_learned_outer = mar_model(marsrc, mardes, ancestor_embedding_outer)
                    loss_margin_outer = nn.MarginRankingLoss(margin=float(margin_learned_outer.max()))
                    sum_hierarchy_outer = torch.tensor(float(0)).to(device)
                    sum_hierarchy_outer += loss_margin(outputs[src_ind], outputs[des_ind], torch.ones(outputs[src_ind].shape).to(device))
                    # get edges whose nodes in tree --end
                    # time_3 = time.time()
                    # print('3 time:{}', time_3-time_2)

                # # cancel end
                neg_score = pred(neg_graph, outputs)
                score = torch.cat([pos_score, neg_score])
                label = torch.cat([torch.ones_like(pos_score), torch.zeros_like(neg_score)])
                loss = F.binary_cross_entropy_with_logits(score, label)
                if not (ablation_Marg ==1 ):
                    loss += 0.3*sum_hierarchy
                if not (ablation_Marg ==1 or ablation_Ance ==1):
                    loss += 0.1*sum_hierarchy_outer

                if int(epoch) < 30:
                    optimizer_ADAM.zero_grad()
                    loss.backward()
                    optimizer_ADAM.step()
                else:
                    optimizer_SGD.zero_grad()
                    loss.backward()
                    optimizer_SGD.step()
                loss_out = loss.item()
                tq.set_postfix({'loss': '%.04f' % loss.item(), 'epoch': '%d' % epoch}, refresh=False)
                writer.add_scalar('loss', loss.item(), epoch)
            with torch.no_grad():
                node_reprs = inference(model, g_train, g_train.ndata['feat'])
                h_pos_src = node_reprs[train_pos_u][0:10000]
                h_pos_dst = node_reprs[train_pos_v][0:10000]
                h_neg_src = node_reprs[test_neg_u][0:10000]
                h_neg_dst = node_reprs[test_neg_v][0:10000]
                score_pos = pred(train_pos_g, node_reprs)[0:10000]
                score_neg = pred(test_neg_g, node_reprs)[0:10000]
                # score_neg = torch.zeros_like(score_neg)
                test_preds = torch.cat([score_pos, score_neg]).cpu().numpy()
                test_labels = torch.cat([torch.ones_like(score_pos), torch.zeros_like(score_neg)]).cpu().numpy()
                # print(list(score_pos).sort()[-10:-1])
                # print(list(score_neg).sort()[0:10])
                auc = roc_auc_score(test_labels, test_preds)
                print('(Training) Link Prediction AUC:', auc)
            # debug output train acc end
            with torch.no_grad():
                node_reprs = inference(model, g_test, g_test.ndata['feat'])
                h_pos_src = node_reprs[test_pos_u]
                h_pos_dst = node_reprs[test_pos_v]
                h_neg_src = node_reprs[test_neg_u]
                h_neg_dst = node_reprs[test_neg_v]
                # score_pos = (h_pos_src * h_pos_dst).sum(1)
                # score_neg = (h_neg_src * h_neg_dst).sum(1)
                score_pos = pred(test_pos_g, node_reprs)
                score_neg = pred(test_neg_g, node_reprs)
                test_preds = torch.cat([score_pos, score_neg]).cpu().numpy()
                test_labels = torch.cat([torch.ones_like(score_pos), torch.zeros_like(score_neg)]).cpu().numpy()
                # print(list(score_pos).sort()[-10:-1])
                # print(list(score_neg).sort()[0:10])
                auc = roc_auc_score(test_labels, test_preds)
                print('(Validation) Link Prediction AUC:', auc)
                # print(0.01*float(margin_learned.max()))
        queue_loss.append(loss_out)
        if len(queue_loss) > 10:
            queue_loss.pop(0)
            if (abs(max(queue_loss)- min(queue_loss)) <= 1e-2):
                print('convergence, over.')
                torch.save(model, config_store.dic_config['model_pt_path'])
                torch.save(pred, config_store.dic_config['pred_pt_path'])
                break

else:
    print('model.pt at {} loaded'.format(config_store.dic_config['model_pt_path']))
    model = torch.load(config_store.dic_config['model_pt_path'], map_location=torch.device(device))
    pred = torch.load(config_store.dic_config['pred_pt_path'], map_location=torch.device(device))
# ----------- 5. check results ------------------------ #
with torch.no_grad():
    # print(model['conv1'])
    node_reprs = inference(model, g_test, g_test.ndata['feat'])
    # print(node_reprs.shape)
    h_pos_src = node_reprs[test_pos_u]
    h_pos_dst = node_reprs[test_pos_v]
    # score_pos = (h_pos_src * h_pos_dst).sum(1)
    # score_neg = (h_neg_src * h_neg_dst).sum(1)
    score_pos = pred(test_pos_g, node_reprs)
    pos_score = score_pos
    print('sorting')
    #time_0 = time.time()
    ## ind_list = range(0, len(pos_score))
    #list1, list2, list3 = zip(*sorted(zip(pos_score, test_pos_u, test_pos_v), reverse=True))
    #time_1= time.time()
    time_0 = time.time()

    pos_score = pos_score.detach().cpu().numpy()
    test_pos_u = test_pos_u.detach().cpu().numpy()
    test_pos_v = test_pos_v.detach().cpu().numpy()

    indices = np.argsort(-pos_score)
    list1 = pos_score[indices]
    list2 = test_pos_u[indices]
    list3 = test_pos_v[indices]

    time_1 = time.time()

    print('sort_time: {}'.format(str(time_1-time_0)))
    # from scipy.special import softmax
    # list1 = softmax(list1)
    # 最后列出排名前count_topk个的结果
    count_topk = 10
    # 最后列出排名前count_topk个的结果 --end

    # 修改于 2025/6/16
    #with open('./raw_data/mg_dict_store_1.json', 'r') as fp:
    #    mg_dict = json.load(fp)
    #src_name = concept_check(cid_input, mg_dict)[0]
    #for i,j,k in tqdm(zip(list1,list2,list3)):
    #    if (inner_id, k) not in G_train.edges() or (j, inner_id) not in G_train.edges():
    #        if k == inner_id:
    #            id_tar = j
    #        elif j == inner_id:
    #            id_tar = k
    #        target_name = translator[str(int(id_tar))]
    #        id_taget = target_name
    #        target_name = concept_check(target_name, mg_dict)[0]
    #        print(id_taget, target_name, str(float(i)))
    #        count_topk = count_topk - 1
    #        if count_topk == 0:
    #            break
    print(f"\nTop-{count_topk} predictions:")
    for i in range(count_topk):
        id_src = int(list2[i])
        id_tar = int(list3[i])
        source_name = translator.get(str(id_src), f"[{id_src}]")
        target_name = translator.get(str(id_tar), f"[{id_tar}]")
        print(f"[{i + 1:02d}] {source_name} → {target_name} | Score: {list1[i]:.4f}")
    # 修改毕 2025/6/16



with torch.no_grad():
    # print(model['conv1'])
    node_reprs = inference(model, g_test, g_test.ndata['feat'])
    # print(node_reprs.shape)
    h_pos_src = node_reprs[test_pos_u]
    h_pos_dst = node_reprs[test_pos_v]
    h_neg_src = node_reprs[test_neg_u]
    h_neg_dst = node_reprs[test_neg_v]
    # score_pos = (h_pos_src * h_pos_dst).sum(1)
    # score_neg = (h_neg_src * h_neg_dst).sum(1)
    score_pos = pred(test_pos_g, node_reprs)
    score_neg = pred(test_neg_g, node_reprs)
    test_preds = torch.cat([score_pos, score_neg]).cpu().numpy()
    test_labels = torch.cat([torch.ones_like(score_pos), torch.zeros_like(score_neg)]).cpu().numpy()

    roc_auc = roc_auc_score(test_labels, test_preds)
    print('Link Prediction ROC_AUC final:', roc_auc)
    pr_auc = average_precision_score(test_labels, test_preds)
    print('Link Prediction PR_AUC final:', pr_auc)

if visual:
    print('train_pos_u, train_pos_v: ', (train_pos_u, train_pos_v))
    print('pos_score: ', score_pos)
    print('pos_score_length: ', len(score_pos))
    print('pos_score_loc: ', (test_pos_u, test_pos_v))
    print('neg_score: ', score_neg)
    print('neg_score_length: ', len(score_neg))
    print('neg_score_loc: ', (test_neg_u, test_neg_v))
import matplotlib.pyplot as plt
x = np.arange(0, len(score_pos))
x2 = np.arange(len(score_pos), len(score_pos)*2)
y1 = score_pos.tolist()
y2 = score_neg.tolist()
plt.scatter(x, y1, c='r')
n = min(len(x2), len(y2))
plt.scatter(x2[:n], y2[:n], c='b')
plt.savefig('./viewpic_norelu.png')