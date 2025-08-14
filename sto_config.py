# -*- coding: utf-8 -*-
'''
@Date: 2022/5/9
@Time: 20:22
@Author: Wenzheng Song
@Email: gyswz123@gmail.com
'''
import os
class ConfigStore:
    def __init__(self, name):
        self.name = name
        self.dic_config = {
            'store_path': os.path.join('./exp_store', str(name)),
            'train_neg_uv_tuple_path': os.path.join('./exp_store', str(name), 'train_neg_uv_tuple.pt'),
            'test_neg_uv_tuple_path': os.path.join('./exp_store', str(name), 'test_neg_uv_tuple.pt'),
            'model_pt_path': os.path.join('./exp_store', str(name), 'model.pt'),
            'tensorboard_path': os.path.join('./exp_store', str(name), 'tensorboard'),
            'console_log_path': os.path.join('./exp_store', str(name), 'console_log.txt'),
            'paras_dict_json_path': os.path.join('./exp_store', str(name), 'paras_dict.json'),
            'biobert_embedding_path': os.path.join('./exp_store', str(name), 'biobert_embedding.pt'),
            'random_embedding_path': os.path.join('./exp_store', str(name), 'random_embedding.pt'),
            'onehot_embedding_path': os.path.join('./exp_store', str(name), 'onehot_embedding.pt'),
            'pred_pt_path': os.path.join('./exp_store', str(name), 'pred.pt')
        }
        path_exists = os.path.exists(self.dic_config['store_path'])
        if not path_exists:
            os.mkdir(self.dic_config['store_path'])