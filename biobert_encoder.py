# -*- coding: utf-8 -*-
'''
@Date: 2022/5/8
@Time: 18:40
@Author: Wenzheng Song
@Email: gyswz123@gmail.com
'''
import torch
from transformers import AutoTokenizer, AutoModel
from umls_get import concept_check

def encode(cid, tokenizer, model, mg_dict):
    text = concept_check(cid, mg_dict)[0]
    text = '[CLS] '+ str(text) + ' [SEP]'
    tokenized_text = tokenizer.tokenize(text)
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    tokens_tensor = torch.tensor([indexed_tokens])
    tokens_tensor = tokens_tensor.to('cuda')
    with torch.no_grad():
        outputs = model(tokens_tensor)
        encoded_layers = outputs[0]
    x = encoded_layers[0][0].flatten()
    x = x[None, :]
    return x

def token_code(cid, tokenizer, mg_dict, device):
    text = concept_check(cid, mg_dict)[0]
    text = '[CLS] '+ str(text) + ' [SEP]'
    tokenized_text = tokenizer.tokenize(text, padding='max_length', max_length=128)
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    tokens_tensor = torch.tensor([indexed_tokens])
    tokens_tensor = tokens_tensor.to(device)

    return tokens_tensor
# pretrained_model = r'./data/pretrained_biobert'
#
# tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=pretrained_model)
# model = AutoModel.from_pretrained(pretrained_model_name_or_path=pretrained_model)
# model.to('cuda')
#
# print(encode('C0002962', tokenizer, model))