# -*- coding: utf-8 -*-
'''
@Date: 2022/4/28
@Time: 16:07
@Author: Wenzheng Song
@Email: gyswz123@gmail.com
'''
import torch
import re
import json
from py_umls_master.umls import *
import mygene
# def concept_check(cid):
#     pattern_gene = '^[0-9].*'
#     pattern_UMLS = '^C.*'
#     if re.search(pattern_gene, cid):
#         return (str(cid), '', 'T028')
#     elif re.search(pattern_UMLS, cid):
#         UMLS.check_database()
#         look = UMLSLookup()
#         code = cid
#         meaning = look.lookup_code(code, preferred=False)
#         if len(meaning) == 0:
#             return ('XXX', 'XXX', 'XXX')
#         return meaning[0]
#     else:
#         return (' ', ' ', ' ')

def concept_check(cid, mg_dict):
    pattern_gene = '^[0-9].*'
    pattern_UMLS = '^C.*'
    if re.search(pattern_gene, cid):
        if cid in mg_dict.keys():
            return (mg_dict[cid], '', 'T028')
        else:
            return ('XXX', 'XXX', 'XXX')
    elif re.search(pattern_UMLS, cid):
        UMLS.check_database()
        look = UMLSLookup()
        code = cid
        meaning = look.lookup_code(code, preferred=False)
        if len(meaning) == 0:
            return ('XXX', 'XXX', 'XXX')
        return meaning[0]
    else:
        return (' ', ' ', ' ')
# print(os.path.abspath('.'))
# mg = mygene.MyGeneInfo()
# print(mg.getgene(100101629)['name'])
# with open('./mg_dict_store_1.json', 'r') as fp:
#     mg_dict = json.load(fp)
# print(concept_check('C0475830', mg_dict))