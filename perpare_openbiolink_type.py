import json
import os

entity_dict_path = './raw_data/openbiolink/entity_dictionary_0.json'
with open(entity_dict_path, 'r') as f:
    entity_dict = json.load(f)

# 自动识别类型
type_dict = {}
type_set = set()

for name, idx in entity_dict.items():
    # 假设 CHEMBL 代表 drug，HGNC 代表 gene，MONDO 代表 disease
    if name.startswith('CHEMBL'):
        t = 'drug'
    elif name.startswith('HGNC'):
        t = 'gene'
    elif name.startswith('MONDO'):
        t = 'disease'
    else:
        t = 'other'
    type_dict[str(idx)] = t
    type_set.add(t)

type_all = sorted(list(type_set))

# 保存
save_path = './raw_data/openbiolink'
with open(os.path.join(save_path, 'type_dict.json'), 'w') as f:
    json.dump(type_dict, f, indent=2)
with open(os.path.join(save_path, 'type_all.json'), 'w') as f:
    json.dump(type_all, f, indent=2)

print('✅ 已生成 type_dict.json 与 type_all.json')
