import os
import json
import torch
import networkx as nx
import pandas as pd
from tqdm import tqdm

# ====== 配置路径 ======
data_dir = './raw_data/openbiolink'
train_file = os.path.join(data_dir, 'train.csv')
entities_file = os.path.join(data_dir, 'entities.csv')

save_dir = os.path.join(data_dir, 'converted')
os.makedirs(save_dir, exist_ok=True)

# ====== 1. 读取 train.csv，指定列名 ======
print(" 正在读取 train.csv...")
df_train = pd.read_csv(train_file, header=None, names=['head', 'relation', 'tail'])
triples = list(zip(df_train['head'], df_train['relation'], df_train['tail']))
print(f" 读取 {len(triples)} 条三元组")
# ====== 2. 读取实体列表，构建映射 ======
print(" 正在构建 entity_dict 和 translator...")
df_entities = pd.read_csv(entities_file, header=None, names=['entity_id', 'entity_name'])
entity_ids = df_entities['entity_id'].tolist()
entity_dict = {eid: idx for idx, eid in enumerate(entity_ids)}
translator = {str(idx): eid for eid, idx in entity_dict.items()}

# 保存 entity_dict（str->int）
with open(os.path.join(data_dir, 'entity_dictionary_0.json'), 'w') as f:
    json.dump(entity_dict, f, indent=2)

# 保存 index_translator（int->str）
with open(os.path.join(data_dir, 'index_translator.json'), 'w') as f:
    json.dump(translator, f, indent=2)

# ====== 3. 构建 NetworkX 图 ======
print(" 正在构建 G_train 图...")
G_train = nx.DiGraph()
train_pos_u = []
train_pos_v = []

skipped = 0
for h, r, t in tqdm(triples):
    if h not in entity_dict or t not in entity_dict:
        skipped += 1
        continue
    u = entity_dict[h]
    v = entity_dict[t]
    G_train.add_edge(u, v, relation=r,weight=1.0)
    train_pos_u.append(u)
    train_pos_v.append(v)

print(f" 有效边数: {len(train_pos_u)}，跳过非法边: {skipped}")
print(f" 节点总数: {G_train.number_of_nodes()}，边总数: {G_train.number_of_edges()}")

# ====== 4. 保存图为 .gpickle ======
gpickle_path = os.path.join(save_dir, 'myGraph_train.gpickle')

import pickle
# Save as .gpickle
with open(gpickle_path, 'wb') as f:
    pickle.dump(G_train, f)
#nx.write_gpickle(G_train, gpickle_path)
print(f" 已保存训练图至: {gpickle_path}")

# ====== 5. 保存边为 torch tensor ======
torch.save(torch.tensor(train_pos_u), os.path.join(save_dir, 'train_pos_u.pt'))
torch.save(torch.tensor(train_pos_v), os.path.join(save_dir, 'train_pos_v.pt'))
print("📁 已保存边集 train_pos_u.pt / train_pos_v.pt")

print("✅ OpenBioLink 训练图构建完成。你现在可以开始训练！")
