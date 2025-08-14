import pandas as pd
import networkx as nx
import json
import os
import pickle

# ==== 配置路径 ====
base_path = './raw_data/openbiolink'
save_path = './raw_data/openbiolink/converted'
os.makedirs(save_path, exist_ok=True)

train_file = os.path.join(base_path, 'train_sample.csv')
val_file = os.path.join(base_path, 'val_sample.csv')
test_file = os.path.join(base_path, 'test_sample.csv')

# ==== 1. 加载所有三元组 ====
df_train = pd.read_csv(train_file, sep='\t', header=None, names=['head', 'relation', 'tail'])
df_val = pd.read_csv(val_file, sep='\t', header=None, names=['head', 'relation', 'tail'])
df_test = pd.read_csv(test_file, sep='\t', header=None, names=['head', 'relation', 'tail'])

# ==== 2. 创建统一实体字典 ====
train_ents = pd.concat([df_train['head'], df_train['tail']]).dropna().astype(str)
val_ents = pd.concat([df_val['head'], df_val['tail']]).dropna().astype(str)
test_ents = pd.concat([df_test['head'], df_test['tail']]).dropna().astype(str)

all_entities = set(train_ents) | set(val_ents) | set(test_ents)
all_entities = sorted(list(all_entities))

entity_dict = {ent: idx for idx, ent in enumerate(all_entities)}
translator = {str(idx): ent for idx, ent in enumerate(all_entities)}

# ==== 3. 保存映射字典 ====
with open('./raw_data/openbiolink/entity_dictionary_0.json', 'w') as f:
    json.dump(entity_dict, f, indent=2)
with open('./raw_data/openbiolink/index_translator.json', 'w') as f:
    json.dump(translator, f, indent=2)

print(f"✅ 实体映射完成，共 {len(entity_dict)} 个实体")

# ==== 辅助函数：根据 df 和 entity_dict 构图 ====
def build_graph(df, name):
    G = nx.DiGraph()
    edges_added = 0
    for h, _, t in df.values:
        u = entity_dict.get(h, None)
        v = entity_dict.get(t, None)
        if u is not None and v is not None:
            G.add_edge(u, v, weight=1)
            edges_added += 1
    save_file = os.path.join(save_path, f'myGraph_{name}.gpickle')
    with open(save_file, 'wb') as f:
        pickle.dump(G, f)
    print(f"✅ 保存{name}图: {save_file}，共 {edges_added} 条边")
    return G

# ==== 4. 构建图 ====
build_graph(df_train, 'all')
build_graph(df_val, 'val')
build_graph(df_test, 'test')
