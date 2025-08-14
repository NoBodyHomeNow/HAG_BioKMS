import pandas as pd
import networkx as nx
import json
import os

val_file = './raw_data/openbiolink/val_sample.csv'
entity_dict_path = './raw_data/openbiolink/entity_dictionary_0.json'

with open(entity_dict_path, 'r') as f:
    entity_dict = json.load(f)

df_val = pd.read_csv(val_file, header=None, names=['head', 'relation', 'tail'])

G_val = nx.DiGraph()
for h, _, t in df_val.values:
    if h in entity_dict and t in entity_dict:
        u = entity_dict[h]
        v = entity_dict[t]
        G_val.add_edge(u, v,weight=1)

missing_h = 0
missing_t = 0
for h, _, t in df_val.values:
    if h not in entity_dict:
        missing_h += 1
    if t not in entity_dict:
        missing_t += 1

print(f"Missing head entities: {missing_h}")
print(f"Missing tail entities: {missing_t}")
print(f"Total rows in val: {len(df_val)}")

#nx.write_gpickle(G_val, './raw_data/openbiolink/converted/myGraph_val.gpickle')
import pickle
# Save as .gpickle
with open('./raw_data/openbiolink/converted/myGraph_val.gpickle', 'wb') as f:
    pickle.dump(G_val, f)
print(' 已保存验证图 myGraph_val.gpickle')
