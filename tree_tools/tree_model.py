import os
import re
dict_tree = {}
dict_treeB = {}
with open('./tree_tools/SRDEF', 'r') as fp:
    for line in fp.readlines():
        if re.search('\|A([0-9+.]*)\|', line):
            start_tree = re.search('\|A([0-9+.]*)\|', line).span()[0]
            start_description = re.search('\|A([0-9+.]*)\|', line).span()[1]
            start_name = re.search('\|T.*', line).span()[0]
            name = ''
            description = ''
            tree = ''
            ind = start_description
            while line[ind] != '|':
                description += line[ind]
                ind+=1
            ind = start_name + 1
            while line[ind] != '|':
                name += line[ind]
                ind+=1
            ind = start_tree + 1
            while line[ind] != '|':
                tree += line[ind]
                ind+=1
            dict_tree[name] = tree
with open('./tree_tools/SRDEF', 'r') as fp:
    for line in fp.readlines():
        if re.search('\|B([0-9+.]*)\|', line):
            start_tree = re.search('\|B([0-9+.]*)\|', line).span()[0]
            start_description = re.search('\|B([0-9+.]*)\|', line).span()[1]
            start_name = re.search('\|T.*', line).span()[0]
            name = ''
            description = ''
            tree = ''
            ind = start_description
            while line[ind] != '|':
                description += line[ind]
                ind+=1
            ind = start_name + 1
            while line[ind] != '|':
                name += line[ind]
                ind+=1
            ind = start_tree + 1
            while line[ind] != '|':
                tree += line[ind]
                ind+=1
            dict_treeB[name] = tree
# print(dict_tree)
length_max = 0
for k, v in dict_tree.items():
    if len(v)> length_max:
        length_max = len(v)
length_maxb = 0
for k, v in dict_treeB.items():
    if len(v)> length_maxb:
        length_maxb = len(v)

# print(dict_treeB)
# print(length_maxb)
from treelib import Node, Tree
tree = Tree()
treeb = Tree()
root = ''
tree.create_node('root', 'root')
for i in range(1, length_max+1):
    for k, v in dict_tree.items():
        if len(v) > i:
            continue
        else:
            if len(v) == 1 and i == 1:
                tree.create_node(k, k, parent='root')
                root = k
            elif len(v) == 2 and i == 2:
                tree.create_node(k, k, parent=root)
            elif len(v) > 2 and i > 2 and len(v) == i:
                pattern = v[:-2]
                for k_p, v_p in dict_tree.items():
                    if re.search('^'+pattern+'$', v_p):
                        tree.create_node(k, k, parent=k_p)


rootb = ''
for i in range(1, length_maxb+1):
    for k, v in dict_treeB.items():
        if len(v) > i:
            continue
        else:
            if len(v) == 1 and i == 1:
                treeb.create_node(k, k)
                root = k
            elif len(v) == 2 and i == 2:
                treeb.create_node(k, k, parent=root)
            elif len(v) > 2 and i > 2 and len(v) == i:
                pattern = v[:-2]
                for k_p, v_p in dict_treeB.items():
                    if re.search('^'+pattern+'$', v_p):
                        treeb.create_node(k, k, parent=k_p)
# print(treeb)
tree.paste('root', treeb)

# print(tree)
# print(tree.parent('root'))

# def common_ancestor(type_all, tree, nodea, nodeb):
#     nodea = tree.get_node(nodea)
#     nodeb = tree.get_node(nodeb)
#     if nodea.identifier == 'root':
#         return nodea
#     if nodeb.identifier == 'root':
#         return nodeb
#     if tree.is_ancestor(nodea.identifier, nodeb.identifier):
#         return nodea
#     if tree.is_ancestor(nodeb.identifier, nodea.identifier):
#         return nodeb
#     if nodea == nodeb:
#         return nodea
#     else:
#         parent = tree.parent(nodea.identifier)
#         while parent != None:
#             if tree.is_ancestor(parent.identifier, nodeb.identifier) and parent.identifier in type_all:
#                 return parent
#             parent = tree.parent(parent.identifier)

# # print(common_ancestor(['root'], tree, 'T005', 'T007').identifier)
# def common_ancestor_sto(type_all, tree, nodea_l, nodeb_l):
#     ancestor = []
#     for i in range(0, len(nodea_l)):
#         nodea = nodea_l[i]
#         nodeb = nodeb_l[i]
#         nodea = tree.get_node(nodea)
#         print(i, nodea)
#         nodeb = tree.get_node(nodeb)
#         if nodea.identifier == 'root':
#             ancestor.append(nodea.identifier)
#             break
#         if nodeb.identifier == 'root':
#             ancestor.append(nodeb.identifier)
#             break
#         if tree.is_ancestor(nodea.identifier, nodeb.identifier):
#             ancestor.append(nodea.identifier)
#             break
#         if tree.is_ancestor(nodeb.identifier, nodea.identifier):
#             ancestor.append(nodeb.identifier)
#             break
#         if nodea == nodeb:
#             ancestor.append(nodea.identifier)
#             break
#         else:
#             parent = tree.parent(nodea.identifier)
#             while parent != None:
#                 if tree.is_ancestor(parent.identifier, nodeb.identifier) and parent.identifier in type_all:
#                     ancestor.append(parent.identifier)
#                     break
#                 parent = tree.parent(parent.identifier)
#     return ancestor
# print(common_ancestor_sto(['root'], tree, ['T005', 'T013'], ['T007', 'T015']))
