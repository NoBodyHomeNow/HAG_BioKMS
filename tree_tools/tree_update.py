from treelib import Tree
tree = Tree()

def common_ancestor(type_all, tree, nodea, nodeb):
    if nodea.identifier == 'root':
        return nodea
    if nodeb.identifier == 'root':
        return nodeb
    if tree.is_ancestor(nodea, nodeb):
        return nodea
    if tree.is_ancestor(nodeb, nodea):
        return nodeb
    if nodea == nodeb:
        return nodea
    else:
        parent = tree.parent(nodea)
        while parent != None:
            if tree.is_ancestor(parent, nodeb):
                return parent
