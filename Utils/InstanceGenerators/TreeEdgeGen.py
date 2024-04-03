import networkx as nx
import random
from Utils.TreeNetUtils import showTree, showGraph, showBetterTreeSet
from Utils.TreeNetUtils import getLeaves

def genTreeByEdge(nLvs):
    tree = nx.DiGraph()

    tree.add_edge(0, 1)

    extraNodes = nLvs+1
    for i in range(2,nLvs+1):
        random_edge = random.choice(list(tree.edges()))
        tree.remove_edge(*random_edge)
        node1 = random_edge[0]
        node2 = random_edge[1]
        tree.add_edge(node1, extraNodes)
        tree.add_edge(extraNodes, node2)
        tree.add_edge(extraNodes, i)
        extraNodes+=1
    root = list(tree.successors(0))[0]
    tree.remove_node(0)
    mapping = {root: 0}
    tree = nx.relabel_nodes(tree, mapping, copy=False)

    return tree

def genTreeSetByEdge(nLvs,mTree):
    treeSet = dict()
    for m in range(mTree):
        treeSet[m] = genTreeByEdge(nLvs)
    return treeSet

def genTreeByEdgeTemp(nLvs):
    tree = nx.DiGraph()

    tree.add_edge(0, 1)

    extraNodes = nLvs+1
    for i in range(2,nLvs+1):
        random_node = random.choice(range(1,i))

        random_edge = list(tree.in_edges(random_node))[0]
        tree.remove_edge(*random_edge)
        node1 = random_edge[0]
        node2 = random_edge[1]
        tree.add_edge(node1, extraNodes)
        tree.add_edge(extraNodes, node2)
        tree.add_edge(extraNodes, i)
        extraNodes+=1
    root = list(tree.successors(0))[0]
    tree.remove_node(0)
    mapping = {root: 0}
    tree = nx.relabel_nodes(tree, mapping, copy=False)

    return tree

def genTreeSetByEdgeTemp(nLvs,mTree):
    treeSet = dict()
    for m in range(mTree):
        treeSet[m] = genTreeByEdgeTemp(nLvs)
    return treeSet

