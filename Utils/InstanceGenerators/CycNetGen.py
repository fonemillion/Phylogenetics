import networkx as nx
import random
import copy
import matplotlib.pyplot as plt
from DataGen.pickClass import reLabelTreeSet
def showGraph(Net,labels=True):
    subax1 = plt.plot()
    nx.draw_kamada_kawai(Net, with_labels=labels, font_weight='bold')
    #nx.draw_planar(Net, with_labels=True, font_weight='bold')
    plt.show()


def isTC(net):
    for leaf in [u for u in net.nodes() if net.out_degree(u) != 0]:
        treechild = False
        for child in net.successors(leaf):
            if net.in_degree(child) != 2:
                treechild = True
        if not treechild:
            return False, leaf
    return True

def reticulations(G):
    return [v for v in G.nodes() if G.in_degree(v) == 2]


def getLeaves(net):
    """
    returns list of leaves
    :param net:
    :return:
    """
    return [u for u in net.nodes() if net.out_degree(u) == 0]


def genNetCyc(nLeaves,ret):
    if nLeaves < 3 or ret < 1:
        raise Exception("input error in genNetCT")

    subLvs = []
    for i in range(ret):
        subLvs.append(4)
    if sum(subLvs) + 1 > nLeaves:
        raise Exception("need more leaves")


    lvsOver = nLeaves - sum(subLvs) -1

    for i in range(lvsOver):
        subLvs[random.randrange(ret)] += 1


    subSplitlvs = []
    for i in range(ret):
        rand = random.randrange(subLvs[i] - 3)
        subSplitlvs.append([2 + rand, subLvs[i] - 2 - rand ])


    net = nx.DiGraph()
    leaves = [0]
    nrNodes = 1
    net = nx.DiGraph()
    net.add_node(0)

    for i in range(ret):
        leaves = getLeaves(net)
        node = random.choice(leaves)
        net.add_edge(node,nrNodes)
        nrNodes += 1

        for j in range(subSplitlvs[i][0]):
            net.add_edge(nrNodes-1,nrNodes)
            net.add_edge(nrNodes-1,nrNodes+1)
            nrNodes += 2

        retNode1 = nrNodes - 1

        net.add_edge(node,nrNodes)
        nrNodes += 1

        for j in range(subSplitlvs[i][1]):
            net.add_edge(nrNodes-1,nrNodes)
            net.add_edge(nrNodes-1,nrNodes+1)
            nrNodes += 2

        retNode2 = nrNodes - 1
        net.add_edge(retNode1, nrNodes)
        net.add_edge(retNode2, nrNodes)
        net.add_edge(nrNodes, nrNodes+1)
        nrNodes += 2

        redun = [n for n in net.nodes() if net.out_degree(n) == 1 and net.in_degree(n) == 1]
        for rNode in redun:
            p = list(net.predecessors(rNode))[0]
            c = list(net.successors(rNode))[0]
            net.add_edge(p,c)
            net.remove_node(rNode)

    return net

def netCheckTemp(net):
    # checks if net is temporal
    tree_set = dict()
    ret_set = list([u for u in net.nodes() if net.in_degree(u) == 2])
    if len(ret_set) == 0:
        return True

    net2 = copy.deepcopy(net)

    for child in ret_set:
        for parent in net.predecessors(child):
            net2.add_edge(child,parent)
    try:
        nx.simple_cycles(net2)
        cycl = sorted(nx.simple_cycles(net2))
        for i in cycl:
            if len(i) > 2:
                #print(i)
                return False
        return True
    except nx.exception.NetworkXNoCycle:
        return True

def net_to_tree2(net, numberOfTrees = None):
    tree_set = dict()
    ret_set = list([u for u in net.nodes() if net.in_degree(u) == 2])
    #print(ret_set)
    if len(ret_set) == 0:
        return None
    ret_size = len(ret_set)
    par = dict()
    for i in ret_set:
        par[i] = list(net.predecessors(i))

    if numberOfTrees == None:
        for i in range(2**ret_size):
            Tree = copy.deepcopy(net)
            for j in range(ret_size):
                bit = numToBin(i,ret_size)
                Tree.remove_edge(par[ret_set[j]][bit[j]], ret_set[j])
            node_red = list([u for u in Tree.nodes() if Tree.in_degree(u) == 1 and Tree.out_degree(u) == 1] )
            for nodek in node_red:
                parent = list(Tree.predecessors(nodek))[0]
                child = list(Tree.successors(nodek))[0]
                Tree.remove_node(nodek)
                Tree.add_edge(parent,child)
            if Tree.out_degree(0) == 1:
                Tree.remove_node(0)

            tree_set[i] = (Tree)
        return tree_set
    else:
        if numberOfTrees > 2**ret_size:
            numberOfTrees = 2**ret_size

        ISet =[]
        TotalSet = list(range(2**ret_size))
        for TreeI in range(numberOfTrees-1):
            I = random.randint(0,len(TotalSet)-1)
            ISet.append(TotalSet[I])
            TotalSet.remove(TotalSet[I])

        lastNr = numToBin(ISet[0], ret_size)
        subSet = [i for i in ISet if i != ISet[0]]
        for i in subSet:
            compNr = numToBin(i, ret_size)
            for j in range(ret_size):
                if compNr[j] != lastNr[j]:
                    lastNr[j] = 2

        for j in range(ret_size):
            if lastNr[j] == 0:
                lastNr[j] = 1
            elif lastNr[j] == 1:
                lastNr[j] = 0
            else:
                lastNr[j] = random.randint(0, 1)

        lastI = 0
        for j in range(ret_size):
            lastI += lastNr[j]*(2**int(j))

        ISet.append(lastI)

        #for i in range(numberOfTrees):
        #    print(numToBin(ISet[i], ret_size),ISet[i])


        for TreeI in range(numberOfTrees):
            Tree = copy.deepcopy(net)
            for j in range(ret_size):
                bit = numToBin(ISet[TreeI],ret_size)
                Tree.remove_edge(par[ret_set[j]][bit[j]], ret_set[j])
            node_red = list([u for u in Tree.nodes() if Tree.in_degree(u) == 1 and Tree.out_degree(u) == 1] )
            for nodek in node_red:
                parent = list(Tree.predecessors(nodek))[0]
                child = list(Tree.successors(nodek))[0]
                Tree.remove_node(nodek)
                Tree.add_edge(parent,child)
            if Tree.out_degree(0) == 1:
                Tree.remove_node(0)
            tree_set[TreeI] = Tree
        return tree_set

def numToBin(nr, size):
    bi = []
    for i in range(size):
        rest = nr % 2
        bi.append(int(rest))
        nr = (nr - rest) / 2
    return bi

def treeToTN(treeSet):
    net = nx.DiGraph()
    for i in treeSet:
        mapping = {}
        for node in treeSet[i]:
            if treeSet[i].out_degree(node) == 0:
                mapping[node] = node
            else:
                children = allChildren(node, treeSet[i])
                children.sort()
                mapping[node] = ' '.join([str(elem) for elem in children])
        H = nx.relabel_nodes(treeSet[i], mapping)
        #print(mapping)
        #showGraph(H)
        net.add_nodes_from(H)
        net.add_edges_from(H.edges())
        #showGraph(net)
    del H
    i = 0
    mapping = {}
    for node in net:
        mapping[node] = i
        i += 1
    H = nx.relabel_nodes(net, mapping)
    del net
    return H

def allChildren(node,net):
    set = [node]
    notchild = []
    child = []
    while len(set) > 0:
        for i in set:
            for suc in net.successors(i):
                if net.out_degree(suc) == 0:
                    child.append(suc)
                else:
                    notchild.append(suc)
        set = notchild.copy()
        notchild = []
    return child


if __name__ == '__main__':
    net = genNetCyc(20,2)
    print(len(getLeaves(net)))
    showGraph(net)
    treeSet = net_to_tree2(net,2)
    print(len(treeSet))
    showGraph(treeSet[0])
    showGraph(treeSet[1])