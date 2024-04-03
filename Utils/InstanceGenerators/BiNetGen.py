import networkx as nx
import random
import copy
import matplotlib.pyplot as plt



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

def genNetTC(nLeaves,ret):
    if nLeaves < 3 or ret < 1:
        raise Exception("input error in genNetCT")
    # making a cherry
    net = nx.DiGraph()
    net.add_edge(0, 1)
    net.add_edge(0, 2)

    currRet = 0
    currLeaves = 2

    leaves = list()
    leaves.append(1)
    leaves.append(2)
    leavesTC = list()
    leavesTC.append(1)
    leavesTC.append(2)
    iNode = 2


    for i in range(2,nLeaves + ret):
        if i == 2:
            event = 'spec'
        else:
            event = random.choices(['spec', 'lgt'], [nLeaves-currLeaves, (ret-currRet)])[0]

        if event == 'lgt' and len(leavesTC) > 0:
            currRet += 1
            m = random.choice(leavesTC)
            copyleaves = leaves.copy()
            for l in net.successors(list(net.predecessors(m))[0]):
                if l in copyleaves:
                    copyleaves.remove(l)
            l = random.choice(copyleaves)
            net.add_edge(l, m)
            net.add_edge(l, iNode + 1)
            net.add_edge(m, iNode + 2)
            leaves.remove(l)
            leaves.remove(m)
            if l in leavesTC:
                leavesTC.remove(l)
            leavesTC.remove(m)
            leaves.append(iNode + 1)
            leaves.append(iNode + 2)
            for l in net.successors(list(net.predecessors(m))[0]):
                if l in leavesTC:
                    leavesTC.remove(l)
            iNode += 2

        else:
            currLeaves +=1
            l = random.choice(leaves)
            net.add_edge(l, iNode + 1)
            net.add_edge(l, iNode + 2)
            leaves.remove(l)
            if l in leavesTC:
                leavesTC.remove(l)
            leaves.append(iNode + 1)
            leaves.append(iNode + 2)
            leavesTC.append(iNode + 1)
            leavesTC.append(iNode + 2)
            iNode += 2
    return net

def netCheckTemp(net):
    # checks if net is temporal
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

        ISet = []
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

def genRandomTreeSet(nLeaves,nTree):
   tree_set = dict()
   for iTree in range(nTree):
       tree = nx.DiGraph()
       leaves = []
       nodeNr = nLeaves
       for leaf in range(nLeaves):
           leaves.append(leaf)
           tree.add_node(leaf)
       while len(leaves) > 1:
           leafA = leaves[random.randint(0,len(leaves)-1)]
           leaves.remove(leafA)
           leafB = leaves[random.randint(0,len(leaves)-1)]
           leaves.remove(leafB)
           tree.add_edge(nodeNr,leafA)
           tree.add_edge(nodeNr,leafB)
           leaves.append(nodeNr)
           nodeNr += 1
       tree_set[iTree] = tree
   return tree_set
