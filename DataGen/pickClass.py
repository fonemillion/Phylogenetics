import networkx as nx


def allChildren(node,net):
    """
    return list of all children of a node
    :param node:
    :param net:
    :return:
    """
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
    if len(child) == 0:
        child.append(node)
    return sorted(child)

def canPick(TreeSet):
    """
    Returns list of pickAble leaves
    :param TreeSet:
    :return:
    """
    lvs = getLeaves(TreeSet[0])
    DelSet = []
    for leaf in lvs:
        for i in TreeSet:
            parent = list(TreeSet[i].predecessors(leaf))[0]
            chldren = list(TreeSet[i].successors(parent))
            chldren.remove(leaf)
            if chldren[0] not in lvs:
                DelSet.append(leaf)
                break
    for leaf in DelSet:
        lvs.remove(leaf)
    return lvs

def getLeaves(net):
    """
    returns list of leaves
    :param net:
    :return:
    """
    return [u for u in net.nodes() if net.out_degree(u) == 0]


def getRoot(net):
    """
    returns root
    :param net: the network
    :return: root (not a list)
    """
    return [n for n in net.nodes() if net.in_degree(n) == 0][0]



def pickCher(Tset,leaf):
    """
    Pick leaf
    :param Tset:
    :param leaf:
    :return:
    """
    for i in Tset:

        p = list(Tset[i].predecessors(leaf))[0]
        if len(list(Tset[i].predecessors(p))) == 0:
            # no grandparent
            Tset[i].remove_node(leaf)
            Tset[i].remove_node(p)
        else:
            gp = list(Tset[i].predecessors(p))[0]
            chldren = list(Tset[i].successors(p))
            chldren.remove(leaf)
            Tset[i].remove_node(leaf)
            Tset[i].remove_node(p)
            Tset[i].add_edge(gp,chldren[0])



def reLabelTreeSet(treeSet):
    """
    relabel treeSet, for network
    :param treeSet:
    :return:
    """
    leaves = getLeaves(treeSet[0])
    root = getRoot(treeSet[0])
    nodeDict = []
    nodeDict.append(allChildren(root,treeSet[0]))
    for leaf in leaves:
        nodeDict.append(allChildren(leaf,treeSet[0]))

    for tree in treeSet:
        for node in treeSet[tree]:
            check = allChildren(node,treeSet[tree])
            if check not in nodeDict:
                nodeDict.append(check)


    for tree in treeSet:
        mapping = dict()
        for node in treeSet[tree]:
            mapping[node] = nodeDict.index(allChildren(node,treeSet[tree]))
        treeSet[tree] = nx.relabel_nodes(treeSet[tree], mapping, copy=True)
    return nodeDict.copy()



def weightLeaf(treeSet, leaf):
    pickAble = canPick(treeSet)
    if leaf not in pickAble:
        raise Exception("leaf not pickable")
    neighboors = set()
    for tree in treeSet:
        p = list(treeSet[tree].predecessors(leaf))[0]
        children = list(treeSet[tree].successors(p))
        children.remove(leaf)
        neighboors.add(children[0])
    return len(neighboors) - 1

