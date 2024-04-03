import networkx as nx
import random
import copy
import matplotlib.pyplot as plt


def hierarchy_pos(G, root=None, width=1., vert_gap=0.2, vert_loc=0, xcenter=0.5):
    '''
    From Joel's answer at https://stackoverflow.com/a/29597209/2966723.
    Licensed under Creative Commons Attribution-Share Alike

    If the graph is a tree this will return the positions to plot this in a
    hierarchical layout.

    G: the graph (must be a tree)

    root: the root node of current branch
    - if the tree is directed and this is not given,
      the root will be found and used
    - if the tree is directed and this is given, then
      the positions will be just for the descendants of this node.
    - if the tree is undirected and not given,
      then a random choice will be used.

    width: horizontal space allocated for this branch - avoids overlap with other branches

    vert_gap: gap between levels of hierarchy

    vert_loc: vertical location of root

    xcenter: horizontal location of root
    '''
    if not nx.is_tree(G):
        raise TypeError('cannot use hierarchy_pos on a graph that is not a tree')

    if root is None:
        if isinstance(G, nx.DiGraph):
            root = next(iter(nx.topological_sort(G)))  # allows back compatibility with nx version 1.11
        else:
            root = random.choice(list(G.nodes))

    def _hierarchy_pos(G, root, width=1., vert_gap=0.2, vert_loc=0, xcenter=0.5, pos=None, parent=None):
        '''
        see hierarchy_pos docstring for most arguments

        pos: a dict saying where all nodes go if they have been assigned
        parent: parent of this branch. - only affects it if non-directed

        '''

        if pos is None:
            pos = {root: (xcenter, vert_loc)}
        else:
            pos[root] = (xcenter, vert_loc)
        children = list(G.neighbors(root))
        if not isinstance(G, nx.DiGraph) and parent is not None:
            children.remove(parent)
        if len(children) != 0:
            dx = width / len(children)
            nextx = xcenter - width / 2 - dx / 2
            for child in children:
                nextx += dx
                pos = _hierarchy_pos(G, child, width=dx, vert_gap=vert_gap,
                                     vert_loc=vert_loc - vert_gap, xcenter=nextx,
                                     pos=pos, parent=root)
        return pos

    return _hierarchy_pos(G, root, width, vert_gap, vert_loc, xcenter)

def showTree(tree,labels=True):
    subax1 = plt.plot()
    root = [u for u in tree.nodes() if tree.in_degree(u) == 0]
    root = root[0]
    pos = hierarchy_pos(tree, root)
    nx.draw(tree, pos=pos, with_labels=True, font_weight='bold')
    plt.show()
def showTreeSet(treeSet,labels=True):
    for i in treeSet:
        plt.figure(i)
        subax1 = plt.plot()
        root = [u for u in treeSet[i].nodes() if treeSet[i].in_degree(u) == 0]
        root = root[0]
        pos = hierarchy_pos(treeSet[i], root)
        nx.draw(treeSet[i], pos=pos, with_labels=True, font_weight='bold')
    plt.show()

def showGraph(Net, labels=True):
    subax1 = plt.plot()
    nx.draw_kamada_kawai(Net, with_labels=labels, font_weight='bold')
    # nx.draw_planar(Net, with_labels=True, font_weight='bold')
    plt.show()

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
    :return: root (not in list)
    """
    return [n for n in net.nodes() if net.in_degree(n) == 0][0]


def reLabelTreeSet(treeSet):
    """
    relabel treeSet, for network
    :param treeSet:
    :return:
    """
    leaves = getLeaves(treeSet[0])
    root = getRoot(treeSet[0])
    nodeDict = list()
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
    return nodeDict

def allChildren(node,net):
    """
    return list of all children of a node
    :param node:
    :param net:
    :return:
    """
    toDo = [node]
    notchild = []
    child = set()
    while len(toDo) > 0:
        for i in toDo:
            for suc in net.successors(i):
                if net.out_degree(suc) == 0:
                    child.add(suc)
                else:
                    notchild.append(suc)
        toDo = notchild.copy()
        notchild = []
    if len(child) == 0:
        child.add(node)
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

def treeToNet(treeSet):
    """
    Creates Network from treeSet
    :param treeSet:
    :return: network, dict: inNrTrees
    """
    reLabelTreeSet(treeSet)
    net = nx.DiGraph()
    for i in treeSet:
        net.add_nodes_from(treeSet[i].nodes())
        net.add_edges_from(treeSet[i].edges())
    return net

def getInNrTrees(treeSet):
    """
    Get inNrTrees
    :param treeSet: must be labeled
    :return: inNrTrees
    """
    inNrTrees = dict()
    for tree in treeSet:
        for node in treeSet[tree]:
            if node not in inNrTrees:
                inNrTrees[node] = 1
            else:
                inNrTrees[node] += 1
    lvs = getLeaves(treeSet[0])
    for leaf in lvs:
        inNrTrees.pop(leaf)
    return inNrTrees

def ripeCherry(TSet):
    lvs = getLeaves(TSet[0])
    setall = [(l1, l2) for l1 in lvs for l2 in lvs if l1 != l2]
    set = [(l1, l2) for l1 in lvs for l2 in lvs if l1 != l2]
    for cher in setall: # doet dubbel voor (x,y) en (y,x)
        for i in TSet:
            if not isCher(TSet[i],lvs,cher):
                set.remove(cher)
                break
    return set

def isCher(tree,lSet,cher):
    node1 = cher[0]
    node2 = cher[1]
    if node1 not in lSet:
        return False
    if node2 not in lSet:
        return False
    par1 = list(tree.predecessors(node1))[0]
    par2 = list(tree.predecessors(node2))[0]
    return par1 == par2

def pickLeaf(treeSet,leaf):
    leaves = getLeaves(treeSet[0])
    for i in treeSet:
        if leaf in leaves:

            p = list(treeSet[i].predecessors(leaf))[0]
            if len(list(treeSet[i].predecessors(p))) == 0:
                # no grandparent
                treeSet[i].remove_node(leaf)
                treeSet[i].remove_node(p)
            else:
                gp = list(treeSet[i].predecessors(p))[0]
                chldren = list(treeSet[i].successors(p))
                chldren.remove(leaf)
                treeSet[i].remove_node(leaf)
                treeSet[i].remove_node(p)
                treeSet[i].add_edge(gp,chldren[0])

def pickRipe(treeSet):
    while True:
        ripe = ripeCherry(treeSet)
        if len(ripe) > 0:
            pickLeaf(treeSet, ripe[0][0])
            continue
        break
    return

def removeTrivialLeaves(treeSet):
    while True:
        pickRipe(treeSet)
        if len(getLeaves(treeSet[0])) < 3:
            return
        pickAble = canPick(treeSet)
        if len(pickAble) == 1:
            pickLeaf(treeSet, pickAble[0])
            continue
        break
    return



