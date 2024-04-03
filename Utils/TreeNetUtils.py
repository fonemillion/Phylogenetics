import networkx as nx
import random
import copy
import matplotlib.pyplot as plt
import numpy as np

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

def showTree(tree,show=True):
    subax1 = plt.plot()
    root = [u for u in tree.nodes() if tree.in_degree(u) == 0]
    root = root[0]
    pos = hierarchy_pos(tree, root)
    nx.draw(tree, pos=pos, with_labels=True, font_weight='bold')
    if show:
        plt.show()
def showTreeSet(treeSet,show=True):
    for i in treeSet:
        plt.figure(i)
        subax1 = plt.plot()
        root = [u for u in treeSet[i].nodes() if treeSet[i].in_degree(u) == 0]
        root = root[0]
        pos = hierarchy_pos(treeSet[i], root)
        nx.draw(treeSet[i], pos=pos, with_labels=True, font_weight='bold')
    if show:
        plt.show()
def showGraph(Net, show=True):
    plt.figure(-1)
    nx.draw_kamada_kawai(Net, with_labels=True,  font_weight='bold')
    if show:
        plt.show()

def showBetterTreeSet(treeSet,show=True,compre = True):
    def getPosTree(tree):
        """
        :param treeSet:
        :return: nodeDict
        Does not check for two diff keys with same value
        """

        def getPos(tree, node, nodeDict, pos):
            # check if leaf
            if tree.out_degree(node) == 0:
                if len(pos) == 0:
                    pos[node] = [0,0]
                else:
                    pos[node] = [max([pos[key][0] for key in pos])+1,0]
                return
            # solve for children
            list1 = []
            for child in tree.successors(node):
                list1.append(child)
            while len(list1) != 0:
                listsize = [len(nodeDict[item]) for item in list1]
                max_value = max(listsize)
                item = list1[listsize.index(max_value)]
                list1.remove(item)
                getPos(tree, item, nodeDict,pos)


            allChildren = nodeDict[node]
            ymax = max([pos[key][0] for key in allChildren])
            ymin = min([pos[key][0] for key in allChildren])
            pos[node] = [ (ymax + ymin)/2, (ymax - ymin)/2]
            return

        nodeDict = getLabelTree(tree)
        pos = dict()
        root = getRoot(tree)
        getPos(tree, root, nodeDict, pos)
        return pos.copy()

    def compr(node,child,tree,pos):
        lvs = getLeaves(tree)
        if child in lvs:
            distance = np.array([0.5,-0.5]) + np.array(pos[node])-np.array(pos[child])
            pos[child] = list(distance + np.array(pos[child]))
            return
        child2 = list(tree.successors(node))
        child2.remove(child)
        child2 = child2[0]
        distance = (pos[node][0]-pos[child2][0])*np.array([1,-1]) + np.array(pos[node])-np.array(pos[child])
        todo = [child]
        while len(todo)>0:
            item = todo.pop()
            pos[item] = list(distance+np.array(pos[item]))
            for children in tree.successors(item):
                todo.append(children)
    for i in treeSet:
        plt.figure(i)
        subax1 = plt.plot()
        lvs = getLeaves(treeSet[i])
        notlvs = [n for n in treeSet[i] if n not in lvs]
        #print(lvs)
        root = [u for u in treeSet[i].nodes() if treeSet[i].in_degree(u) == 0]
        root = root[0]
        pos = getPosTree(treeSet[i])
        # print(pos)
        nodeDict = getLabelTree(treeSet[i])

        if compre:
            for node in treeSet[i]:
                if node in lvs:
                    continue
                else:
                    for child in treeSet[i].successors(node):
                        if pos[child][0] > pos[node][0]:
                            compr(node,child,treeSet[i],pos)
        #print(pos)
        labellvs = dict()
        for edge in treeSet[i].edges:
            #print(edge)
            plt.plot([pos[edge[0]][0],pos[edge[1]][0]],[pos[edge[0]][1],pos[edge[1]][1]], color='black')
        for leaf in lvs:
            labellvs[leaf] = leaf
        notlvs = []
        for node in treeSet[i].nodes:
            if node not in lvs:
                notlvs.append(node)
        nx.draw_networkx_nodes(treeSet[i], pos,  nodelist=lvs,node_color = 'white')
        nx.draw_networkx_nodes(treeSet[i], pos,  nodelist=notlvs,node_color = 'black', node_size=40)
        nx.draw_networkx_labels(treeSet[i].subgraph(lvs), pos)
        #nx.draw_networkx_edges(treeSet[i], pos, width=1.0, alpha=0.5)

    if show:
        plt.show()



def getLeaves(net):
    """
    returns list of leaves
    :param net:
    :return:
    """
    return [n for n in net.nodes() if net.out_degree(n) == 0]
def getRoot(net):
    """
    returns root
    :param net: the network
    :return: root (not in list)
    """
    return [n for n in net.nodes() if net.in_degree(n) == 0][0]
def getRoots(net):
    """
    returns root
    :param net: the network
    :return: root is a list
    """
    return [n for n in net.nodes() if net.in_degree(n) == 0]


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

def getLabelTree(tree):
    """
    :param treeSet:
    :return: nodeDict
    Does not check for two diff keys with same value
    """
    def getLabel(tree,node,nodeDict):
        # check if leaf
        if tree.out_degree(node) == 0:
            nodeDict[node] = {node}
            return
        # solve for children
        allChildren = set()
        for child in tree.successors(node):
            getLabel(tree,child,nodeDict)
            allChildren |= nodeDict[child]
        nodeDict[node] = allChildren
        return


    TreeNodeDict = dict()
    root = getRoot(tree)
    getLabel(tree, root, TreeNodeDict)




    return TreeNodeDict.copy()

def getLabelTreeSet(treeSet):
    """
    :param treeSet:
    :return: nodeDict
    Does not check for two diff keys with same value
    """
    def getLabel(tree,node,nodeDict):
        # check if leaf
        if tree.out_degree(node) == 0:
            nodeDict[node] = {node}
            return
        # solve for children
        allChildren = set()
        for child in tree.successors(node):
            getLabel(tree,child,nodeDict)
            allChildren |= nodeDict[child]
        nodeDict[node] = allChildren
        return

    nodeDict = dict()

    TreeNodeDict = dict()
    for tree in treeSet:
        TreeNodeDict[tree] = dict()
        root = getRoot(treeSet[tree])
        getLabel(treeSet[tree], root, TreeNodeDict[tree])

    for Tree in TreeNodeDict:
        for i in TreeNodeDict[Tree]:
            if i not in nodeDict:
                nodeDict[i] = TreeNodeDict[Tree][i]
            else:
                if nodeDict[i] != TreeNodeDict[Tree][i]:
                    raise Exception("Labeling error")


    return nodeDict.copy()

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


