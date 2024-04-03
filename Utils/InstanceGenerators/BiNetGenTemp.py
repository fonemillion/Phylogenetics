import copy

import networkx as nx
import random
import matplotlib.pyplot as plt



def showGraph(Net,labels=True):
    subax1 = plt.plot()
    nx.draw_kamada_kawai(Net, with_labels=labels, font_weight='bold')
    #nx.draw_planar(Net, with_labels=True, font_weight='bold')
    plt.show()
# get last node from network
def last_node(net):
    return max(net.nodes())


# speciation event
def speciate(net, leaf):
    l = last_node(net)
    net.add_edge(leaf, l+1)
    net.add_edge(leaf, l+2)


# lateral gene transfer event
def lgt_temp(net, leaf1, leaf2):
    l = last_node(net)
    net.add_edge(leaf1, l + 1)
    net.add_edge(leaf2, l + 2)
    net.add_edge(leaf1, l + 3)
    net.add_edge(leaf2, l + 3)
    net.add_edge(l + 3, l + 4)


# return leaves from network
def leaves(net):
    return [u for u in net.nodes() if net.out_degree(u) == 0]




# return a random leave from the network
def random_leaf(net):
    return random.choice(leaves(net))


# return a random pair of leaves from the network
def random_pair(net):
    lvs = leaves(net)
    pairs = [(l1,l2) for l1 in lvs for l2 in lvs if l1 != l2]
    return random.choices(pairs)[0]


# SIMULATION

def simulation_temp(num_leafs, num_ret):

    # initialize network
    net = nx.DiGraph()
    net.add_edge(0, 1)
    net.add_edge(0, 2)
    curr_ret = 0
    num_steps = num_leafs - 2
    for i in range(num_steps):

        event = random.choices(['spec', 'lgt'], [num_steps-i - (num_ret-curr_ret), (num_ret-curr_ret)])[0]

        if event == 'spec':
            l = random.choice(leaves(net))
            speciate(net, l)
        else:
            curr_ret += 1
            pair = random_pair(net)
            lgt_temp(net, pair[0], pair[1])
    return net, curr_ret

# return reticulation nodes
def reticulations(G):
    return [v for v in G.nodes() if G.in_degree(v) == 2]


def is_ret_cherry(net, x, y):
    for p in net.pred[y]:
        if net.out_degree(p) > 1:
            for cp in net.succ[p]:
                if cp == y:
                    continue
                if net.in_degree(cp) > 1:
                    for ccp in net.succ[cp]:
                        if ccp == x:
                            return True
    return False


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

            tree_set[i] = (Tree)
        return tree_set
    else:
        if numberOfTrees > 2**ret_size:
            numberOfTrees = 2**ret_size

        ISet =[]
        TotalSet = list(range(2**ret_size))
        for TreeI in range(numberOfTrees):
            I = random.randint(0,len(TotalSet)-1)
            ISet.append(TotalSet[I])
            TotalSet.remove(TotalSet[I])

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
            tree_set[TreeI] = Tree
        return tree_set

def net_to_tree3(net, numberOfTrees = None):
    tree_set = dict()
    ret_set = list([u for u in net.nodes() if net.in_degree(u) == 2])
    #print(ret_set)
    if len(ret_set) == 0:
        return None
    ret_size = len(ret_set)
    par = dict()
    for i in ret_set:
        par[i] = list(net.predecessors(i))

    par_left = copy.deepcopy(par)

    if numberOfTrees == None:
        return None


    for TreeI in range(numberOfTrees-1):
        Tree = copy.deepcopy(net)
        for ret_node in par:
            rem_node = random.choice(par[ret_node])
            if rem_node in par_left[ret_node]:
                par_left[ret_node].remove(rem_node)
            Tree.remove_edge(rem_node,ret_node)
        node_red = list([u for u in Tree.nodes() if Tree.in_degree(u) == 1 and Tree.out_degree(u) == 1] )
        for nodek in node_red:
            parent = list(Tree.predecessors(nodek))[0]
            child = list(Tree.successors(nodek))[0]
            Tree.remove_node(nodek)
            Tree.add_edge(parent,child)
        if Tree.out_degree(0) == 1:
            Tree.remove_node(0)
        tree_set[TreeI] = Tree

    #last tree
    Tree = copy.deepcopy(net)
    for ret_node in par:
        if len(par_left[ret_node]) == 1:
            rem_node = par_left[ret_node][0]
        else:
            rem_node = random.choice(par[ret_node])
        if len(par_left[ret_node]) > 1:
            raise Exception("error")
        Tree.remove_edge(rem_node, ret_node)
    node_red = list([u for u in Tree.nodes() if Tree.in_degree(u) == 1 and Tree.out_degree(u) == 1] )
    for nodek in node_red:
        parent = list(Tree.predecessors(nodek))[0]
        child = list(Tree.successors(nodek))[0]
        Tree.remove_node(nodek)
        Tree.add_edge(parent,child)
    if Tree.out_degree(0) == 1:
        Tree.remove_node(0)
    tree_set[numberOfTrees-1] = Tree

    return tree_set

def pre_prune(treeSet, prob = 0.05):
    newTreeSet = dict()
    for i in treeSet:
        tree = copy.deepcopy(treeSet[i])
        leavesS = leaves(tree)
        for leaf in leavesS:
            if prob > random.random():
                p = list(tree.predecessors(leaf))[0]
                if len(list(tree.predecessors(p))) == 0:
                    # no grandparent
                    tree.remove_node(leaf)
                    tree.remove_node(p)
                else:
                    gp = list(tree.predecessors(p))[0]
                    chldren = list(tree.successors(p))
                    chldren.remove(leaf)
                    tree.remove_node(leaf)
                    tree.remove_node(p)
                    tree.add_edge(gp, chldren[0])
        if len(leaves(tree)) > 0:
            newTreeSet[i] = tree
    return newTreeSet

def numToBin(nr,size):
    bi = []
    for i in range(size):
        rest = nr % 2
        bi.append(int(rest))
        nr = (nr - rest)/2
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
    return net


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
    net, _ = simulation_temp(10,2)
    showGraph(net)
