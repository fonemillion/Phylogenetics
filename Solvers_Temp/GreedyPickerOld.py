import copy
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from Utils.Solvers.TreeAn import reLabelTreeSet

def showGraph(Net,labels=True):
    """
    Show network
    :param Net: The network
    :param labels: show Labels. T/F
    :return: None
    """
    subax1 = plt.plot()
    nx.draw_kamada_kawai(Net, with_labels=labels, font_weight='bold')
    #nx.draw_planar(Net, with_labels=True, font_weight='bold')
    plt.show()
def getLeaves(net):
    """
    Get leaves from a network
    :param net: the network
    :return: list of leaves
    """
    return [n for n in net.nodes() if net.out_degree(n) == 0]
def getRoot(net):
    """
    Get leaves from a network
    :param net: the network
    :return: list of roots
    """
    return [n for n in net.nodes() if net.in_degree(n) == 0]

def treeToNet(treeSet):
    """
    Creates Network from treeSet
    :param treeSet:
    :return: network, dict: inNrTrees
    """
    treeSet = copy.deepcopy(treeSet)
    reLabelTreeSet(treeSet)
    net = copy.deepcopy(treeSet[0])
    # Define a dictionary of node labels
    labels = {}
    for node in net:
        labels[node] = allChildren(node,net)
    #nx.set_node_attributes(net, labels, 'children')

    for i in range(1,len(treeSet)):
        treeSet[i]
        labels2 = {}
        for node in treeSet[i]:
            labels2[node] = allChildren(node,treeSet[i])
        #nx.set_node_attributes(treeSet[i], labels2, 'children')
        #print(nx.get_node_attributes(tree2, 'children'))

        key_list = list(labels.keys())
        val_list = list(labels.values())
        #position = val_list.index(100)

        nrNode = len(net)
        mapping = {}
        for node in treeSet[i]:
            if labels2[node] in labels.values():
                position = val_list.index(labels2[node])
                newNode = key_list[position]
            else:
                newNode = nrNode
                nrNode += 1
            mapping[node] = newNode
        treeSet[i] = nx.relabel_nodes(treeSet[i], mapping)
        labels2 = {}
        for node in treeSet[i]:
            labels2[node] = allChildren(node, treeSet[i])
        labels.update(labels2)
        #print(mapping)
        net.add_nodes_from(treeSet[i].nodes())
        net.add_edges_from(treeSet[i].edges())

        #print(nx.get_node_attributes(treeSet[i], 'children'))
    inNrTrees = dict()
    for node in net:
        inNrTrees[node] = 0
    for i in treeSet:
        #showGraphLabel(treeSet[i])
        for node in treeSet[i]:
            inNrTrees[node] += 1
    #nx.set_node_attributes(net, inNrTrees, 'inNrTrees')
    #showGraph(net)
    rNet = nx.DiGraph()
    rNet.add_nodes_from(net.nodes())
    rNet.add_edges_from(net.edges())
    lvs = [n for n in rNet.nodes() if rNet.out_degree(n) == 0]
    for node in lvs:
        inNrTrees.pop(node)
    del net, treeSet, lvs
    return rNet, inNrTrees
def allChildren(node,net):
    """
    Returns set of leaves bellow node. (also works for network)
    :param node: node in net
    :param net: networkx
    :return: list() of leaves
    """
    toDo = [node]
    notleaf = []
    leaf = set()
    while len(toDo) > 0:
        for i in toDo:
            for suc in net.successors(i):
                if net.out_degree(suc) == 0:
                    leaf.add(suc)
                else:
                    notleaf.append(suc)
        toDo = notleaf.copy()
        notleaf = []
    if len(leaf) == 0:
        leaf.add(node)
    return sorted(leaf)


class Case:
    """
    An object to check for the solver
    """
    def __init__(self,netIn, inTreeIn, leavesIn = None,childKIn = None,childDIn = None,cherriesIn = None,pickAbleIn = None, wIn = 0,CIn = None,sIn = None):

        self.net = netIn
        self.inTree = inTreeIn

        if leavesIn == None:
            self.leaves = getLeaves(self.net)
        else:
            self.leaves = leavesIn

        if childKIn == None:
            self.childK = []
            self.childD = []
            for node in self.net:
                self.childK.append(node)
                self.childD.append(allChildren(node, self.net))
            #showGraph(self.net)
            #print(self.childK)
            #print(self.childD)
        else:
            self.childK = childKIn
            self.childD = childDIn

        if cherriesIn == None:
            self.cherries = [self.childK[i] for i in range(len(self.childK)) if len(self.childD[i]) == 2]
        else:
            self.cherries = cherriesIn
        #print("nr of cherries",self.cherries)

        if pickAbleIn == None:
            self.pickAble = []
            for leaf in self.leaves:
                if all([p in self.cherries for p in self.net.predecessors(leaf)]):
                    self.pickAble.append(leaf)
        else:
            self.pickAble = pickAbleIn



        self.w = wIn
        if CIn == None:
            self.C = []
        else:
            self.C = CIn.copy()
        if sIn == None:
            self.s = []
        else:
            self.s = sIn


    def solveCluster(self,k):



        cluster = []
        root = getRoot(self.net)[0]

        for key in self.inTree:
            if key == root:
                continue
            if self.inTree[key] == self.inTree[root]:
                cluster.append(key)

        if len(cluster) == 0:
            return False

        # find smallest Cluster
        biggistNode = cluster[0]
        for node in cluster:
            if len(self.childD[self.childK.index(node)]) > len(self.childD[self.childK.index(biggistNode)]):
                biggistNode = node

        subLeaves = self.childD[self.childK.index(biggistNode)].copy()

        newNet = nx.DiGraph()
        newNet.add_nodes_from(subLeaves)



        toDo = subLeaves.copy()
        while len(toDo) > 1:
            p = list(self.net.predecessors(toDo[0]))[0]
            setI = [p]
            notleaf = []
            while len(setI) > 0:
                for i in setI:
                    for suc in self.net.successors(i):
                        if suc in toDo:
                            toDo.remove(suc)
                        else:
                            notleaf.append(suc)
                        newNet.add_edge(i, suc)
                setI = notleaf.copy()
                notleaf = []
            toDo.append(p)


        subInTree = dict()
        for node in newNet:
            if newNet.out_degree(node) != 0:
                subInTree[node] = self.inTree[node]
        # showGraph(newTree)
        sol = greedyPick(newNet,subInTree,input = "net", k = k)

        if len(sol) == 0:
            return None
        sol = sol[0][1]
        #print(sol)
        #print(self.leaves)
        #print(self.childK)
        #print(self.childD)
        #print(self.cherries)
        for i in range(len(sol) - 1):
            self.pickLeaf(sol[i])
        return True


    def pickLeaf(self,leaf):
        """
        picks a leaf
        :param leaf: to be picked
        :return: T/F if cluster could exist
        """
        #print("leaf",leaf)
        #print(self.leaves)
        #print(self.childK)
        #print(self.childD)
        #print(self.pickAble)
        #print(self.cherries)
        #print(self.net.edges.data())
        #print("after:")
        #showGraph(self.net)
        merge = []
        self.w -= 1
        parents = list(self.net.predecessors(leaf)).copy()
        for p in parents:
            self.w += 1

            toDo = list(self.net.predecessors(p)).copy()

            while len(toDo) > 0:
                node = toDo.pop(0)
                if leaf in self.childD[self.childK.index(node)]:
                    tempList = self.childD[self.childK.index(node)].copy()
                    tempList.remove(leaf)
                    if tempList in self.childD:
                        # found two the same nodes
                        merge.append([node,self.childK[self.childD.index(tempList)]])
                    del tempList
                    self.childD[self.childK.index(node)].remove(leaf)
                    for pnode in self.net.predecessors(node):
                        toDo.append(pnode)
            # remove node p
            self.cherries.remove(p)
            self.inTree.pop(p)
            self.childD.pop(self.childK.index(p))
            self.childK.remove(p)
            chldren = list(self.net.successors(p))
            chldren.remove(leaf)
            for gp in self.net.predecessors(p):
                self.net.add_edge(gp, chldren[0])
                if len(self.childD[self.childK.index(gp)]) == 2 and gp not in self.cherries:
                    self.cherries.append(gp)
            self.net.remove_node(p)
            # check if pickable
            #print([par in self.cherries for par in self.net.predecessors(chldren[0])])
            #if all([par in self.cherries for par in self.net.predecessors(chldren[0])]) and chldren[0] not in self.pickAble:
            #    self.pickAble.append(chldren[0])
            #    print("self.pickAble",self.pickAble)

        self.leaves.remove(leaf)
        self.pickAble = []
        for l in self.leaves:
            if all([par in self.cherries for par in self.net.predecessors(l)]):
                self.pickAble.append(l)
        # remove leaf
        self.net.remove_node(leaf)
        self.childK.remove(leaf)
        self.childD.remove([leaf])
        #self.pickAble.remove(leaf)
        # Constraint correction
        self.C = [cons for cons in self.C if cons[0] != leaf and cons[1] != leaf]
        # pick sequence
        self.s.append(leaf)

        merged = False
        for item in merge:
            # edges
            for p in self.net.predecessors(item[0]):
                self.net.add_edge(p, item[1])
            for child in self.net.successors(item[0]):
                self.net.add_edge(item[1], child)
            # check cluster
            #print(item,self.inTree)
            #print(self.childK,self.childD)
            self.inTree[item[1]] += self.inTree[item[0]]
            if self.inTree[item[1]] == self.inTree[getRoot(self.net)[0]]:
                merged = True
            # remove node
            self.net.remove_node(item[0])
            self.inTree.pop(item[0])
            self.childD.pop(self.childK.index(item[0]))
            self.childK.remove(item[0])
            if len(self.childD[self.childK.index(item[1])]) == 2:
                self.cherries.remove(item[0])

        #print(self.childK)
        #print(self.childD)
        #print(self.pickAble)
        #print(self.cherries)
        #print(self.net.edges())
        #print("..")
        return merged




    def ripeCherry(self):
        # NoWeightCherry
        for leaf in self.pickAble:
            if self.net.in_degree(leaf) == 1:
                return leaf
        return None

    def ripeConstrCherry(self):

        pickAble = self.pickAble

        for leaf in self.pickAble:
            allInCon = True
            for parent in self.net.predecessors(leaf):
                children = list(self.net.successors(parent))
                children.remove(leaf)
                if (leaf,children[0]) not in self.C:
                    allInCon = False
                    break
            if allInCon:
                return leaf
        return None


    def getPC(self):
        phi = np.log(2)/np.log(5)
        pi = set()
        for cons in self.C:
            pi.add(cons[0])

        return phi * len(self.C) + (1 - 2 * phi) * len(pi)

    def getPi1(self):
        pi = set()
        for i in self.C:
            pi.add(i[0])
        return pi

    def getPi2(self):
        pi = set()
        for i in self.C:
            pi.add(i[1])
        return pi

    def case1(self):
        cherSet = set()
        for node in self.cherries:
            cher = list(self.net.successors(node))
            cherSet.add((cher[0], cher[1]))
            cherSet.add((cher[1], cher[0]))

        weight = dict()
        for leaf in self.leaves:
            weight[leaf] = -2
        for cher in cherSet:
            weight[cher[0]] += 1
            weight[cher[1]] += 1

        for con in self.C:
            cherSet.remove((con[0], con[1]))
            cherSet.remove((con[1], con[0]))


        Pi1 = self.getPi1()


        for cher in cherSet:
            if weight[cher[0]] > 0 and cher[0] in Pi1:
                return cher
        return None

    def case2(self):
        cherSet = set()
        for node in self.cherries:
            cher = list(self.net.successors(node))
            cherSet.add((cher[0], cher[1]))
            cherSet.add((cher[1], cher[0]))

        weight = dict()
        for leaf in self.leaves:
            weight[leaf] = -2
        for cher in cherSet:
            weight[cher[0]] += 1
            weight[cher[1]] += 1

        for con in self.C:
            cherSet.remove((con[0],con[1]))
            cherSet.remove((con[1],con[0]))

        Pi2 = self.getPi2()

        for cher in cherSet:
            if weight[cher[0]] > 0 and cher[0] not in Pi2:
                for cher2 in cherSet:
                    if cher2[0] == cher[0] and cher2[1] != cher[1]:
                        b = cher2[1]
                        break

                return [cher[0],cher[1],b]

        return None


def Pick(case):

    while True:
        if len(case.leaves) == 2:
            return
        ripe = case.ripeCherry()
        if ripe != None:
            case.pickLeaf(ripe)
            continue
        if len(case.pickAble) == 1:
            case.pickLeaf(case.pickAble[0])
            continue
        return









def greedyPick(T ,subInTree = None, input = 'treeSet',k = None):
    """
    Does give best anwser
    :param T:
    :param subInTree:
    :param input:
    :param k:
    :return:
    """
    # T: set of trees
    check = []
    sol = []

    if input == 'treeSet':
        net, inNrTrees = treeToNet(T)
        check.append(Case(net, inNrTrees))
    else:
        check.append(Case(T, subInTree))

    if k == None:
        k = len(check[0].leaves) ** 2
    else:
        k += 1

    while len(check) > 0:
        case = check.pop(0)

        # print( case.s, case.w)
        Pick(case)



        if len(case.leaves) == 2:
            s = case.s.copy()
            s.append(case.leaves[0])
            s.append(case.leaves[1])
            if k > case.w:
                k = case.w
                sol = []
            if k == case.w:
                # print("found solution" , k, s )
                if [k,s] not in sol:
                    sol.append([k,s])
            del case
            continue

        if k <= case.w + 1:
            del case
            continue

        rc = case.solveCluster(k)
        if rc == None:
            del case
            continue
        if rc:
            check.insert(0,copy.deepcopy(case))
            del case
            continue

        #weight = 0
        #children = []
        #for cher in case.cherries:
        #    if all([child not in children for child in case.net.successors(cher)]):
        #        for child in case.net.successors(cher):
        #            children.append(child)
        #        weight += 1

        #weight = dict()
        #for leaf in case.leaves:
        #    weight[leaf] = -1
        #for cher in case.cherries:
        #    for child in case.net.successors(cher):
        #        weight[child] += 1

        #totWeight = 0
        #for leaf in weight:
        #    if weight[leaf] > 0:
        #        totWeight += weight[leaf]


        #if k - case.w - len(case.cherries) + totWeight <= 0:
        #    #print(k - case.w - len(case.cherries) + totWeight)
        #    del case
        #    #print("del")
        #    continue

        #case.cherries - totWeight
        #if k - case.w - len(case.cherries) + totWeight <= 0:
        #    #print(k - case.w - len(case.cherries) + totWeight)
        #    del case
        #    #print("del")
        #    continue

##############################

        root = getRoot(case.net)
        weight = 0
        lvs = []
        for cher in case.cherries:
            children = list(case.net.successors(cher))
            if children[0] in lvs or children[1] in lvs:
                continue
            cherSplit = False
            for node in case.net.successors(root[0]):
                subLeaves = case.childD[case.childK.index(node)]
                if children[0] in subLeaves and children[1] not in subLeaves:
                    # print(children[0],children[1],subLeaves)
                    cherSplit = True
            if cherSplit:
                weight += 1
                # print(children[0],children[1])
                lvs.append(children[0])
                lvs.append(children[1])

        if k - case.w <= weight:
            #print(k,case.w,weight,lvs)
            del case
            continue

        for leaf in case.pickAble:
            nCase = copy.deepcopy(case)

            nCase.pickLeaf(leaf)
            check.insert(0,nCase)

        del case

    return sol






