import copy
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from Utils.Solvers.TreeAn import *
from Utils.Solvers.GreedyPickerOld import greedyPick




class Case:
    """
    An object to check for the solver
    """
    def __init__(self,netIn, inTreeIn, leavesIn = None,childKIn = None,childDIn = None,cherriesIn = None, pickAbleIn = None, wIn = 0,CIn = None,sIn = None, pickedIn = None):

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

        if pickedIn == None:
            self.picked = set()
        else:
            self.picked == pickedIn


    def solveCluster(self,k):


        cluster = []
        root = getRoot(self.net)

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
        #print(newNet.nodes())
        if len(subLeaves) < 8:
            sol = greedyPick(newNet,subInTree,input = "net",k = k)
        else:
            sol = greedyPick3(newNet,subInTree,input = "net",k = k)

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
            if self.inTree[item[1]] == self.inTree[getRoot(self.net)]:
                merged = True
            # remove node
            self.net.remove_node(item[0])
            self.inTree.pop(item[0])
            self.childD.pop(self.childK.index(item[0]))
            self.childK.remove(item[0])
            if len(self.childD[self.childK.index(item[1])]) == 2:
                self.cherries.remove(item[0])
        self.picked.add(leaf)
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









def greedyPick3(T ,subInTree = None, input = 'treeSet',k = None):
    """
    Does give best anwser
    :param T:
    :param subInTree:
    :param input:
    :param k:
    :return:
    """
    # T: set of trees
    check = dict()
    check2 = dict()
    sol = []

    if input == 'treeSet':
        for i in range(len(getLeaves(T[0]))+1):
            check[i] = []
            check2[i] = []
        maxLayer = len(getLeaves(T[0]))
        net = treeToNet(T)
        inNrTrees = getInNrTrees(T)
        case = Case(net, inNrTrees)
    else:
        for i in range(len(getLeaves(T))+1):
            check[i] = []
            check2[i] = []
        maxLayer = len(getLeaves(T))
        case = Case(T, subInTree)

    check[0].append(case)
    check2[0].append(case.picked)

    if k == None:
        k = len(check[0][0].leaves) ** 2
    else:
        k += 1

    for layer in range(maxLayer+1):
        bestWInLayer = k
        while len(check[layer]) > 0:

            case = check[layer].pop(0)
            # print( case.s, case.w)
            Pick(case)
            if bestWInLayer < case.w:
                bestWInLayer = case.w
                uB, soluB = getUpperBound(case)
                if uB is not None:
                    if uB < k:
                        k = uB
                        sol = []
                        sol.append([k, soluB])

            # check if done
            if len(case.leaves) == 2:
                s = case.s.copy()
                s.append(case.leaves[0])
                s.append(case.leaves[1])
                if k > case.w:
                    k = case.w
                    sol = []
                if k == case.w:
                    #print("found solution" , k, s )
                    if [k,s] not in sol:
                        sol.append([k,s])
                del case
                continue

            # check if high weight
            if k <= case.w + 1:
                del case
                continue

            # check cluster
            rc = case.solveCluster(k)
            if rc == None:
                del case
                continue
            if rc:
                if len(check[len(case.picked)]) == 0:
                    uB, soluB = getUpperBound(case)
                    if uB is not None:
                        if uB < k:
                            k = uB
                            sol = []
                            sol.append([k,soluB])

                if case.picked in check2[len(case.picked)]:
                    if check[len(case.picked)][check2[len(case.picked)].index(case.picked)].w > case.w:
                        check[len(case.picked)][check2[len(case.picked)].index(case.picked)] = copy.deepcopy(case)
                else:
                    check[len(case.picked)].append(copy.deepcopy(case))
                    check2[len(case.picked)].append(copy.deepcopy(case.picked))
                del case
                continue



            root = getRoot(case.net)
            weight = 0
            lvs = []
            for cher in case.cherries:
                children = list(case.net.successors(cher))
                if children[0] in lvs or children[1] in lvs:
                    continue
                cherSplit = False
                for node in case.net.successors(root):
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

            #
            # print(case.pickAble)
            # if len(case.pickAble) == 0:
            #     print(case.s, case.w)
            for leaf in case.pickAble:
                nCase = copy.deepcopy(case)

                nCase.pickLeaf(leaf)
                if len(check[len(nCase.picked)]) == 0:
                    #print(getUpperBound(nCase))
                    uB, soluB = getUpperBound(nCase)
                    if uB is not None:
                        if uB < k:
                            k = uB
                            sol = []
                            sol.append([k,soluB])
                if nCase.picked in check2[len(nCase.picked)]:
                    if check[len(nCase.picked)][check2[len(nCase.picked)].index(nCase.picked)].w > nCase.w:
                        check[len(nCase.picked)][check2[len(nCase.picked)].index(nCase.picked)] = copy.deepcopy(nCase)
                else:
                    check[len(nCase.picked)].append(copy.deepcopy(nCase))
                    check2[len(nCase.picked)].append(copy.deepcopy(nCase.picked))
                continue

            del case
    #print(check2)
    #for i in check2:
        #print(len(check2[i]))
        #my_list_of_sets = check2[i]
        #for i, set_i in enumerate(my_list_of_sets):
        #    for set_j in my_list_of_sets[i + 1:]:
        #        if sorted(list(set_i)) == sorted(list(set_j)):
        #            print("Duplicate set found:", set_i)
    #print(check)
    return sol




def getUpperBound(case):
    net = copy.deepcopy(case.net)
    s = copy.deepcopy(case.s)
    w = case.w
    while True:
        #
        if len(getLeaves(net)) == 2:
            break
        pickAble, weigh = canPickNet(net)
        if len(pickAble) == 0:
            break
        leaf = None
        for l in pickAble:
            if weigh[l] == 0:
                leaf = l
        if leaf == None:
            leaf = random.choice(pickAble)
        #print(leaf,weigh[leaf])
        #showGraph(net)
        pickLeafNet(net,leaf)
        w += weigh[leaf]
        s.append(leaf)
    if len(getLeaves(net)) == 2:
        lvs = getLeaves(net)
        s.append(lvs[0])
        s.append(lvs[1])
        return w, s
    return None, None




def pickLeafNet(net,leaf):
    parents = copy.deepcopy(net.predecessors(leaf))

    for p in parents:
        chldren = list(net.successors(p))
        chldren.remove(leaf)
        for gp in net.predecessors(p):
            net.add_edge(gp, chldren[0])
        net.remove_node(p)
    net.remove_node(leaf)



def canPickNet(net):
    """
    return pickAble for net
    :param net:
    :return: lvs, weigh
    """
    lvs = getLeaves(net)
    weigh = dict()
    DelSet = []
    for leaf in lvs:
        neighboors = set()
        for p in net.predecessors(leaf):
            chldren = list(net.successors(p))
            if len(chldren) != 2:
                DelSet.append(leaf)
                break
            chldren.remove(leaf)
            if chldren[0] not in lvs:
                DelSet.append(leaf)
                break
            neighboors.add(chldren[0])
        weigh[leaf] = len(neighboors)-1
    for leaf in DelSet:
        weigh.pop(leaf)
        lvs.remove(leaf)
    return lvs, weigh