import copy
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

def showGraph(Net,labels=True):
    subax1 = plt.plot()
    nx.draw_kamada_kawai(Net, with_labels=labels, font_weight='bold')
    #nx.draw_planar(Net, with_labels=True, font_weight='bold')
    plt.show()
def Getleaves(net):
    return [u for u in net.nodes() if net.out_degree(u) == 0]



class Case:
    def __init__(self,TreeSet,Cherries,leaves,w,C,s):
        self.TreeSet = TreeSet
        if leaves == None:
            leaves = Getleaves(TreeSet[0])
        self.leaves = leaves

        if Cherries == None:
            Cherries = dict()
            for iTree in TreeSet:
                #print(iTree)
                cherSet = []
                checkleaves = self.leaves.copy()
                while len(checkleaves) > 1:
                    leaf = checkleaves[0]
                    p = list(TreeSet[iTree].predecessors(leaf))[0]
                    children = list(TreeSet[iTree].successors(p))
                    if children[0] in self.leaves and children[1] in self.leaves:
                        cherSet.append({children[0],children[1]})
                        checkleaves.remove(children[0])
                        checkleaves.remove(children[1])
                    else:
                        checkleaves.remove(leaf)
                Cherries[iTree] = cherSet
        self.Cherries = Cherries
        #print(Cherries)
        self.w = w

        self.C = C
        self.s = s
        self.phi = np.log(2)/np.log(5)


    def pickLeaf(self,leaf):
        RCherries = []
        for i in self.TreeSet:
            # finding parent, grand_parent and neighbor
            p = list(self.TreeSet[i].predecessors(leaf))[0]
            gp = list(self.TreeSet[i].predecessors(p))[0]
            chldren = list(self.TreeSet[i].successors(p))
            chldren.remove(leaf)

            # remove cherry from list
            cher = [list(self.TreeSet[i].successors(p))[0],list(self.TreeSet[i].successors(p))[1]]
            if (cher[0],cher[1]) not in RCherries and (cher[1],cher[0]) not in RCherries:
                RCherries.append((cher[0],cher[1]))
            self.Cherries[i].remove(set(self.TreeSet[i].successors(p)))

            # Correction tree

            self.TreeSet[i].remove_node(leaf)
            self.TreeSet[i].remove_node(p)
            self.TreeSet[i].add_edge(gp, chldren[0])

            # adding new cherry
            gchldren = list(self.TreeSet[i].successors(gp))
            gchldren.remove(chldren[0])
            if gchldren[0] in self.leaves:
                self.Cherries[i].append(set([gchldren[0],chldren[0]]))

        # Constraint correction
        self.C = [cons for cons in self.C if cons[0] != leaf and cons[1] != leaf]
        # pick sequence
        self.s.append(leaf)
        # weight
        self.w += len(RCherries) - 1
        # leaves
        self.leaves.remove(leaf)

    def ripeCherry(self):
        # NoWeightCherry
        for cherry in self.Cherries[0]:
            for Itree in range(1,len(self.Cherries)):
                if cherry not in self.Cherries[Itree]:
                    break
                if Itree == len(self.Cherries)-1:
                    return list(cherry)[0]
        return None

    def ripeConstrCherry(self):

        pickAble = set(self.leaves)
        for Itree in self.Cherries:
            combSet = set()
            for cherry in self.Cherries[Itree]:
                combSet |= cherry
            pickAble &= combSet
            if len(pickAble) == 0:
                return None

        for leaf in pickAble:
            nextleaf = False
            for Itree in self.Cherries:
                for cher in self.Cherries[Itree]:
                    if leaf in cher:
                        for neigh in cher:
                            if neigh == leaf:
                                continue
                            if (leaf,neigh) not in self.C:
                                nextleaf = True
                                break
                if nextleaf:
                    break
                if Itree == len(self.Cherries)-1:
                    return leaf

        return None

    def getPC(self):
        pi = set()
        for cons in self.C:
            pi.add(cons[0])

        return self.phi * len(self.C) + (1 - 2 * self.phi) * len(pi)

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
        for I in self.Cherries:
            for cher in self.Cherries[I]:
                cher2 = list(cher)
                cherSet.add((cher2[0], cher2[1]))
                cherSet.add((cher2[1], cher2[0]))

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
        for I in self.Cherries:
            for cher in self.Cherries[I]:
                cher2 = list(cher)
                cherSet.add((cher2[0], cher2[1]))
                cherSet.add((cher2[1], cher2[0]))

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


def Pick(Case):
    while True:
        if len(Case.leaves) == 2:
            return
        ripe = Case.ripeCherry()
        if ripe != None:
            Case.pickLeaf(ripe)
            #print("Found ripe ", ripe)
            continue
        ripe = Case.ripeConstrCherry()
        if ripe != None:
            Case.pickLeaf(ripe)
            #print("Found C ripe ", ripe)
            continue

        return



def FPT(T):
    # T: set of trees
    check = []
    sol = []
    check.append(Case(T,None,None,0,[],[]))

    k = len(Getleaves(T[0])) ** 2

    while len(check) > 0:
        case = check.pop(0)

        if k - case.w - case.getPC() < 0:
            del case
            continue
        #print("check C", case.C, case.s)
        #for tree in case.TreeSet:
        #    showGraph(case.TreeSet[tree])
        Pick(case)

        #for tree in case.TreeSet:
        #    showGraph(case.TreeSet[tree])

        if len(case.leaves) == 2:
            s = case.s.copy()
            s.append(case.leaves[0])
            s.append(case.leaves[1])
            if k > case.w:
                k = case.w
                sol = []
            #print("found solution" , k, s )
            if [k,s] not in sol:
                sol.append([k,s])
            del case
            continue

        cher = case.case1()
        if cher != None:
            check.insert(0,Case(copy.deepcopy(case.TreeSet),copy.deepcopy(case.Cherries),case.leaves.copy(),case.w,case.C + [(cher[0],cher[1])],case.s.copy()))
            check.insert(0,Case(copy.deepcopy(case.TreeSet),copy.deepcopy(case.Cherries),case.leaves.copy(),case.w,case.C + [(cher[1],cher[0])],case.s.copy()))
            #print("cher", cher)
            del case
            continue

        triple = case.case2()
        if triple != None:
            check.insert(0,Case(copy.deepcopy(case.TreeSet),copy.deepcopy(case.Cherries),case.leaves.copy(),case.w,case.C + [(triple[1],triple[0])],case.s.copy()))
            check.insert(0,Case(copy.deepcopy(case.TreeSet),copy.deepcopy(case.Cherries),case.leaves.copy(),case.w,case.C + [(triple[2],triple[0])],case.s.copy()))
            check.insert(0,Case(copy.deepcopy(case.TreeSet),copy.deepcopy(case.Cherries),case.leaves.copy(),case.w,case.C + [(triple[0],triple[1]),(triple[0],triple[2])],case.s.copy()))
            #print("triple", triple)
            del case
            continue

        del case

    return sol




















