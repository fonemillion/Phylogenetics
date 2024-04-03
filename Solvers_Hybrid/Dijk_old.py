import copy
import pickle
import time
from itertools import combinations
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from DataGen.pickClass import pickCher
from Utils.Solvers.GreedyPicker import greedyPick3
from Utils.Solvers.TempSolver import tempSolver
from Utils.Solvers.TreeAn import reLabelTreeSet, getLeaves, showGraph, getRoot
from Utils.CherryDepend import isBasicTemp4_2, isBasicTemp3_2, isLastTemp3, isLastTemp3_2
import networkx as nx

from Utils.TreeNetUtils import showBetterTreeSet


class Case_T:
    def __init__(self,
                 input_data=None,
                 input_is_tree=True):
        self.net = nx.DiGraph()
        self.node_dict = None
        self.in_nr_tree = None
        self.leaves = None
        self.nr_tree = None
        self.clusters = None
        self.picked = set()
        self.sol = []
        self.weight = 0

        if input_is_tree:
            self.treeToNet(input_data)
            self.leaves = getLeaves(self.net)
        # else:
        #     self.net = input_data.net.copy()
        #     self.node_dict = input_data.node_dict.copy()
        #     self.in_nr_tree = input_data.in_nr_tree.copy()
        #     self.leaves = input_data.leaves.copy()
        #     self.nr_tree = input_data.nr_tree
        #     self.clusters = input_data.clusters.copy()
        #     self.picked = input_data.picked.copy()

    def get_cluster(self, pick):

        descend = []
        todo = [pick]
        while len(todo) > 0:
            item = todo.pop()
            if item in descend:
                continue
            else:
                descend.append(item)
                for child in self.net.successors(item):
                    todo.append(child)

        new_case = Case_T(input_is_tree=False)

        new_case.net = self.net.copy()
        new_case.node_dict = copy.deepcopy(self.node_dict)
        new_case.in_nr_tree = copy.deepcopy(self.in_nr_tree)
        new_case.leaves = self.leaves.copy()
        new_case.nr_tree = self.nr_tree
        new_case.clusters = self.clusters.copy()
        new_case.picked = set()

        new_case.clusters.remove(pick)
        for node in self.net.nodes():
            if node not in descend:
                new_case.net.remove_node(node)
                del new_case.node_dict[node]
                del new_case.in_nr_tree[node]
                if node in new_case.leaves:
                    new_case.leaves.remove(node)
                if node in new_case.clusters:
                    new_case.clusters.remove(node)
        return new_case

    def copy(self):
        new_case = Case_T(input_is_tree=False)
        new_case.net = self.net.copy()
        new_case.node_dict = copy.deepcopy(self.node_dict)
        new_case.in_nr_tree = copy.deepcopy(self.in_nr_tree)
        new_case.leaves = self.leaves.copy()
        new_case.nr_tree = self.nr_tree
        new_case.clusters = self.clusters.copy()
        new_case.picked = self.picked.copy()
        new_case.sol = self.sol.copy()
        new_case.weight = self.weight

        return new_case

    def treeToNet(self, tree_set) -> nx.DiGraph:
        reLabelTreeSet(tree_set)
        self.node_dict = dict(enumerate(reLabelTreeSet(tree_set)))
        # create network
        net = nx.DiGraph()
        for i in tree_set:
            net.add_nodes_from(tree_set[i].nodes())
            net.add_edges_from(tree_set[i].edges())
        self.net = net
        #
        in_nr_tree = dict()
        for node in net:
            in_nr_tree[node] = 0
        for tree in tree_set:
            for node in tree_set[tree]:
                in_nr_tree[node] += 1
        self.in_nr_tree = in_nr_tree
        self.nr_tree = len(tree_set)
        self.clusters = []
        for node in net:
            if node in getLeaves(net) or node == getRoot(net):
                continue
            if self.in_nr_tree[node] == self.nr_tree:
                self.clusters.append(node)

    def pick_leaf(self, node):
        # showGraph(self.net)
        # delete node itself

        ancestors = []
        todo = []
        for p in self.net.predecessors(node):
            todo.append(p)
            self.weight += 1
        self.weight -= 1
        self.sol.append(node)
        self.net.remove_node(node)
        del self.node_dict[node]
        del self.in_nr_tree[node]
        self.picked.add(node)
        self.leaves.remove(node)
        while len(todo) > 0:
            item = todo.pop(0)
            if item in ancestors:
                continue
            for p in self.net.predecessors(item):
                if not p in todo:
                    todo.append(p)
            if len(self.node_dict[item]) == 2:
                ancestors.append(item)
                for child in self.net.successors(item):
                    for p in self.net.predecessors(item):
                        self.net.add_edge(p, child)
                del self.node_dict[item]
                del self.in_nr_tree[item]
                self.net.remove_node(item)
                if item in self.clusters:
                    self.clusters.remove(item)
            else:
                self.node_dict[item].remove(node)
                same_key = [key for key in self.node_dict if
                            key != item and self.node_dict[item] == self.node_dict[key]]
                if len(same_key) != 0:
                    ancestors.append(same_key[0])
                    for child in self.net.successors(item):
                        if child != same_key[0]:
                            self.net.add_edge(same_key[0], child)
                    for p in self.net.predecessors(item):
                        self.net.add_edge(p, same_key[0])
                    in_tree = self.in_nr_tree[item]
                    del self.node_dict[item]
                    del self.in_nr_tree[item]
                    self.net.remove_node(item)
                    self.in_nr_tree[same_key[0]] += in_tree
                    if self.in_nr_tree[same_key[0]] == self.nr_tree:  # new cluster
                        self.clusters.append(same_key[0])
                    self.in_nr_tree[same_key[0]] = min(self.in_nr_tree[same_key[0]], self.nr_tree)  # root correction
                else:
                    ancestors.append(item)

    def can_pick(self):
        pickable = []

        for leaf in self.leaves:
            if all([len(self.node_dict[p]) == 2 for p in self.net.predecessors(leaf)]):
                pickable.append(leaf)

        return pickable

    def can_pick2(self):
        pickable = []

        for leaf in self.leaves:
            if all([len(self.node_dict[p]) == 2 for p in self.net.predecessors(leaf)]):
                if len(list(self.net.predecessors(leaf))) == 1:
                    return [leaf]
                pickable.append(leaf)
        return pickable


    def can_pick3(self):
        pickable = []

        for leaf in self.leaves:
            if all([len(self.node_dict[p]) == 2 for p in self.net.predecessors(leaf)]):
                if len(list(self.net.predecessors(leaf))) == 1:
                    return [leaf]
                pickable.append(leaf)

        for leaf in pickable:
            LH = True
            for p in self.net.predecessors(leaf):
                children = list(self.net.successors(p))
                children.remove(leaf)
                child = children[0]
                Block = False
                for p2 in self.net.predecessors(child):
                    if leaf in self.node_dict[p2] and len(self.node_dict[p2]) > 2:
                        Block = True
                        break
                if not Block:
                    LH = False
                    break
            if LH:
                return [leaf]

        return pickable

    def can_pick4(self):


        if len(self.clusters) > 0:
            max_clus = 0
            pick = 0
            for clus in self.clusters:
                if len(self.node_dict[clus]) == 2:
                    return [self.node_dict[clus][0]]
                length = len(self.node_dict[clus])
                if length > max_clus:
                    max_clus = length
                    pick = clus

            Lset = self.node_dict[pick]
        else:
            Lset = self.leaves


        pickable = []


        for leaf in Lset:
            if all([len(self.node_dict[p]) == 2 for p in self.net.predecessors(leaf)]):
                pickable.append(leaf)

        for leaf in pickable:
            LH = True
            for p in self.net.predecessors(leaf):
                children = list(self.net.successors(p))
                children.remove(leaf)
                child = children[0]
                Block = False
                for p2 in self.net.predecessors(child):
                    if leaf in self.node_dict[p2] and len(self.node_dict[p2]) > 2:
                        Block = True
                        break
                if not Block:
                    LH = False
                    break
            if LH:
                return [leaf]

        return pickable

def Branch(tree_set, isTree=True):
    if isTree:
        main_case = Case_T(tree_set, input_is_tree=True)
    else:
        main_case = tree_set

    todo = dict()
    for i in range(main_case.nr_tree * len(main_case.leaves)):
        todo[i] = []
    todo[0] = [main_case]
    done = []
    for _ in range(len(getLeaves(main_case.net)) + 2):
        done.append([])


    while len(todo) > 0:
        s_key = min(todo.keys())

        while len(todo[s_key]) > 0:

            case = todo[s_key].pop() # get subproblem min weight

            if len(case.leaves) <= 2: # check done
                sol = case.sol
                sol.append(case.leaves[0])
                if len(case.leaves) > 1:
                    sol.append(case.leaves[1])
                # print(I)
                return case.weight, sol


            leaf_set = case.leaves.copy()
            if leaf_set in done[len(leaf_set)]:
                continue
            else:
                done[len(leaf_set)].append(leaf_set)



            can_pick = case.can_pick()
            for leaf in can_pick:

                new_leaf_set = case.leaves.copy()
                new_leaf_set.remove(leaf)

                if new_leaf_set in done[len(new_leaf_set)]:
                    continue

                new_case = case.copy()
                new_case.pick_leaf(leaf)
                todo[new_case.weight].append(new_case)
                # print(new_case.node_dict)
        del todo[s_key]

    # print(I)
    return -1, []

def Branch2(tree_set, isTree=True):
    if isTree:
        main_case = Case_T(tree_set, input_is_tree=True)
    else:
        main_case = tree_set

    todo = dict()
    for i in range(main_case.nr_tree * len(main_case.leaves)):
        todo[i] = []
    todo[0] = [main_case]
    done = []
    for _ in range(len(getLeaves(main_case.net)) + 2):
        done.append([])


    while len(todo) > 0:
        s_key = min(todo.keys())

        while len(todo[s_key]) > 0:

            case = todo[s_key].pop() # get subproblem min weight

            if len(case.leaves) <= 2: # check done
                sol = case.sol
                sol.append(case.leaves[0])
                if len(case.leaves) > 1:
                    sol.append(case.leaves[1])
                # print(I)
                return case.weight, sol


            leaf_set = case.leaves.copy()
            if leaf_set in done[len(leaf_set)]:
                continue
            else:
                done[len(leaf_set)].append(leaf_set)



            can_pick = case.can_pick2()
            for leaf in can_pick:

                new_leaf_set = case.leaves.copy()
                new_leaf_set.remove(leaf)

                if new_leaf_set in done[len(new_leaf_set)]:
                    continue

                new_case = case.copy()
                new_case.pick_leaf(leaf)
                todo[new_case.weight].append(new_case)
                # print(new_case.node_dict)
        del todo[s_key]

    # print(I)
    return -1, []

def Branch3(tree_set, isTree=True):
    if isTree:
        main_case = Case_T(tree_set, input_is_tree=True)
    else:
        main_case = tree_set

    todo = dict()
    for i in range(main_case.nr_tree * len(main_case.leaves)):
        todo[i] = []
    todo[0] = [main_case]
    done = []
    for _ in range(len(getLeaves(main_case.net)) + 2):
        done.append([])


    while len(todo) > 0:
        s_key = min(todo.keys())

        while len(todo[s_key]) > 0:

            case = todo[s_key].pop() # get subproblem min weight

            if len(case.leaves) <= 2: # check done
                sol = case.sol
                sol.append(case.leaves[0])
                if len(case.leaves) > 1:
                    sol.append(case.leaves[1])
                # print(I)
                return case.weight, sol


            leaf_set = case.leaves.copy()
            if leaf_set in done[len(leaf_set)]:
                continue
            else:
                done[len(leaf_set)].append(leaf_set)



            can_pick = case.can_pick3()
            for leaf in can_pick:

                new_leaf_set = case.leaves.copy()
                new_leaf_set.remove(leaf)

                if new_leaf_set in done[len(new_leaf_set)]:
                    continue

                new_case = case.copy()
                new_case.pick_leaf(leaf)
                todo[new_case.weight].append(new_case)
                # print(new_case.node_dict)
        del todo[s_key]

    # print(I)
    return -1, []

def Branch4_old(tree_set, isTree=True):
    if isTree:
        main_case = Case_T(tree_set, input_is_tree=True)
    else:
        main_case = tree_set
    todo = dict()

    for i in range(main_case.nr_tree * len(main_case.leaves)):
        todo[i] = []
    todo[0] = [main_case]
    done = []
    for _ in range(len(getLeaves(main_case.net)) + 2):
        done.append([])
    # I = 1
    clusters_solved = []
    clust_ans = []

    while len(todo) > 0:
        s_key = min(todo.keys())

        while len(todo[s_key]) > 0:
            # I+=1

            case = todo[s_key].pop()
            leaf_set = case.leaves.copy()
            if leaf_set in done[len(leaf_set)]:
                continue
            else:
                done[len(leaf_set)].append(leaf_set)

            if len(case.leaves) <= 2:
                sol = case.sol
                sol.append(case.leaves[0])
                if len(case.leaves) > 1:
                    sol.append(case.leaves[1])
                # print(I)
                return case.weight, sol

            if len(case.clusters) > 0:
                max_clus = 0
                pick = 0
                for clus in case.clusters:
                    if len(case.node_dict[clus]) == 2:
                        pick = min(case.node_dict[clus])
                        break
                    length = len(case.node_dict[clus])
                    if length > max_clus:
                        max_clus = length
                        pick = clus
                if len(case.node_dict[pick]) == 1:
                    case.pick_leaf(pick)
                else:
                    if case.node_dict[pick] in clusters_solved:
                        sol_partial = clust_ans[clusters_solved.index(case.node_dict[pick])]
                    else:
                        _, sol_partial = Branch4(case.get_cluster(pick), isTree=False)
                        clust_ans.append(sol_partial.copy())
                        clusters_solved.append(case.node_dict[pick].copy())
                    if len(sol_partial) == 0:
                        continue
                    else:
                        for i in range(len(sol_partial) - 1):
                            case.pick_leaf(sol_partial[i])
                        todo[case.weight].append(case)
                        continue

            can_pick = case.can_pick()
            for leaf in can_pick:

                new_leaf_set = case.leaves.copy()
                new_leaf_set.remove(leaf)
                if new_leaf_set in done[len(new_leaf_set)]:
                    continue

                new_case = case.copy()
                new_case.pick_leaf(leaf)
                todo[new_case.weight].append(new_case)
                # print(new_case.node_dict)
        del todo[s_key]

    # print(I)
    return -1, []

def Branch4(tree_set, isTree=True):
    if isTree:
        main_case = Case_T(tree_set, input_is_tree=True)
    else:
        main_case = tree_set
    todo = dict()

    for i in range(main_case.nr_tree * len(main_case.leaves)):
        todo[i] = []
    todo[0] = [main_case]
    done = []
    for _ in range(len(getLeaves(main_case.net)) + 2):
        done.append([])
    # I = 1
    clusters_solved = []
    clust_ans = []

    while len(todo) > 0:
        s_key = min(todo.keys())

        while len(todo[s_key]) > 0:
            # I+=1

            case = todo[s_key].pop()
            leaf_set = case.leaves.copy()
            if leaf_set in done[len(leaf_set)]:
                continue
            else:
                done[len(leaf_set)].append(leaf_set)

            if len(case.leaves) <= 2:
                sol = case.sol
                sol.append(case.leaves[0])
                if len(case.leaves) > 1:
                    sol.append(case.leaves[1])
                # print(I)
                return case.weight, sol

            can_pick = case.can_pick4()
            for leaf in can_pick:

                new_leaf_set = case.leaves.copy()
                new_leaf_set.remove(leaf)
                if new_leaf_set in done[len(new_leaf_set)]:
                    continue

                new_case = case.copy()
                new_case.pick_leaf(leaf)
                todo[new_case.weight].append(new_case)
                # print(new_case.node_dict)
        del todo[s_key]

    # print(I)
    return -1, []



if __name__ == '__main__':
    for i in range(40000):
        # i = 4
        # filename = "../../../Data/ret/combL16-20/inst_" + str(i) + ".pickle"
        filename = "../../../Data/ret/Net/NL22/T2/inst_" + str(i) + ".pickle"
        # filename = "../../../Data/ret/combL6-10/inst_" + str(i) + ".pickle"
        # i = 116
        # filename = "../../../Data/ret/40L_15R_2T/inst_" + str(i) + ".pickle"
        # filename = "../../../Data/ret/25L_15R_3T/inst_" + str(i) + ".pickle"
        with open(filename, 'rb') as f:
            [treeSet, retValues] = pickle.load(f)
    # print(retValues)
        case = Case_T(treeSet, input_is_tree=True)
        minr = min([i[1] for i in retValues if i[1] != -1])
        # pickCher(treeSet,2)
        reLabelTreeSet(treeSet)
        print(minr,i)

        start = time.time()
        copyset = copy.deepcopy(treeSet)
        print(Branch(copyset)[0])
        eind = time.time()
        print(eind-start)

        start = time.time()
        copyset = copy.deepcopy(treeSet)
        print(Branch2(copyset)[0])
        eind = time.time()
        print(eind-start)

        start = time.time()
        copyset = copy.deepcopy(treeSet)
        print(Branch3(copyset)[0])
        eind = time.time()
        print(eind-start)

        start = time.time()
        copyset = copy.deepcopy(treeSet)
        print(Branch4(copyset)[0])
        eind = time.time()
        print(eind-start)

