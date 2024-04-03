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
from Utils.Solvers.TempSolver_2 import TempSolver_2
from Utils.Solvers.TreeAn import reLabelTreeSet, getLeaves, showGraph, getRoot
from Utils.CherryDepend import isBasicTemp4_2, isBasicTemp3_2, isLastTemp3, isLastTemp3_2
import networkx as nx

from Utils.TreeNetUtils import showBetterTreeSet


class Case_T3:
    def __init__(self,
                 input_tree_set=None):
        self.tree_set = dict()
        self.label = None
        self.weight = 0
        self.leaves = dict()
        self.to_pick = None
        self.sol = []

        if input_tree_set is not None:
            for tree in input_tree_set:
                self.tree_set[tree] = input_tree_set[tree].copy()
            self.label = reLabelTreeSet(self.tree_set)
            self.label = [set(i) for i in self.label]
            self.to_pick = getLeaves(self.tree_set[0])

    def copy(self):
        new_case = Case_T3()

        new_case.tree_set = dict()
        for tree in self.tree_set:
            new_case.tree_set[tree] = self.tree_set[tree].copy()

        new_case.label = copy.deepcopy(self.label)
        new_case.weight = self.weight
        new_case.leaves = copy.deepcopy(self.leaves)
        new_case.to_pick = self.to_pick.copy()
        new_case.sol = self.sol.copy()

        return new_case

    def start(self,leaf):
        start_set = []
        for _ in self.tree_set:
            start_set.append(0)
        self.leaves[leaf] = start_set
        self.to_pick.remove(leaf)

    def add(self,leaf):
        buddies = []
        for tree in self.tree_set:
            for bud in self.leaves:
                if leaf in self.label[self.leaves[bud][tree]]:
                    children = list(self.tree_set[tree].successors(self.leaves[bud][tree]))
                    inter = len(self.label[children[0]] & {bud,leaf})
                    if inter == 1:
                        buddies.append(bud)
                        break
                    else:
                        return None
        return buddies

    def grow(self,leaf,buddies):
        new_set = []

        for tree in range(len(self.tree_set)):
            bud = buddies[tree]
            for child in self.tree_set[tree].successors(self.leaves[bud][tree]):
                if bud in self.label[child]:
                    self.leaves[bud][tree] = child
                if leaf in self.label[child]:
                    new_set.append(child)

        self.weight += len(set(buddies))-1
        self.leaves[leaf] = new_set

        self.to_pick.remove(leaf)
        self.sol.append(leaf)




def rBranch(tree_set):

    nr_tree = len(tree_set)
    nr_leaves = len(getLeaves(tree_set[0]))
    # print(nr_tree * nr_leaves)
    main_case = Case_T3(tree_set)
    todo = dict()
    for i in range(nr_tree * nr_leaves):
        todo[i] = []


    for leaf in main_case.to_pick:
        new_case = main_case.copy()
        new_case.start(leaf)
        todo[0].append(new_case)


    checked = []
    for _ in range(len(getLeaves(tree_set[0]))+2):
        checked.append([])

    done = []
    for _ in range(len(getLeaves(tree_set[0])) + 2):
        done.append([])


    while len(todo) > 0:
        s_key = min(todo.keys())

        while len(todo[s_key]) > 0:

            case = todo[s_key].pop() # get subproblem min weight

            if len(case.to_pick) == 0: # check done
                return case.weight, case.sol

            leaf_set = case.to_pick.copy()
            if leaf_set in done[len(leaf_set)]:
                continue
            else:
                done[len(leaf_set)].append(leaf_set)



            for leaf in case.to_pick:
                buddies = case.add(leaf)
                if buddies is not None:
                    new_leaf_set = case.to_pick.copy()
                    new_leaf_set.remove(leaf)

                    if new_leaf_set in checked[len(new_leaf_set)]:
                        continue

                    new_case = case.copy()
                    new_case.grow(leaf, buddies)
                    # print(new_case.weight)
                    todo[new_case.weight].append(new_case)

        # print('del', s_key)
        del todo[s_key]


    # print(I)
    return -1, []



if __name__ == '__main__':
    for i in range(40000):
        # i = 4
        filename = "../../../Data/ret/combL11-15/inst_" + str(i) + ".pickle"
        # filename = "../../../Data/ret/combL6-10/inst_" + str(i) + ".pickle"
        # i = 116
        # filename = "../../../Data/ret/40L_15R_2T/inst_" + str(i) + ".pickle"
        # filename = "../../../Data/ret/25L_15R_3T/inst_" + str(i) + ".pickle"
        with open(filename, 'rb') as f:
            [treeSet, retValues] = pickle.load(f)
    # print(retValues)

        minr = min([i[1] for i in retValues if i[1] != -1])
        # pickCher(treeSet,2)
        reLabelTreeSet(treeSet)
        start = time.time()
        copyset = copy.deepcopy(treeSet)
        print(minr)
        print(rBranch(copyset))
        eind = time.time()
        print(eind-start)


