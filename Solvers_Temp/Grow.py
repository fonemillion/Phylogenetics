# Temp solver grow


import copy
import time

from Utils.TreeAn import reLabelTreeSet, getLeaves
from Utils.InstanceGenerators.BiNetGenTemp import net_to_tree2, simulation_temp

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

    def start(self, leaf):
        start_set = []
        for _ in self.tree_set:
            start_set.append(0)
        self.leaves[leaf] = start_set
        self.to_pick.remove(leaf)

    def add(self, leaf):
        buddies = []
        for tree in self.tree_set:
            for bud in self.leaves:
                if leaf in self.label[self.leaves[bud][tree]]:
                    children = list(self.tree_set[tree].successors(self.leaves[bud][tree]))
                    inter = len(self.label[children[0]] & {bud, leaf})
                    if inter == 1:
                        buddies.append(bud)
                        break
                    else:
                        return None
        return buddies

    def grow(self, leaf, buddies):
        new_set = []

        for tree in range(len(self.tree_set)):
            bud = buddies[tree]
            for child in self.tree_set[tree].successors(self.leaves[bud][tree]):
                if bud in self.label[child]:
                    self.leaves[bud][tree] = child
                if leaf in self.label[child]:
                    new_set.append(child)

        self.weight = len(set(buddies)) - 1
        self.leaves[leaf] = new_set

        self.to_pick.remove(leaf)
        self.sol.append(leaf)


def TempSolver_3(tree_set):
    main_case = Case_T3(tree_set)
    todo = []
    sol = []

    for leaf in main_case.to_pick:
        new_case = main_case.copy()
        new_case.start(leaf)
        todo.append(new_case)

    checked = []

    for _ in range(len(getLeaves(tree_set[0])) + 2):
        checked.append([])

    while len(todo) > 0:
        # I+=1
        case = todo.pop()

        if len(case.to_pick) == 0:
            sol = case.sol
            return sol

        for leaf in case.to_pick:
            buddies = case.add(leaf)
            if buddies is not None:
                new_set = case.to_pick.copy()
                new_set.remove(leaf)
                if new_set in checked[len(new_set)]:
                    continue
                else:
                    checked[len(new_set)].append(new_set)

                new_case = case.copy()
                new_case.grow(leaf, buddies)
                todo.append(new_case)

    # print(I)
    return sol


if __name__ == '__main__':

    R = 10
    L = 30
    T = 10
    net, _ = simulation_temp(L, R)
    treeSet = net_to_tree2(net, T)

    reLabelTreeSet(treeSet)
    print('----------------')

    start = time.time()
    print(TempSolver_3(treeSet))
    eind = time.time()
    print(eind - start)

