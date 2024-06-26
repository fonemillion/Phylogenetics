# The Grow algorithm used

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

    # print(todo)

    for leaf in main_case.to_pick:
        new_case = main_case.copy()
        new_case.start(leaf)
        todo[0].append(new_case)


    checked = []
    for _ in range(len(getLeaves(tree_set[0]))+2):
        checked.append([])

    done = []
    for _ in range(len(getLeaves(tree_set[0])) + 2):
        done.append(set())


    while len(todo) > 0:
        s_key = min(todo.keys())

        while len(todo[s_key]) > 0:

            case = todo[s_key].pop() # get subproblem min weight

            if len(case.to_pick) == 0: # check done
                return case.weight, case.sol

            leaf_set = tuple(case.to_pick)
            if leaf_set in done[len(leaf_set)]:
                continue
            else:
                done[len(leaf_set)].add(leaf_set)



            for leaf in case.to_pick:
                buddies = case.add(leaf)
                if buddies is not None:
                    new_leaf_set = case.to_pick.copy()
                    new_leaf_set.remove(leaf)

                    if tuple(new_leaf_set) in done[len(new_leaf_set)]:
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
    R = 10
    L = 25
    T = 10
    for i in range(10):
        net, _ = simulation_temp(L, R)
        treeSet = net_to_tree2(net, T)


        reLabelTreeSet(treeSet)
        print(i, '----------------')

        start = time.time()
        copyset = copy.deepcopy(treeSet)
        print(rBranch(copyset))
        eind = time.time()
        print(eind-start)


