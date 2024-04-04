# The Branch and Bound algorithms used



import copy
import random
import time
from Utils.TreeAn import reLabelTreeSet, getLeaves, getRoot
import networkx as nx
from Utils.InstanceGenerators.BiNetGenTemp import net_to_tree2, simulation_temp

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
            leaves = getLeaves(self.net)
            leaves.sort()
            self.leaves = tuple(leaves)


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
        new_case.leaves = copy.deepcopy(self.leaves)
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
        leaves = list(self.leaves)
        leaves.remove(node)
        self.leaves = tuple(leaves)
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

    def get_ub(self):
        test_case = self.copy()
        while len(test_case.leaves) > 2:
            pickable = test_case.can_pick()
            if len(pickable) == 0:
                return None
            test_case.pick_leaf(random.choice(pickable))
        return test_case

    def get_lb(self, ub):
        return self.weight + 1

    def get_lb2(self, ub):
        if 3*(ub - self.weight) >= len(self.leaves)+5:
            return 0
        pickable = []
        lb = 0
        for leaf in self.leaves:
            pble = True
            in_cherries = set()
            for p in self.net.predecessors(leaf):
                if len(self.node_dict[p]) == 2:
                    leaves = self.node_dict[p].copy()
                    leaves.remove(leaf)
                    in_cherries.add(leaves[0])
                else:
                    pble = False
                    break
            if pble:
                pickable.append([leaf,in_cherries])

        while len(pickable) > 0:
            max_len = 0
            pick = None
            for i in range(len(pickable)):
                if len(pickable[i][1]) > max_len:
                    max_len = len(pickable[i][1])
                    pick = i
            pick = pickable.pop(pick)

            lb += len(pick[1]) - 1
            lvs = set(pick[1])
            lvs.add(pick[0])
            for i in range(len(pickable)-1,-1,-1):
                if pickable[i][0] in lvs:
                    pickable.pop(i)
                else:
                    pickable[i][1] = pickable[i][1] - lvs
                    if len(pickable[i][1]) <= 1:
                        pickable.pop(i)

        return self.weight + lb

    def get_lb3(self, ub):
        if 3*(ub - self.weight) >= len(self.leaves)+5:
            return 0
        pickable = []
        lb = 0

        for leaf in self.leaves:
            in_cherries = set()
            for p in self.net.predecessors(leaf):
                for item in self.node_dict[p]:
                    in_cherries.add(item)
            in_cherries2 = set()
            in_cherries.remove(leaf)
            for leaf2 in in_cherries:
                for p2 in self.net.predecessors(leaf2):
                    if leaf in self.node_dict[p2]:
                        in_cherries2.add(leaf2)
                        break
            if len(in_cherries2) > 1:
                pickable.append([leaf, in_cherries2])
        # print(pickable)

        while len(pickable) > 0:
            max_len = 0
            pick = None
            for i in range(len(pickable)):
                if len(pickable[i][1]) > max_len:
                    max_len = len(pickable[i][1])
                    pick = i
            pick = pickable.pop(pick)

            lb += len(pick[1]) - 1
            lvs = set(pick[1])
            lvs.add(pick[0])
            for i in range(len(pickable)-1,-1,-1):
                if pickable[i][0] in lvs:
                    pickable.pop(i)
                else:
                    pickable[i][1] = pickable[i][1] - lvs
                    if len(pickable[i][1]) <= 1:
                        pickable.pop(i)

        return self.weight + lb

    def get_lb4(self, ub):
        if 3*(ub - self.weight) >= len(self.leaves)+5:
            return 0
        lb = 0
        cherries = set()
        anti = set()
        for leaf in self.leaves:
            for p in self.net.predecessors(leaf):
                if len(self.node_dict[p]) == 2:
                    cherries.add(p)

        for cherry in cherries:
            fca = set()
            leaf0 = self.node_dict[cherry][0]
            leaf1 = self.node_dict[cherry][1]

            todo = []
            for p in self.net.predecessors(leaf0):
                todo.append(p)
            while len(todo) > 0:
                par = todo.pop()
                if leaf1 in self.node_dict[par]:
                    if len(self.node_dict[par]) > 2:
                        fca.add(par)
                else:
                    for p in self.net.predecessors(par):
                        todo.append(p)
            for q in self.net.predecessors(cherry):
                for p in fca:
                    if set(self.node_dict[q]).issubset(set(self.node_dict[p])):
                        anti.add((leaf0,leaf1))
                    break
                if (leaf0,leaf1) in anti:
                    break
        anti = list(anti)
        while len(anti) > 1:
            cher = random.choice(anti)
            anti = [item for item in anti if item[0] not in cher and item[1] not in cher]
            lb += 1





        return self.weight + lb



def BB(tree_set, isTree=True, ub = None):
    if isTree:
        main_case = Case_T(tree_set, input_is_tree=True)
    else:
        main_case = tree_set
    todo = dict()

    for i in range(len(main_case.leaves)+1):
        todo[i] = dict()
    todo[len(main_case.leaves)][main_case.leaves] = main_case

    if ub is None:
        upperbound = len(getLeaves(main_case.net)) ** 2
    else:
        upperbound = ub
    no_up = upperbound
    best_sol = []


    while len(todo) > 0:
        s_key = max(todo.keys())
        best_weight_layer = upperbound
        while len(todo[s_key]) > 0:

            _, case = todo[s_key].popitem()
            if len(case.leaves) <= 2: # only 2 leaves left
                sol = case.sol
                sol.append(case.leaves[0])
                if len(case.leaves) > 1:
                    sol.append(case.leaves[1])
                if case.weight < upperbound:
                    upperbound = case.weight
                    best_sol = sol
                continue

            # dive
            if case.weight < best_weight_layer:
                best_weight_layer = case.weight
                c_up = case.get_ub()
                if c_up is not None:
                    if c_up.weight < upperbound:
                        sol = c_up.sol
                        sol.append(c_up.leaves[0])
                        if len(c_up.leaves) > 1:
                            sol.append(c_up.leaves[1])
                        # print(I)
                        upperbound = c_up.weight
                        best_sol = sol


            can_pick = case.can_pick4()

            if len(can_pick) > 1:
                if upperbound <= case.get_lb(upperbound):
                    continue


            for leaf in can_pick:
                new_leaf_set = list(case.leaves)
                new_leaf_set.remove(leaf)
                new_leaf_set = tuple(new_leaf_set)
                new_case = case.copy()
                new_case.pick_leaf(leaf)
                if new_leaf_set in todo[s_key-1]:
                    if todo[s_key-1][new_leaf_set].weight > new_case.weight:
                        todo[s_key - 1][new_leaf_set] = new_case
                else:
                    todo[s_key - 1][new_leaf_set] = new_case

        del todo[s_key]
    if upperbound == no_up:
        return -1, []
    else:
        return upperbound, best_sol

def BB2(tree_set, isTree=True, ub = None):
    if isTree:
        main_case = Case_T(tree_set, input_is_tree=True)
    else:
        main_case = tree_set
    todo = dict()

    for i in range(len(main_case.leaves)+1):
        todo[i] = dict()
    todo[len(main_case.leaves)][main_case.leaves] = main_case

    if ub is None:
        upperbound = len(getLeaves(main_case.net)) ** 2
    else:
        upperbound = ub
    no_up = upperbound
    best_sol = []

    while len(todo) > 0:
        s_key = max(todo.keys())
        best_weight_layer = upperbound
        while len(todo[s_key]) > 0:

            _, case = todo[s_key].popitem()
            if len(case.leaves) <= 2: # only 2 leaves left
                sol = case.sol
                sol.append(case.leaves[0])
                if len(case.leaves) > 1:
                    sol.append(case.leaves[1])
                if case.weight < upperbound:
                    upperbound = case.weight
                    best_sol = sol
                continue

            # dive
            if case.weight < best_weight_layer:
                best_weight_layer = case.weight
                c_up = case.get_ub()
                if c_up is not None:
                    if c_up.weight < upperbound:
                        sol = c_up.sol
                        sol.append(c_up.leaves[0])
                        if len(c_up.leaves) > 1:
                            sol.append(c_up.leaves[1])
                        # print(I)
                        upperbound = c_up.weight
                        best_sol = sol


            can_pick = case.can_pick4()

            if len(can_pick) > 1:
                if upperbound <= case.get_lb2(upperbound):
                    continue


            for leaf in can_pick:
                new_leaf_set = list(case.leaves)
                new_leaf_set.remove(leaf)
                new_leaf_set = tuple(new_leaf_set)
                new_case = case.copy()
                new_case.pick_leaf(leaf)
                if new_leaf_set in todo[s_key-1]:
                    if todo[s_key-1][new_leaf_set].weight > new_case.weight:
                        todo[s_key - 1][new_leaf_set] = new_case
                else:
                    todo[s_key - 1][new_leaf_set] = new_case

        del todo[s_key]
    if upperbound == no_up:
        return -1, []
    else:
        return upperbound, best_sol

def BB3(tree_set, isTree=True, ub = None):
    if isTree:
        main_case = Case_T(tree_set, input_is_tree=True)
    else:
        main_case = tree_set
    todo = dict()

    for i in range(len(main_case.leaves)+1):
        todo[i] = dict()
    todo[len(main_case.leaves)][main_case.leaves] = main_case

    if ub is None:
        upperbound = len(getLeaves(main_case.net)) ** 2
    else:
        upperbound = ub
    no_up = upperbound
    best_sol = []

    while len(todo) > 0:
        s_key = max(todo.keys())
        best_weight_layer = upperbound
        while len(todo[s_key]) > 0:

            _, case = todo[s_key].popitem()
            if len(case.leaves) <= 2: # only 2 leaves left
                sol = case.sol
                sol.append(case.leaves[0])
                if len(case.leaves) > 1:
                    sol.append(case.leaves[1])
                if case.weight < upperbound:
                    upperbound = case.weight
                    best_sol = sol
                continue

            # dive
            if case.weight < best_weight_layer:
                best_weight_layer = case.weight
                c_up = case.get_ub()
                if c_up is not None:
                    if c_up.weight < upperbound:
                        sol = c_up.sol
                        sol.append(c_up.leaves[0])
                        if len(c_up.leaves) > 1:
                            sol.append(c_up.leaves[1])
                        # print(I)
                        upperbound = c_up.weight
                        best_sol = sol


            can_pick = case.can_pick4()

            if len(can_pick) > 1:
                if upperbound <= case.get_lb3(upperbound):
                    continue


            for leaf in can_pick:
                new_leaf_set = list(case.leaves)
                new_leaf_set.remove(leaf)
                new_leaf_set = tuple(new_leaf_set)
                new_case = case.copy()
                new_case.pick_leaf(leaf)
                if new_leaf_set in todo[s_key-1]:
                    if todo[s_key-1][new_leaf_set].weight > new_case.weight:
                        todo[s_key - 1][new_leaf_set] = new_case
                else:
                    todo[s_key - 1][new_leaf_set] = new_case

        del todo[s_key]
    if upperbound == no_up:
        return -1, []
    else:
        return upperbound, best_sol

def BB4(tree_set, isTree=True, ub = None):
    if isTree:
        main_case = Case_T(tree_set, input_is_tree=True)
    else:
        main_case = tree_set
    todo = dict()

    for i in range(len(main_case.leaves)+1):
        todo[i] = dict()
    todo[len(main_case.leaves)][main_case.leaves] = main_case

    if ub is None:
        upperbound = len(getLeaves(main_case.net)) ** 2
    else:
        upperbound = ub
    no_up = upperbound
    best_sol = []

    while len(todo) > 0:
        s_key = max(todo.keys())
        best_weight_layer = upperbound
        while len(todo[s_key]) > 0:

            _, case = todo[s_key].popitem()
            if len(case.leaves) <= 2: # only 2 leaves left
                sol = case.sol
                sol.append(case.leaves[0])
                if len(case.leaves) > 1:
                    sol.append(case.leaves[1])
                if case.weight < upperbound:
                    upperbound = case.weight
                    best_sol = sol
                continue

            # dive
            if case.weight < best_weight_layer:
                best_weight_layer = case.weight
                c_up = case.get_ub()
                if c_up is not None:
                    if c_up.weight < upperbound:
                        sol = c_up.sol
                        sol.append(c_up.leaves[0])
                        if len(c_up.leaves) > 1:
                            sol.append(c_up.leaves[1])
                        # print(I)
                        upperbound = c_up.weight
                        best_sol = sol


            can_pick = case.can_pick4()

            if len(can_pick) > 1:
                if upperbound <= case.get_lb4(upperbound):
                    continue


            for leaf in can_pick:
                new_leaf_set = list(case.leaves)
                new_leaf_set.remove(leaf)
                new_leaf_set = tuple(new_leaf_set)
                new_case = case.copy()
                new_case.pick_leaf(leaf)
                if new_leaf_set in todo[s_key-1]:
                    if todo[s_key-1][new_leaf_set].weight > new_case.weight:
                        todo[s_key - 1][new_leaf_set] = new_case
                else:
                    todo[s_key - 1][new_leaf_set] = new_case

        del todo[s_key]
    if upperbound == no_up:
        return -1, []
    else:
        return upperbound, best_sol



if __name__ == '__main__':
    R = 10
    L = 30
    T = 10
    for i in range(10):

        net, _ = simulation_temp(L, R)
        treeSet = net_to_tree2(net, T)


        case = Case_T(treeSet, input_is_tree=True)

        reLabelTreeSet(treeSet)
        print(i, '----------------')



        start = time.time()
        copyset = copy.deepcopy(treeSet)
        print(BB(copyset,ub = R+2)[0], time.time()-start)
        start = time.time()
        copyset = copy.deepcopy(treeSet)
        print(BB2(copyset,ub = R+2)[0], time.time()-start)
        start = time.time()
        copyset = copy.deepcopy(treeSet)
        print(BB3(copyset,ub = R+2)[0], time.time()-start)
        start = time.time()
        copyset = copy.deepcopy(treeSet)
        print(BB4(copyset,ub = R+2)[0], time.time()-start)


