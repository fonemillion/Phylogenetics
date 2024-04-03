from Utils.TreeNetUtils import *
from Utils.InstanceGenerators.TreeEdgeGen import genTreeSetByEdge,genTreeSetByEdgeTemp
from itertools import combinations

def getBasicTrivialLeaf(treeSet):
    pickable = canPick(treeSet)
    tlset = set()
    label = getLabelTreeSet(treeSet)
    for leaf in pickable:
        neighboors = getNeighboors(treeSet,leaf)
        # print(leaf, neighboors)
        blocked = dict()
        for neigh in neighboors:
            blocked[neigh] = False
            for tree in treeSet:
                p = list(treeSet[tree].predecessors(neigh))[0]
                children = list(treeSet[tree].successors(p))
                children.remove(neigh)
                child = children[0]
                if len(label[child]) > 1 and leaf in label[child]:
                    blocked[neigh] = True
                    break
        if all(blocked[neigh] for neigh in blocked):
            tlset.add(leaf)
    return tlset



def getBasicTemp3(treeSet):
    lvs = getLeaves(treeSet[0])
    label = getLabelTreeSet(treeSet)

    for subset in combinations(lvs, 3):
        block = 0
        for i in range(3):
            j = (i+1) % 3
            k = (i+2) % 3
            blocked = False
            for tree in treeSet:
                p = list(treeSet[tree].predecessors(subset[i]))[0]
                children = list(treeSet[tree].successors(p))
                children.remove(subset[i])
                child = children[0]

                if subset[j] in label[child] and subset[k] in label[child]:
                    blocked = True
                    break
            if not blocked:
                break
            block += 1
        if block == 3:
            # print(subset)
            return True

    return False

def getBasicTemp4(treeSet):
    lvs = getLeaves(treeSet[0])
    label = getLabelTreeSet(treeSet)

    def sub_check(x,a,b):
        for tree in treeSet:
            p = list(treeSet[tree].predecessors(x))[0]
            children = list(treeSet[tree].successors(p))
            children.remove(x)
            child = children[0]
            if a in label[child] and b in label[child]:
                return True
                break
        return False

    for subset in combinations(lvs, 4):
        for subsubset in combinations(subset, 2):
            # print(subset,subsubset)
            i = subsubset[0]
            j = subsubset[1]
            copied_set = list(subset)
            copied_set.remove(i)
            copied_set.remove(j)
            n = copied_set[0]
            m = copied_set[1]

            blocked = sub_check(i,n,m)
            if not blocked:
                continue
            blocked = sub_check(j,n,m)
            if not blocked:
                continue
            blocked = sub_check(n,i,j)
            if not blocked:
                continue
            blocked = sub_check(m,i,j)
            if not blocked:
                continue

            # print(subset)
            return True

    return False

def isBasicTemp3_2(treeSet):
    # lvs = getLeaves(treeSet[0])
    label = getLabelTreeSet(treeSet)
    nodes = [node for node in label]
    nodes.remove(0)

    for subset in combinations(nodes, 3):
        block = 0

        if not all([label[set1].isdisjoint(label[set2]) for set1 in subset for set2 in subset if set1 != set2]):
            continue

        for i in range(3):
            j = (i+1) % 3
            k = (i+2) % 3

            blocked = False
            for tree in treeSet:
                if not treeSet[tree].has_node(subset[i]):
                    continue
                p = list(treeSet[tree].predecessors(subset[i]))[0]
                children = list(treeSet[tree].successors(p))
                children.remove(subset[i])
                child = children[0]
                if label[subset[j]].issubset(label[child]) and label[subset[k]].issubset(label[child]):
                    blocked = True
                    break
            if not blocked:
                break
            block += 1
        if block == 3:
            # print(subset)
            print(3, [label[item] for item in subset])
            return True

    return False


def isBasicTemp4_2(treeSet):
    # lvs = getLeaves(treeSet[0])
    label = getLabelTreeSet(treeSet)
    nodes = [node for node in label]
    nodes.remove(0)
    def sub_check(x, a, b):
        for tree in treeSet:
            if not treeSet[tree].has_node(x):
                continue
            p = list(treeSet[tree].predecessors(x))[0]
            children = list(treeSet[tree].successors(p))
            children.remove(x)
            child = children[0]
            # print(a)
            # print(label[child])
            if label[a].issubset(label[child]) and label[b].issubset(label[child]):
                return True
                break
        return False

    for subset in combinations(nodes, 4):
        if not all([label[set1].isdisjoint(label[set2]) for set1 in subset for set2 in subset if set1 != set2]):
            continue
        for subsubset in combinations(subset, 2):
            # print(subset, subsubset)
            i = subsubset[0]
            j = subsubset[1]
            copied_set = list(subset)
            copied_set.remove(i)
            copied_set.remove(j)
            n = copied_set[0]
            m = copied_set[1]

            blocked = sub_check(i, n, m)
            if not blocked:
                continue
            blocked = sub_check(j, n, m)
            if not blocked:
                continue
            blocked = sub_check(n, i, j)
            if not blocked:
                continue
            blocked = sub_check(m, i, j)
            if not blocked:
                continue

            print(4,[label[item] for item in subset],[label[item] for item in subsubset])
            return True

    return False

def isLastTemp3(treeSet):

    # label = reLabelTreeSet(treeSet)
    label = getLabelTreeSet(treeSet)
    lvs = getLeaves(treeSet[0])
    root = getRoot(treeSet[0])
    three_label = []
    child_label = []
    for tree in treeSet:
        three_label.append([])
    # print("hoi")
    for tree in treeSet:
        for child in treeSet[tree].successors(root):
            child_label.append(label[child])
            if len(label[child]) == 1:
                three_label[tree].append(label[child])
            else:
                for child2 in treeSet[tree].successors(child):
                    three_label[tree].append(label[child2])


    for subset in combinations(lvs, 3):
        next = False
        for subsubset in combinations(subset,2):
            for tree in treeSet:
                for item in three_label[tree]:
                    if subsubset[0] in item and subsubset[1] in item:
                        next = True
                        break
                if next:
                    break
            if next:
                break
        if next:
            continue
        print(subset)
        for item in range(3):
            can_pick = False
            i = subset[item]
            j = subset[(item + 1) % 3]
            k = subset[(item + 2) % 3]
            # print(i,j,k)
            if any([ (i in child_l and not j in child_l and not k in child_l) for child_l in child_label ]):
                continue
            else:
                can_pick = True
                break
        if can_pick:
            return False
    return True
        # for tree in treeSet:

def isLastTemp3_2(treeSet):
    possible = list()
    label = getLabelTreeSet(treeSet)
    root = getRoot(treeSet[0])
    children = list(treeSet[0].successors(root))
    for i in range(2):
        child1 = children[i]
        child2 = children[(i+1)%2]
        if len(label[child1]) == 1:
            continue
        gchildren = list(treeSet[0].successors(child1))
        for leaf1 in label[gchildren[0]]:
            for leaf2 in label[gchildren[1]]:
                for leaf3 in label[child2]:
                    possible.append([leaf1,leaf2,leaf3])

    label = getLabelTreeSet(treeSet)
    lvs = getLeaves(treeSet[0])
    root = getRoot(treeSet[0])
    three_label = []
    child_label = []
    for tree in treeSet:
        three_label.append([])
    # print("hoi")
    for tree in treeSet:
        for child in treeSet[tree].successors(root):
            child_label.append(label[child])
            if len(label[child]) == 1:
                three_label[tree].append(label[child])
            else:
                for child2 in treeSet[tree].successors(child):
                    three_label[tree].append(label[child2])


    for subset in possible:
        next = False
        for subsubset in combinations(subset,2):
            for tree in treeSet:
                for item in three_label[tree]:
                    if subsubset[0] in item and subsubset[1] in item:
                        next = True
                        break
                if next:
                    break
            if next:
                break
        if next:
            continue
        print(subset)
        for item in range(3):
            can_pick = False
            i = subset[item]
            j = subset[(item + 1) % 3]
            k = subset[(item + 2) % 3]
            # print(i,j,k)
            if any([ (i in child_l and not j in child_l and not k in child_l) for child_l in child_label ]):
                continue
            else:
                can_pick = True
                break
        if can_pick:
            return False
    return True

    return True
        # for tree in treeSet:


def getNeighboors(treeSet,leaf):
    neigh = set()
    lvs = getLeaves(treeSet[0])
    for tree in treeSet:
        p = list(treeSet[tree].predecessors(leaf))[0]
        children = treeSet[tree].successors(p)
        for child in children:
            if child not in lvs:
                raise Exception("Sorry, no a leaf")
            if child != leaf:
                neigh.add(child)
    return neigh

# treeset = genTreeSetByEdge(10,2)
# reLabelTreeSet(treeset)
# while not getBasicTemp4(treeset):
#     print(getBasicTemp4(treeset))
#     treeset = genTreeSetByEdge(10, 2)
#     reLabelTreeSet(treeset)
# print(getBasicTemp4(treeset))
# showBetterTreeSet(treeset)