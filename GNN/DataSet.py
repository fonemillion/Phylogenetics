import copy
import pickle
import os
import shutil
from itertools import combinations

import networkx
import networkx as nx
from torch_geometric.data import Data
import torch
from torch_geometric.data import InMemoryDataset, download_url


from Utils.TreeNetUtils import getRoots, getLeaves, getLabelTreeSet, canPick, getRoot



def GetDepth(dir, N):
    maxDepth = 0
    for i in range(N):
        filename = dir + '/inst_' + str(i) + '.pickle'

        with open(filename, 'rb') as f:
            [treeSet, retValues] = pickle.load(f)

        # get nodeDict
        nodeDict = getLabelTreeSet(treeSet)
        if nodeDict == None:
            raise Exception('error treeSet not correctly labbeled')

        net = nx.DiGraph()
        for i in treeSet:
            net.add_nodes_from(treeSet[i].nodes())
            net.add_edges_from(treeSet[i].edges())
        NUp, EUp = getFlowUp(net)
        NDown, NUP = getFlowDown(net)
        max_up = max(NUp.values())
        max_down = max(NDown.values())
        maxDepth = max(maxDepth,max_up)
        maxDepth = max(maxDepth,max_down)
    print(maxDepth,flush=True)
    return maxDepth


class MyOwnDataset(InMemoryDataset):
    def __init__(self, transform=None, pre_transform=None, pre_filter=None, N = 200, train_dir = 'Temp4/10L_2T',  delete = False, extraFeat = True, isClique = False):
        dir = train_dir.replace('/', '_')
        dir = "Dataset/" + dir + str(N) + 'N'
        if isClique:
            dir += '_Cq'
        if extraFeat:
            dir += '_EF'
        self.train_dir = 'data/' + train_dir
        if delete == True:
            if os.path.exists(dir):
                shutil.rmtree(dir)
                print(f"{dir} deleted successfully.")
            else:
                print(f"{dir} does not exist.")


        self.N = N
        self.extraFeat = extraFeat
        self.isClique = isClique
        super().__init__(dir, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])


    @property
    def processed_file_names(self):
        return ['data.pt']

#    def download(self):
#        # Download to `self.raw_dir`.
#        #download_url(url, self.raw_dir)
#        ...

    def process(self):
        # Read data into huge `Data` list.
        data_list = []

        if not self.isClique:
            Ndepth = GetDepth(self.train_dir,self.N)

        for i in range(self.N):
            if i % 100 == 0:
                print(i)
            filename = self.train_dir + '/inst_' + str(i) + '.pickle'
            if not os.path.isfile(filename):
                raise Exception("no " + filename + " found")
                break
            if self.isClique:
                data = toDataCq(filename,self.extraFeat)
            else:
                data = toData(filename, self.extraFeat, Ndepth)

            if data == None:
                raise Exception("no Data")
            data_list.append(data)


        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]
        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


def toData(filename,extraFeat, Ndepth):
    # get Data
    with open(filename, 'rb') as f:
        [treeSet, retValues] = pickle.load(f)

    # get nodeDict
    nodeDict = getLabelTreeSet(treeSet)
    if nodeDict == None:
        raise Exception('error treeSet not correctly labbeled')

    # Define which to pick in bestPick
    nonTemp = []
    for leaf in retValues:
        if leaf[1] == -1:
            nonTemp.append(leaf[0])
    retValues = [i for i in retValues if i[0] not in nonTemp]
    bestW = min([i[1] for i in retValues])
    bestPick = [i[0] for i in retValues if i[1] == bestW]


    net = nx.DiGraph()
    for i in treeSet:
        net.add_nodes_from(treeSet[i].nodes())
        net.add_edges_from(treeSet[i].edges())

    lvs = getLeaves(net)
    roots = getRoots(net)
    # showBetterTreeSet(treeSet)
    # showGraph(net)

    NUp, EUp = getFlowUp(net)
    NDown, EDown = getFlowDown(net)




    pickAble = canPick(treeSet)
    isLeaf = []
    dataPAble = []
    y = []
    x = []
    nodeUp = []
    nodeDown = []
    for node in range(len(net.nodes())):
        # give values to y if pickable
        if node in bestPick:
            y.append(1)
        elif node in pickAble:
            y.append(0)

        if node in pickAble:
            dataPAble.append(1)
        else:
            dataPAble.append(0)

        inTree = 0
        for tree in treeSet:
            if node in treeSet[tree].nodes():
                inTree += 1

        if extraFeat:
            minL, maxL = findMaxMintoRoot(treeSet, node)
            if node in roots:
                nrTriv = 0
            else:
                copyTreeSet = copy.deepcopy(treeSet)
                for leaf in nodeDict[node]:
                    pickLeaf(copyTreeSet, leaf)

                nrTriv = 0
                while True:
                    ripe = ripeCherry(copyTreeSet)
                    if len(ripe) > 0:
                        pickLeaf(copyTreeSet, ripe[0][0])
                        nrTriv += 1
                        continue
                    if len(getLeaves(copyTreeSet[0])) == 1:
                        nrTriv += 1
                    break

            x.append([net.in_degree(node), net.out_degree(node), inTree, int(node in lvs), int(node in roots),
                      int(node in pickAble), len(nodeDict[node]), minL, maxL, nrTriv])
        else:
            x.append([int(node in lvs), int(node in pickAble), net.out_degree(node), net.in_degree(node)])


        nodeUp.append(NUp[node])
        nodeDown.append(NDown[node])

        if node in lvs:
            isLeaf.append(True)
            # y.append(node in result)  #
        else:
            isLeaf.append(False)


    CaseDepthUp = max(nodeUp)
    CaseDepthDown = max(nodeDown)


    edge_index_up = []
    edge_index_down = []
    edgeUp = []
    edgeDown = []
    for edge in net.edges:
        edge_index_down.append((edge[0], edge[1]))
        edge_index_up.append((edge[1], edge[0]))
        edgeDown.append(EDown[edge])
        edgeUp.append(EUp[(edge[1], edge[0])])

    data = Data()


    data.x = torch.tensor(x, dtype=torch.float)
    data.edge_index_up = torch.tensor(edge_index_up).t().contiguous()
    data.edge_index_down = torch.tensor(edge_index_down).t().contiguous()
    data.y = torch.tensor(y, dtype=torch.long)


    data.DepthUp = torch.tensor(CaseDepthUp, dtype=torch.long)
    data.DepthDown = torch.tensor(CaseDepthDown, dtype=torch.long)

    data_edge_attr_edgeUp = torch.tensor(edgeUp, dtype=torch.long)
    data_edge_attr_edgeDown = torch.tensor(edgeDown, dtype=torch.long)
    data_nodeUp = torch.tensor(nodeUp, dtype=torch.long)
    data_nodeDown = torch.tensor(nodeDown, dtype=torch.long)

    data.isLeaf = torch.tensor(isLeaf, dtype=torch.bool)
    data.pickAble = torch.tensor(dataPAble, dtype=torch.bool)


    DepthDown = Ndepth+1
    DepthUp = Ndepth+1
    if data.DepthUp > DepthUp:
        raise Exception("Depth to low")
    if data.DepthDown > DepthDown:
        raise Exception("Depth to low")

    data.edge_index_flowEdgeUp = []
    for i in range(DepthUp):
        data.edge_index_flowEdgeUp.append(data.edge_index_up[:,data_edge_attr_edgeUp == i])
    data.edge_index_flowEdgeDown = []
    for i in range(DepthDown):
        data.edge_index_flowEdgeDown.append(data.edge_index_down[:,data_edge_attr_edgeDown == i])
    data.nodeMaskUp = []
    for i in range(DepthUp):
        data.nodeMaskUp.append(data_nodeUp == i)
    data.nodeMaskDown = []
    for i in range(DepthDown):
        data.nodeMaskDown.append(data_nodeDown == i)

    # print(data.x)
    # showBetterTreeSet(treeSet)

    del net
    return data

def toDataCq(filename,extraFeat):
    # get Data
    with open(filename, 'rb') as f:
        [treeSet, retValues] = pickle.load(f)

    # get nodeDict
    nodeDict = getLabelTreeSet(treeSet)
    if nodeDict == None:
        raise Exception('error treeSet not correctly labbeled')

    # Define which to pick in bestPick
    nonTemp = []
    for leaf in retValues:
        if leaf[1] == -1:
            nonTemp.append(leaf[0])
    retValues = [i for i in retValues if i[0] not in nonTemp]
    bestW = min([i[1] for i in retValues])
    bestPick = [i[0] for i in retValues if i[1] == bestW]

    lvs = getLeaves(treeSet[0])

    pickAble = canPick(treeSet)

    y = []
    x = []
    dataPAble = []

    for item in range(len(lvs)):
        node = item + 1

        if extraFeat:
            features = getLeafFeat(treeSet, node, nodeDict)
            features.append(int(node in pickAble))
            x.append(features)
        else:
            x.append([int(node in pickAble)])

        if node in bestPick:
            y.append(1)
        elif node in pickAble:
            y.append(0)

        if node in pickAble:
            dataPAble.append(1)
        else:
            dataPAble.append(0)



    edge_index = []
    edge_attr = []
    for item in [(i, j) for i in lvs for j in lvs if i != j]:
        edge_index.append((item[0] - 1, item[1] - 1))
        if extraFeat:
            edge_attr.append(getEdgeFeat2(nodeDict, treeSet, item))
        else:
            edge_attr.append(getEdgeFeat(nodeDict, treeSet, item))



    data = Data()
    data.x = torch.tensor(x, dtype=torch.float)
    data.edge_index = torch.tensor(edge_index).t().contiguous()
    data.y = torch.tensor(y, dtype=torch.long)
    data.edge_attr = torch.tensor(edge_attr, dtype=torch.float)
    data.pickAble = torch.tensor(dataPAble, dtype=torch.bool)

    return data

def getFlowUp(net):


    def getFlow(net,node,nodeDict,EdgeDict):
        # check if leaf
        if net.out_degree(node) == 0:
            nodeDict[node] = 0
            return
        # solve for children
        maxV = 0
        for child in net.successors(node):
            if child not in nodeDict:
                getFlow(net,child,nodeDict,EdgeDict)
            maxV = max(maxV, nodeDict[child])
        for child in net.successors(node):
            EdgeDict[(child,node)] = maxV + 1
        nodeDict[node] = maxV + 1
        return

    nodeDict = dict()
    EdgeDict = dict()

    for root in getRoots(net):
        getFlow(net, root, nodeDict, EdgeDict)
    return nodeDict, EdgeDict

def getFlowDown(net):

    def getFlow(net,node,nodeDict,EdgeDict):
        # check if leaf
        if net.in_degree(node) == 0:
            nodeDict[node] = 0
            return
        # solve for parents
        maxV = 0
        for p in net.predecessors(node):
            if p not in nodeDict:
                getFlow(net,p,nodeDict,EdgeDict)
            maxV = max(maxV, nodeDict[p])
        for p in net.predecessors(node):
            EdgeDict[(p,node)] = maxV + 1
        nodeDict[node] = maxV + 1
        return

    nodeDict = dict()
    EdgeDict = dict()

    for leaf in getLeaves(net):
        getFlow(net, leaf, nodeDict, EdgeDict)

    return nodeDict, EdgeDict

def getBasicTemp3(treeSet):
    lvs = getLeaves(treeSet[0])
    label = getLabelTreeSet(treeSet)
    templeafset = set()
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
            for i in range(3):
                templeafset.add(subset[i])

    return templeafset

def findLengthRoot(tree, leaf):
    root = getRoot(tree)
    lvs = getLeaves(tree)
    if leaf == root:
        return 0
    if leaf not in tree:
        raise Exception( leaf, " not in network")
    p = list(tree.predecessors(leaf))[0]
    length = 1
    while p != root:
        p = list(tree.predecessors(p))[0]
        length += 1
    return length

def findMaxMintoRoot(treeSet,node):
    lvs = getLeaves(treeSet[0])
    minL = len(lvs)
    maxL = 0
    for tree in treeSet:
        if node in treeSet[tree]:
            length = findLengthRoot(treeSet[tree], node)
            minL = min(minL,length)
            maxL = max(maxL,length)
    return minL,maxL

def pickLeaf(Tset,leaf):
    """
    Pick leaf
    :param Tset:
    :param leaf:
    :return:
    """
    for i in Tset:

        p = list(Tset[i].predecessors(leaf))[0]
        if len(list(Tset[i].predecessors(p))) == 0:
            # no grandparent
            Tset[i].remove_node(leaf)
            Tset[i].remove_node(p)
        else:
            gp = list(Tset[i].predecessors(p))[0]
            chldren = list(Tset[i].successors(p))
            chldren.remove(leaf)
            Tset[i].remove_node(leaf)
            Tset[i].remove_node(p)
            Tset[i].add_edge(gp,chldren[0])

def ripeCherry(TSet):
    lvs = getLeaves(TSet[0])
    setall = [(l1, l2) for l1 in lvs for l2 in lvs if l1 != l2]
    set = [(l1, l2) for l1 in lvs for l2 in lvs if l1 != l2]
    for cher in setall: # doet dubbel voor (x,y) en (y,x)
        for i in TSet:
            if not isCher(TSet[i],lvs,cher):
                set.remove(cher)
                break
    return set

def isCher(tree,lSet,cher):
    node1 = cher[0]
    node2 = cher[1]
    if node1 not in lSet:
        return False
    if node2 not in lSet:
        return False
    par1 = list(tree.predecessors(node1))[0]
    par2 = list(tree.predecessors(node2))[0]
    return par1 == par2

def getEdgeFeat(nodeDict, treeSet, item):
    # "is cherry" "blocks" "blocked by"
    isCherr = False
    xyBlock = False
    yxBlock = False
    for tree in treeSet:

        p0 = list(treeSet[tree].predecessors(item[0]))[0]
        p1 = list(treeSet[tree].predecessors(item[1]))[0]
        if not isCherr:
            isCherr = (p0 == p1)

        desc = nodeDict[p0].copy()
        desc.remove(item[0])
        if item[1] in desc and len(desc) > 1:
            xyBlock = True

        desc = nodeDict[p1].copy()
        desc.remove(item[1])
        if item[0] in desc and len(desc) > 1:
            yxBlock = True

    return [isCherr, xyBlock, yxBlock]

def getEdgeFeat2(nodeDict,treeSet,pair):
    lvs = getLeaves(treeSet[0])
    root = getRoot(treeSet[0])
    max_nr_pick_before_leaf = 0
    min_nr_pick_before_leaf = len(lvs)

    max_length_same_par_l1 = 0
    min_length_same_par_l1 = len(lvs)

    max_length_same_par_l2 = 0
    min_length_same_par_l2 = len(lvs)
    isCherry = 0
    blocked = 0
    for tree in treeSet:
        sameP = []
        for node in treeSet[tree]:
            if pair[0] in nodeDict[node] and pair[1] in nodeDict[node]:
                sameP.append(node)
        sameP = sameP[argmin([len(nodeDict[n]) for n in sameP])]
        #print(sameP)
        nr_pick_before_leaf = len(nodeDict[sameP])-2
        length_same_par_l1 = findLengthTo(treeSet[tree],pair[0],sameP)
        length_same_par_l2 = findLengthTo(treeSet[tree],pair[1],sameP)

        max_nr_pick_before_leaf = max(max_nr_pick_before_leaf,nr_pick_before_leaf)
        min_nr_pick_before_leaf = min(min_nr_pick_before_leaf,nr_pick_before_leaf)

        max_length_same_par_l1 = max(max_length_same_par_l1,length_same_par_l1)
        min_length_same_par_l1 = min(min_length_same_par_l1,length_same_par_l1)

        max_length_same_par_l2 = max(max_length_same_par_l2,length_same_par_l2)
        min_length_same_par_l2 = min(min_length_same_par_l2,length_same_par_l2)
        if list(treeSet[tree].predecessors(pair[0])) == list(treeSet[tree].predecessors(pair[1])):
            isCherry = 1
        childRoot = list(treeSet[tree].successors(root))
        if (pair[0] in nodeDict[childRoot[0]] and pair[1] in nodeDict[childRoot[1]]) or (pair[1] in nodeDict[childRoot[0]] and pair[0] in nodeDict[childRoot[1]]):
            blocked += 1
    #print(nodeDict)
    distPPmin = len(lvs) **2
    distPPmax = 0
    cycDistMax = 0
    cycDistMin = len(lvs) ** 2

    loop = [(i, j) for i in treeSet for j in treeSet if i != j and i < j]
    for item in loop:
        i = item[0]
        j = item[1]
        samePi = []
        for node in treeSet[i]:
            if pair[0] in nodeDict[node] and pair[1] in nodeDict[node]:
                samePi.append(node)
        samePi = samePi[argmin([len(nodeDict[n]) for n in samePi])]

        samePj = []
        for node in treeSet[j]:
            if pair[0] in nodeDict[node] and pair[1] in nodeDict[node]:
                samePj.append(node)
        samePj = samePj[argmin([len(nodeDict[n]) for n in samePj])]
        #print(samePi,samePj)
        pathi0 = [pair[0]]
        pathi1 = [pair[1]]
        pathj0 = [pair[0]]
        pathj1 = [pair[1]]
        p = pair[0]
        while True:
            p = list(treeSet[i].predecessors(p))[0]
            pathi0.append(p)
            if p == samePi:
                break
        p = pair[1]
        while True:
            p = list(treeSet[i].predecessors(p))[0]
            if p == samePi:
                break
            pathi1.append(p)
        p = pair[0]
        while True:
            p = list(treeSet[j].predecessors(p))[0]
            pathj0.append(p)
            if p == samePj:
                break
        p = pair[1]
        while True:
            p = list(treeSet[j].predecessors(p))[0]
            if p == samePj:
                break
            pathj1.append(p)
        pathi1.reverse()
        pathj1.reverse()
        pathi = pathi0 + pathi1
        pathj = pathj0 + pathj1
        pathi = [nodeDict[i] for i in pathi]
        pathj = [nodeDict[i] for i in pathj]
        distPP = len(set(nodeDict[samePi]) ^ set(nodeDict[samePj]))
        distPPmin = min(distPPmin,distPP)
        distPPmax = max(distPPmax,distPP)

    isCherr = False
    xyBlock = False
    yxBlock = False
    item = pair
    for tree in treeSet:

        p0 = list(treeSet[tree].predecessors(item[0]))[0]
        p1 = list(treeSet[tree].predecessors(item[1]))[0]
        if not isCherr:
            isCherr = (p0 == p1)

        desc = nodeDict[p0].copy()
        desc.remove(item[0])
        if item[1] in desc and len(desc) > 1:
            xyBlock = True

        desc = nodeDict[p1].copy()
        desc.remove(item[1])
        if item[0] in desc and len(desc) > 1:
            yxBlock = True

    return [max_nr_pick_before_leaf, min_nr_pick_before_leaf, max_length_same_par_l1, min_length_same_par_l1, max_length_same_par_l2, min_length_same_par_l2, isCherry,blocked,distPPmin,distPPmax,xyBlock,yxBlock]

def argmin(lst):
  return lst.index(min(lst))

def findLengthRoot(tree, leaf):
    root = getRoot(tree)
    lvs = getLeaves(tree)
    if leaf == root:
        return 0
    if leaf not in tree:
        raise Exception( leaf, " not in network")
    p = list(tree.predecessors(leaf))[0]
    length = 1
    while p != root:
        p = list(tree.predecessors(p))[0]
        length += 1
    return length

def findLengthTo(tree, leaf, node):
    lvs = getLeaves(tree)
    if leaf not in lvs:
        raise Exception( leaf, " not in network")
    p = list(tree.predecessors(leaf))[0]
    length = 1
    while p != node:
        p = list(tree.predecessors(p))[0]
        length += 1
    return length

def getLeafFeat(treeSet,node,nodeDict):
    lvs = getLeaves(treeSet[0])
    min_to_root = len(lvs) + 2
    max_to_root = 0
    for tree in treeSet:
        length = findLengthRoot(treeSet[tree], node)
        min_to_root = min(min_to_root, length)
        max_to_root = max(max_to_root, length)
    neighbors = set()
    for tree in treeSet:
        p = list(treeSet[tree].predecessors(node))[0]
        children = list(treeSet[tree].successors(p))
        children.remove(node)
        child = children[0]
        neighbors.add(child)
    weight = len(neighbors)


    max_nr_pick_before_leaf = 0
    min_nr_pick_before_leaf = len(lvs)
    for tree in treeSet:
        nr_pick_before_leaf = 0
        root = getRoot(treeSet[tree])
        for child in list(treeSet[tree].successors(root)):
            if node in nodeDict[child]:
                nr_pick_before_leaf = len(nodeDict[child])-1

        max_nr_pick_before_leaf = max(max_nr_pick_before_leaf,nr_pick_before_leaf)
        min_nr_pick_before_leaf = min(min_nr_pick_before_leaf,nr_pick_before_leaf)

    return [min_to_root,max_to_root,max_nr_pick_before_leaf,min_nr_pick_before_leaf,weight]

def getPred(treeSet,leaf):
    predList = dict()
    for i in treeSet:
        toDo = [leaf]
        predListB = [leaf]
        while len(toDo) > 0:
            item = toDo.pop(0)
            for pred in treeSet[i].predecessors(item):
                if pred not in predListB:
                    predListB.append(pred)
                    toDo.append(pred)
        predList[i] = predListB
    return predList

