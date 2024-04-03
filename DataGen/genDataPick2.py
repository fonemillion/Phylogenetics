import copy
import os
import pickle
from DataGen.pickClass import *
from Utils.InstanceGenerators.TreeEdgeGen import genTreeSetByEdgeTemp
from Utils.Solvers.TreeAn import removeTrivialLeaves
from Utils.Solvers.GreedyPicker import greedyPick3
import time



i = 0
j = 0
while i < 20000:

    # R = 15
    L = 20
    T = 2
    dirname =  "../../Data/ret/" + str(L) +"L_" + str(T) + "T"
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    filename = dirname + "/inst_" + str(i) + ".pickle"

    if os.path.exists(filename):
        print(f"The file {filename} already exists.")
        i += 1
        continue

    treeSet = genTreeSetByEdgeTemp(L, T)
    # net, _ = simulation_temp(L, R)
    # treeSet = net_to_tree2(net,T)

    # if len(tempSolver(treeSet)) == 0:
    #    print("non-temporal")
    #    break

    removeTrivialLeaves(treeSet)
    reLabelTreeSet(treeSet)
    if len(treeSet[0]) < 2:
        continue

    options = canPick(treeSet)
    if len(options) > 1:
        retValues = []
        for leaf in options:
            subTreeSet = copy.deepcopy(treeSet)
            sol1 = weightLeaf(treeSet, leaf)
            pickCher(subTreeSet, leaf)

            start = time.time()
            sol = greedyPick3(subTreeSet)
            end = time.time()
            print(end - start)
            # time1 = (end - start)
            # start = time.time()
            # sol = greedyPick(subTreeSet)
            # end = time.time()
            # print(time1,end - start)
            #print(sol)
            if len(sol) == 0:
                sol1 = -1
            else:
                sol1 += sol[0][0]
            retValues.append([leaf,sol1])
            del sol
        retV = {r[1] for r in retValues}
        #if len(retV) == 1:
        #    print('only good solutions')
        #    continue
        # if min({r for r in retV if r != -1}) > R:
        #     showGraph(net,False)
        #     showTreeSet(treeSet)
        with open(filename, 'wb') as f:
            pickle.dump([treeSet,retValues], f)
        # print(i,R,retV,len(retValues))
        print(i,retV,len(retValues), flush=True)
        i += 1
        continue


    else:
        print("Only 1 pickable ")
        pickCher(treeSet,options[0])

