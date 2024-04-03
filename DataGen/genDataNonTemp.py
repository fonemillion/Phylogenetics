import copy
import os
import pickle

from Utils.InstanceGenerators.BiNetGen import genNetTC, net_to_tree2
from Utils.InstanceGenerators.TreeEdgeGen import genTreeSetByEdge
from Utils.Solvers.TempSolver_23 import TempSolver_comb
from Utils.Solvers.TreeAn import removeTrivialLeaves
import time

from Utils.TreeNetUtils import reLabelTreeSet




def genNonTemp(info):

    i = 0
    R, L, T, start_dir, gen = info

    dirname = None
    if gen == 'net':
        dirname = start_dir + "nonTemp/" + str(L) + "L_" + str(R) + "R_" + str(T) + "T"
    if gen == 'edge':
        dirname = start_dir + "nonTemp/" + str(L) + "L_" + str(T) + "T"


    # dirname_time = start_dir + "nonTemp/time2"
    if not os.path.exists(dirname):
        try:
            os.makedirs(dirname)
        except:
            pass
    # if not os.path.exists(dirname_time):
    #     os.makedirs(dirname_time)

    while i < 20000:

        treeSet = None
        if gen == 'net':
            net = genNetTC(L, R)
            treeSet = net_to_tree2(net, T)
        if gen == 'edge':
            treeSet = genTreeSetByEdge(L, T)


        removeTrivialLeaves(treeSet)
        reLabelTreeSet(treeSet)

        if len(treeSet[0]) < 2:
            continue

        subTreeSet1 = copy.deepcopy(treeSet)

        start1 = time.time()
        sol1 = TempSolver_comb(subTreeSet1)
        eind1 = time.time()


        sol1 = (len(sol1) != 0)
        print(i,sol1,eind1 - start1)

        # if sol1:
        #     continue
        filename = dirname + "/inst_" + str(i) + ".pickle"
        while os.path.exists(filename):
            i += 1
            filename = dirname + "/inst_" + str(i) + ".pickle"

        try:
            with open(filename, 'wb') as f:
                pickle.dump([treeSet, sol1], f)
        except:
            pass

        i += 1
        # print(i,sol1,eind1 - start1)

if __name__ == '__main__':
    R = 25
    L = 35
    T = 5
    gen = 'net'
    start_dir = '../../Data/'
    info = R, L, T, start_dir, gen
    genNonTemp(info)