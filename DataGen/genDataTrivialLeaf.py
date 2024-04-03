import copy
import os
import pickle
import random

import winsound
from Utils.CherryDepend import getBasicTrivialLeaf
from Utils.InstanceGenerators.TreeEdgeGen import genTreeSetByEdge,genTreeSetByEdgeTemp
from Utils.InstanceGenerators.BiNetGenTemp import net_to_tree2, simulation_temp
from DataGen.pickClass import *
from Utils.Solvers.TempSolver import tempSolver
from Utils.Solvers.TreeAn import removeTrivialLeaves
from Utils.TreeNetUtils import showTreeSet, showGraph, showBetterTreeSet
from Utils.Solvers.GreedyPicker import greedyPick3, greedyPick


import time
def count_elements(seq) -> dict:
    hist = {}
    for i in seq:
        hist[i] = hist.get(i, 0) + 1
    return hist


Tset = [   3,  3 ,4, 5]
Lset = [ 40, 50 ,30,30]

for item in range(len(Tset)):

    nTL = []
    i = 0
    balans = 0
    T = Tset[item]
    L = Lset[item]
    print(T,L)
    folder_path = "../../Data/trivialLeaf/" + str(L) + "L_" + str(T) + "T"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    while i < 5000:
        if i % 500 == 0:
            print(i)
        filename = "../../Data/trivialLeaf/" + str(L) + "L_" + str(T) + "T/inst_" + str(i) + ".pickle"


        if os.path.exists(filename):
            print(f"The file {filename} already exists.")
            i += 1
            continue




        while True:
            treeset = genTreeSetByEdgeTemp(L, T)
            reLabelTreeSet(treeset)
            tlset = getBasicTrivialLeaf(treeset)
            if len(tlset) != 0:
                break
            if balans > 0:
                break

        with open(filename, 'wb') as f:
            pickle.dump([treeset,tlset], f)
            if len(tlset) == 0:
                balans -= 1
            else:
                balans += 1
        #print(i,tlset,balans)
        i += 1
        nTL.append(len(tlset))

    print(count_elements(nTL))



    # Your code here...

    # Play a simple beep sound when the code is done
    winsound.Beep(1000, 1000)