import copy
import os
import pickle
import random

import winsound
from Utils.CherryDepend import getBasicTrivialLeaf, getBasicTemp3, getBasicTemp4, isBasicTemp4_2, isBasicTemp3_2
from Utils.InstanceGenerators.TreeEdgeGen import genTreeSetByEdge,genTreeSetByEdgeTemp
from Utils.InstanceGenerators.BiNetGenTemp import net_to_tree2, simulation_temp
from DataGen.pickClass import *
from Utils.Solvers.TempSolver import tempSolver
from Utils.Solvers.TreeAn import removeTrivialLeaves
from Utils.TreeNetUtils import showTreeSet, showGraph, showBetterTreeSet
from Utils.Solvers.GreedyPicker import greedyPick3, greedyPick




L = 3
T = 3

i = 0
while True:
    if i % 500 == 0:
        print(i)
    i += 1



    treeset = genTreeSetByEdge(L, T)
    reLabelTreeSet(treeset)

    Temp3 = isBasicTemp3_2(treeset)

    Temp4 = isBasicTemp4_2(treeset)
    # if Temp3:
    #     showBetterTreeSet(treeset)
    sol = tempSolver(treeset)


    Temp = (sol == [])

    print(Temp,Temp3,Temp4)
    if Temp and not (Temp3 or Temp4):
        showBetterTreeSet(treeset)
    if (Temp3 or Temp4) and not Temp:
        showBetterTreeSet(treeset)



