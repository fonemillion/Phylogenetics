from mpi4py import MPI
import copy
import os
import pickle
from Utils.InstanceGenerators.BiNetGenTemp import net_to_tree2, simulation_temp
from DataGen.pickClass import *
from Utils.InstanceGenerators.TreeEdgeGen import genTreeSetByEdgeTemp
from Utils.Solvers.TreeAn import removeTrivialLeaves
from Utils.Solvers.GreedyPicker import greedyPick3
import time
import sys


comm = MPI.COMM_WORLD
nprocs = comm.Get_size()
myrank = comm.Get_rank()

# gen = 'net'
# gen = 'edge'
R = None
gen = str(sys.argv[1])
L = int(sys.argv[2])
T = int(sys.argv[3])
if gen == 'net':
    R = int(sys.argv[4])

print(sys.argv, flush=True)
print(gen,L,T,R, flush=True)

i = 0
j = 0


dirname = None
if gen == 'net':
    dirname = "ret/" + str(L) +"L_"+ str(R) +"R_"+ str(T) +"T"
if gen == 'edge':
    dirname = "ret/" + str(L) +"L_" + str(T) + "T"


if not os.path.exists(dirname):
    try:
        os.makedirs(dirname)
    except:
        pass


while i < 3000:

    treeSet = None
    if gen == 'net':
        net, _ = simulation_temp(L, R)
        treeSet = net_to_tree2(net,T)
    if gen == 'edge':
        treeSet = genTreeSetByEdgeTemp(L, T)


    reLabelTreeSet(treeSet)
    # removeTrivialLeaves(treeSet)
    reLabelTreeSet(treeSet)
    if len(treeSet[0]) < 2:
        continue

    options = canPick(treeSet)
    if len(options) > 1:
        retValues = []
        start = time.time()
        for leaf in options:
            subTreeSet = copy.deepcopy(treeSet)
            sol1 = weightLeaf(treeSet, leaf)
            pickCher(subTreeSet, leaf)


            sol = greedyPick3(subTreeSet)
            end = time.time()
            if len(sol) == 0:
                sol1 = -1
            else:
                sol1 += sol[0][0]
            retValues.append([leaf,sol1])
            del sol
        retV = {r[1] for r in retValues}


        filename = dirname + "/inst_" + str(i) + ".pickle"
        while os.path.exists(filename):
            i += 1
            filename = dirname + "/inst_" + str(i) + ".pickle"
            continue

        try:
            with open(filename, 'wb') as f:
                pickle.dump([treeSet,retValues], f)
        except:
            pass
        print(i,R,retV,len(retValues),end - start, flush=True)
        i += 1
        continue


    else:
        print("Only 1 pickable ")
        pickCher(treeSet,options[0])

