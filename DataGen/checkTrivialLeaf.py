import os
import pickle

from DataGen.pickClass import pickCher
from Utils.TreeNetUtils import showBetterTreeSet


def count_elements(seq) -> dict:
    hist = {}
    for i in seq:
        hist[i] = hist.get(i, 0) + 1
    return hist

i = 0

nTL = []
while True:
    if i % 5000 == 0:
        print(i)
    T = 2
    L = 30
    filename = "../Data/trivialLeaf/" + str(L) + "L_" + str(T) + "T/inst_" + str(i) + ".pickle"
    i+=1
    if not os.path.isfile(filename):
        break
    with open(filename, 'rb') as f:
        [treeSet, tlset] = pickle.load(f)

    nTL.append(len(tlset))

print(count_elements(nTL))