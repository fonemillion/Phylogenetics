import pickle
from DataGen.pickClass import getLeaves
import matplotlib.pyplot as plt

i = 0
ret = []
nrret = []
nontri = []
nrleaves = []
nrchoise = []

def count_elements(seq) -> dict:
    hist = {}
    for i in seq:
        hist[i] = hist.get(i, 0) + 1
    return hist


while i < 5000:
    if i % 500 == 0:
        print(i)
    filename = "../Data/ret/20L_5R_2T/inst_" + str(i) + ".pickle"

    with open(filename, 'rb') as f:
        [treeSet, retValues] = pickle.load(f)
    ret.append(min([i[1] for i in retValues if i[1] != -1 ]))
    nrret.append(len({i[1] for i in retValues }))
    nontri.append(min([i[1] for i in retValues]) == -1)
    nrleaves.append(len(getLeaves(treeSet[0])))
    nrchoise.append(len(retValues))
    i += 1

print(count_elements(ret))
print(count_elements(nrret))
print(count_elements(nontri))
print(count_elements(nrleaves))
print(count_elements(nrchoise))