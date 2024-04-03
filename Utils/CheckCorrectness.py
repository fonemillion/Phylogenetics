from Utils.TreeNetUtils import getLeaves, getRoot


def CheckCorrFlowUp(net, nodeDict, EdgeDict):
    nrNodes = len(net.nodes())
    nrEdges = len(net.edges())
    lvs = getLeaves(net)
    root = getRoot(net)

    if len(nodeDict) != nrNodes:
        raise Exception("Flow is not correct")
    if len(EdgeDict) != nrEdges:
        raise Exception("Flow is not correct")

    for item in EdgeDict:
        if EdgeDict[item] == 0:
            raise Exception("Flow is not correct")
        if nodeDict[item[1]] != EdgeDict[item]:
            raise Exception("Flow is not correct")
        if nodeDict[item[0]] >= EdgeDict[item]:
            raise Exception("Flow is not correct")

    for item in nodeDict:
        if nodeDict[item] == 0 and item not in lvs:
            raise Exception("Flow is not correct")


def CheckCorrFlowDown(net, nodeDict, EdgeDict):
    nrNodes = len(net.nodes())
    nrEdges = len(net.edges())
    lvs = getLeaves(net)
    root = getRoot(net)

    if len(nodeDict) != nrNodes:
        raise Exception("Flow is not correct")
    if len(EdgeDict) != nrEdges:
        raise Exception("Flow is not correct")

    for item in EdgeDict:
        if EdgeDict[item] == 0:
            raise Exception("Flow is not correct")
        if nodeDict[item[1]] != EdgeDict[item]:
            raise Exception("Flow is not correct")
        if nodeDict[item[0]] >= EdgeDict[item]:
            raise Exception("Flow is not correct")

    for item in nodeDict:
        if nodeDict[item] == 0 and item != root:
            raise Exception("Flow is not correct")