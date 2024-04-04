# Temp solver for both


from Solvers_Temp.Grow import Case_T3
from Solvers_Temp.Pick import Case_T, TempSolver_2
from Utils.TreeAn import getLeaves








def TempSolver_comb(tree_set,isTree = True):
    # initialize 3
    main_case_3 = Case_T3(tree_set)
    todo_3 = []
    sol_3 = []

    for leaf in main_case_3.to_pick:
        new_case = main_case_3.copy()
        new_case.start(leaf)
        todo_3.append(new_case)



    # initialize 2
    if isTree:
        main_case = Case_T(tree_set, input_is_tree=True)
    else:
        main_case = tree_set
    sol_2 = []
    todo_2 = []
    todo_2.append(main_case)

    clusters_solved = []
    clust_ans = []

    done = []
    checked = []
    for _ in range(len(getLeaves(tree_set[0]))+2):
        done.append([])
        checked.append([])


    while True:
        if len(todo_3) == 0:
            return sol_3
        if len(todo_2) == 0:
            return sol_2
        # I+=1
        case_3 = todo_3.pop()

        if len(case_3.to_pick) == 0:
            sol_3 = case_3.sol
            return sol_3

        for leaf in case_3.to_pick:
            buddies = case_3.add(leaf)
            if buddies is not None:
                new_set = case_3.to_pick.copy()
                new_set.remove(leaf)
                if new_set in checked[len(new_set)]:
                    continue
                else:
                    checked[len(new_set)].append(new_set)

                new_case = case_3.copy()
                new_case.grow(leaf,buddies)
                todo_3.append(new_case)

        # solver 2 part
        case_2 = todo_2.pop()

        if len(case_2.leaves) <= 2:
            sol = case_2.sol
            sol.append(case_2.leaves[0])
            if len(case_2.leaves) > 1:
                sol.append(case_2.leaves[1])
            # print(I)
            return sol

        if len(case_2.clusters) > 0:
            max_clus = 0
            pick = 0
            for clus in case_2.clusters:
                if len(case_2.node_dict[clus]) == 2:
                    pick = min(case_2.node_dict[clus])
                    break
                length = len(case_2.node_dict[clus])
                if length > max_clus:
                    max_clus = length
                    pick = clus
            if len(case_2.node_dict[pick]) == 1:
                case_2.pick_leaf(pick)
            else:
                if case_2.node_dict[pick] in clusters_solved:
                    sol_partial = clust_ans[clusters_solved.index(case_2.node_dict[pick])]
                else:
                    sol_partial = TempSolver_2(case_2.get_cluster(pick), isTree = False)
                    clust_ans.append(sol_partial.copy())
                    clusters_solved.append(case_2.node_dict[pick].copy())
                if len(sol_partial) == 0:
                    continue
                else:
                    for i in range(len(sol_partial)-1):
                        case_2.pick_leaf(sol_partial[i])
                    todo_2.append(case_2)
                    continue

        can_pick = case_2.can_pick()
        for leaf in can_pick:
            new_set = case_2.leaves.copy()
            new_set.remove(leaf)
            new_set.sort()
            if new_set in done[len(new_set)]:
                continue
            else:
                done[len(new_set)].append(new_set)
            new_case = case_2.copy()
            new_case.pick_leaf(leaf)
            todo_2.append(new_case)
            # print(new_case.node_dict)
    # print(I)