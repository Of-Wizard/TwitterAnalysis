import csv
import networkx as nx
import operator


def k_shell(G):
    """
    返回k核节点列表
    :param G:
    :return:
    """
    # 计算每个节点属于K_core
    k_shell_list = []
    for k in range(0, 100):
        flag = True
        total_delete_nodes = []
        # 迭代删除节点度数小于等于k的节点
        while flag:
            nodes = G.nodes
            delete_nodes = []
            for node in nodes:
                if G.in_degree(node) <= k:
                    delete_nodes.append(node)
            total_delete_nodes = total_delete_nodes + delete_nodes
            G.remove_nodes_from(delete_nodes)
            # 在此轮迭代中若没有删除节点则说明属于k_core的节点已筛选完毕，可以进行下一轮迭代
            if len(delete_nodes) == 0:
                flag = False
        k_shell_list.append(total_delete_nodes)
    return k_shell_list


def upgraded_algorithm(G, alpha=0.75, max_iter=100, tol=1.0e-9):
    """
    基于k-shell和pagerank改进算法
    :param G:有向图
    :param alpha: 阻尼系数
    :param max_iter: 最大迭代次数
    :param tol: 迭代误差
    :return:
    """
    # 增加一个背景点，使得所有节点与该背景点双向链接
    nodes = G.nodes
    add_edges = []
    for node in nodes:
        add_edges.append((0, node))
        add_edges.append((node, 0))
    G.add_node(0)
    G.add_edges_from(add_edges)

    D = G
    N = D.number_of_nodes()
    nbr_weight = {}

    k_core_list = k_shell(G.copy())
    for index, Kth_core_nodes in enumerate(k_core_list):
        if len(k_core_list) > 0:
            for node in Kth_core_nodes:
                D.nodes[node]['k_weight'] = index

    for node in D.nodes:
        weight = 0
        for nbr in D[node]:
            # weight+=D.in_degree(nbr)
            # weight += (D.in_degree(nbr)/D.out_degree(nbr))
            weight += D.nodes[nbr]['k_weight']
        nbr_weight.update({node: weight})

    x = dict.fromkeys(D, 1.0 / (N - 1))
    p = dict.fromkeys(D, 1.0 / (N - 1))

    for _ in range(max_iter):
        xlast = x
        x = dict.fromkeys(xlast.keys(), 0)
        for n in x:
            for nbr in D[n]:
                # x[nbr] += alpha * xlast[n]*(D.in_degree(nbr)/nbr_weight.get(n,0))
                # x[nbr] += alpha * xlast[n]*((D.in_degree(nbr) / D.out_degree(nbr))/nbr_weight.get(n,0))
                x[nbr] += alpha * xlast[n] * (D.nodes[nbr]['k_weight'] / nbr_weight.get(n, 0))
            x[n] += (1.0 - alpha) * p.get(n, 0)
        avg_ground_value = x[0] / (N - 1)
        for i in range(1, N):
            x[i] = avg_ground_value + x[i]
        x[0] = 0
        err = sum([abs(x[n] - xlast[n]) for n in x])
        # print(err)
        if err < N * tol:
            x.pop(0)
            G.remove_node(0)
            return x
