import networkx as nx


def leader_rank(G, alpha=0.75, max_iter=100, tol=1.0e-9):
    """
    :param G: 有向图
    :param alpha: 阻尼系数
    :param max_iter: 最大迭代次数
    :param tol: 迭代的阈值,小于tol 退出迭代
    :return:
    """
    nodes = G.nodes
    add_edges = []
    for node in nodes:
        add_edges.append((0, node))
        add_edges.append((node, 0))
    G.add_node(0)
    G.add_edges_from(add_edges)

    D = G
    W = nx.stochastic_graph(D, weight='weight')
    N = W.number_of_nodes()
    # Choose fixed starting vector if not given
    x = dict.fromkeys(W, 1.0 / (N - 1))

    p = dict.fromkeys(W, 1.0 / (N - 1))
    dangling_weights = p
    dangling_nodes = [n for n in W if W.out_degree(n, weight='weight') == 0.0]

    # power iteration: make up to max_iter iterations
    for _ in range(max_iter):
        xlast = x
        x = dict.fromkeys(xlast.keys(), 0)
        danglesum = alpha * sum(xlast[n] for n in dangling_nodes)
        for n in x:
            for nbr in W[n]:
                x[nbr] += alpha * xlast[n] * W[n][nbr]['weight']
            x[n] += danglesum * dangling_weights.get(n, 0) + (1.0 - alpha) * p.get(n, 0)
        avg_ground_value = x[0] / (N - 1)
        for i in range(1, N):
            x[i] = avg_ground_value + x[i]
        x[0] = 0
        # check convergence, l1 norm
        err = sum([abs(x[n] - xlast[n]) for n in x])
        if err < N * tol:
            x.pop(0)
            G.remove_node(0)
            return x
