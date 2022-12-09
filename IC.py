import networkx as nx
import random
import pandas as pd
import matplotlib.pyplot as plt


class ICModel:
    G = nx.DiGraph()
    active_nodes = set()
    active_current = []
    active_next = []
    P = 0.1

    def __init__(self, graph, init_active, p):
        """
        类的初始化
        :param graph:
        :param init_active: 初始种子节点集合
        :param p: 传播概率
        """
        self.G = graph
        self.active_current = init_active
        self.active_nodes = set(init_active)
        self.P = p

    def spread(self):
        """
        IC模型的传播部分
        """
        while True:
            for node in self.active_current:
                for adj in list(self.G.predecessors(node)):
                    if adj in self.active_nodes:
                        continue
                    if random.random() <= self.P:
                        self.active_nodes.add(adj)
                        self.active_next.append(adj)
            if len(self.active_next) == 0:
                return
            self.active_current = self.active_next.copy()
            self.active_next.clear()

    def run(self):
        random.seed(150)
        self.spread()
        return len(self.active_nodes)


def readCsv(path):
    df_list = pd.read_csv(path).values.tolist()
    ans_list = []
    for item in df_list:
        ans_list.append((str(int(item[0])), item[1]))
    return ans_list


# IC评价模型
if __name__ == '__main__':
    DG_path = 'follow_relationships.graphml'
    DG = nx.read_graphml(DG_path)

    PR = readCsv("./datacsv/pg_rank.csv")
    HIT = readCsv("./datacsv/hits_rank.csv")
    LR = readCsv("./datacsv/leader_rank.csv")
    COM = readCsv("./datacsv/upgraded_rank.csv")

    K_list = [10, 15, 20, 25, 30]
    probs = [0.01]
    # 经过实验发现p=0.01最适合用于判断评价种子节点的传播影响力，故仅留下0.01

    for prob in probs:
        eval1s = []
        eval2s = []
        eval3s = []
        eval4s = []
        for i in K_list:
            PRtopk = [j[0] for j in PR[:i]]
            HITtopk = [j[0] for j in HIT[:i]]
            LRTtopk = [j[0] for j in LR[:i]]
            CoTtopk = [j[0] for j in COM[:i]]
            eval1 = ICModel(DG, PRtopk, prob).run()
            eval2 = ICModel(DG, HITtopk, prob).run()
            eval3 = ICModel(DG, LRTtopk, prob).run()
            eval4 = ICModel(DG, CoTtopk, prob).run()

            eval1s.append(eval1)
            eval2s.append(eval2)
            eval3s.append(eval3)
            eval4s.append(eval4)

        # 创建画布
        plt.figure()
        plt.plot(K_list, eval1s, marker='o', color='r', label='PageRank')
        plt.plot(K_list, eval2s, marker='*', color='b', label='Hits')
        plt.plot(K_list, eval3s, marker='s', color='y', label='LeaderRank')
        plt.plot(K_list, eval4s, marker='s', color='g', label='our_algorithm')

        plt.legend()
        plt.xlabel('K-Value')
        plt.ylabel('Influencer-Num')
        # 保存图片到本地
        plt.savefig('./' + str(prob) + '.png')
