import copy

import numpy as np
import itertools

# paper here: https://arxiv.org/pdf/1901.08162.pdf
# N = 5 nodes, edges in upper triangular matrix from {-1, 0, 1}
global N
N = 8
# ALL_ADJ_LISTS = [tuple(l) for l in
#                  itertools.product([-1, 0, 1], repeat=int(N * (N - 1) / 2))]
#
# adj_list1 = [0, 1, 0, 0,
#                -1, 0, 0,
#                   -1, 0,
#                      -1]


def _get_random_adj_list(train):
    idx = np.random.randint(0, len(ALL_ADJ_LISTS))
    return ALL_ADJ_LISTS[idx]


def _swap_rows_and_cols(arr_original, permutation):
    """
    根据 permutation 的节点遍历顺序，调整 arr 的节点遍历顺序
    :param arr_original:
    :param permutation:
    :return:
    """
    if not isinstance(permutation, list):
        permutation = list(permutation)
    arr = arr_original.copy()
    arr[:] = arr[permutation]   # 按照 permutation 的顺序重新排列
    arr[:, :] = arr[:, permutation]     # 按照 permutation 的顺序重新排列
    return arr


def get_permuted_adj_mats(adj_list):
    """
    Returns adjacency matrices which are valid permutations, meaning that
    the root node (index = 4) does not have any parents.
    :param adj_list: 10 ints in {-1, 0, 1} which form the upper-tri adjacency matrix
    :return perms: list of adjacency matrices
    """
    adj_mat = np.zeros((N, N))
    adj_triu_list = np.triu_indices(N, 1)
    adj_mat[adj_triu_list] = adj_list
    perms = set()

    for perm in itertools.permutations(np.arange(N), N):
        permed = _swap_rows_and_cols(adj_mat, perm)
        if not any(permed[N - 1]):
            perms.add(tuple(permed.reshape(-1)))

    return perms


def true_separate_train_and_test():
    """Return a list of adjacency matrices for each, training and testing."""
    test = set()
    adj_lists_copy = list(ALL_ADJ_LISTS)
    while True:
        idx = np.random.randint(0, len(adj_lists_copy))
        perms = get_permuted_adj_mats(adj_lists_copy[idx])
        test.update(perms)
        # print(len(test))
        if len(test) > 408:
            break


class CausalGraph:
    def __init__(self,
                 train=True,
                 permute=True,
                 intervene_idx=None,
                 vertex=[3],
                 last_vertex=[12]):
        """
        Create the causal graph structure.
        :param adj_list: 10 ints in {-1, 0, 1} which form the upper-tri adjacency matrix
        """
        self.nodes = [CausalNode(i) for i in range(N)]
        self.nodes[0].val = 0.0
        self.nodes[1].val = 10.0
        self.nodes[2].val = 0.01
        self.nodes[3].val = 0.01
        self.nodes[4].val = 0.01
        self.nodes[5].val = 5.0
        self.nodes[6].val = 0.01
        self.nodes[7].val = 0.01
        self.vertex = vertex
        self.time = 0
        self.last_vertex = last_vertex
        self.adj_mat = self.handle(self.vertex, self.last_vertex)
        self.flag_list, self.len_obe = self.flag_vertex()
        self.obe_state = self.obersevation()
        self.last_state = self.obe_last_state()
        print(self.last_vertex, "Graph init:", self.print_graph() )

    def handle(self, vertex, last_vertex):
        """
        将干预节点的父节点切掉，上一层干预节点的子节点切掉
        """
        adj_mat = np.array(
            [[0, 0, 0, 0, 0, 0, 0, 0],  # 0
             [-1,0,-1, 0, 0, 0, 0, 0],  # 1
             [0, 0, 0,-1, 0, 0, 0, -1],  # 2
             [0, 0, 0, 0,-1, 0, 0, 0],  # 3
             [0, 0, 0, 0, 0,-1, 0, 0],  # 4
             [0, 0, 0, 0, 0, 0, 0, 0],    # 5
             [0, 0, 0, 0, 0,-1, 0, 0],    # 6
             [0, 0, 0, 0, 0, 0, -1, 0]]   # 7
        )
        for i in vertex:
            for j in range(N):
                adj_mat[i][j] = 0
        for i in last_vertex:
            for j in range(N):
                adj_mat[j][i] = 0
        return adj_mat

    def flag_vertex(self):
        """
        顶点是否在该层强化学习中，如果是返回1，不是返回0，本质上是一个有向图的遍历,或许return一个列表
        然后传入SimGlucose比较合适(0,1)列表
        """
        flag_list = [0] * N
        que = []

        for m in range(len(self.vertex)):
            start_vertex = self.vertex[m]
            flag_list[start_vertex] = 1
            que.insert(0, start_vertex)
            while len(que) > 0:
                front = que[0]
                que.pop(0)
                for i in range(N):
                    if self.adj_mat[i][front] == -1 and flag_list[i] == 0:
                        flag_list[i] = 1
                        que.append(i)

        for p in range(len(self.last_vertex)):
            start_vertex = self.last_vertex[p]
            flag_list[start_vertex] = 1
            que.insert(0, start_vertex)
            while len(que) > 0:
                front = que[0]
                que.pop(0)
                for i in range(N):
                    if self.adj_mat[front][i] == -1 and flag_list[i] == 0:
                        flag_list[i] = 1
                        que.append(i)
        num_ = 0
        for i in range(N):
            if flag_list[i] == 1:
                num_ += 1
        return flag_list, num_

    # TODO: get equivalence classes for graphs
    def print_graph(self):
        str_return = ''
        for idx in range(len(self.nodes)):
            str_return = str_return + str(self.nodes[idx].val) + " "
        return str_return

    def intervene(self, node_idx, val):
        """
        Intervene on the node at node_idx by setting its value to val.
        :param node_idx: (int) node to intervene on
        :param val: (float) value to set
        """
        self.nodes[node_idx].intervene(val)
        if node_idx == 4:
            self.nodes[6].intervene(val*8)

    def increase(self, node_idx, val):
        self.nodes[node_idx].increase(val)

    def calculate(self):
        """
        干预完了之后计算没干预点的值
        :return:
        """
        self.time = self.time + 1
        delta = np.zeros(N)
        delta[1] = self.nodes[0].val * 0.5 \
                   + (0.5 + 100 / (1 + pow(self.nodes[2].val, 2) * max((self.time - 2), 0)) - self.nodes[1].val) * 0.1
        delta[0] = - self.nodes[0].val * 0.1 - 1.0  # 自衰减
        delta[2] = (0.5 + 110 / (1 + pow(self.nodes[3].val, -2) * max((self.time - 2), 0)) - self.nodes[2].val) * 0.1 \
                    + (0.5 + 100 / (1 + pow(self.nodes[7].val, 2) * max((self.time - 2), 0)) - self.nodes[2].val) * 0.1
        delta[3] = (0.5 + 110 / (1 + pow(max(self.nodes[4].val, 0.01), -2) * max((self.time - 2), 0)) - self.nodes[3].val) * 0.1
        delta[4] = (0.5 + 110 / (1 + pow(max(self.nodes[5].val, 0.01), -2) * max((self.time - 2), 0)) - self.nodes[4].val) * 0.1
        delta[6] = (0.5 + 110 / (1 + pow(max(self.nodes[5].val, 0.01), -2) * max((self.time - 2), 0)) - self.nodes[6].val) * 0.8
        delta[7] = (0.5 + 110 / (1 + pow(max(self.nodes[6].val, 0.01), -2) * max((self.time - 2), 0)) - self.nodes[7].val) * 0.1
        for idx in range(len(self.nodes)):
            if not self.nodes[idx].intervened:
                self.nodes[idx].last_val = copy.deepcopy(self.nodes[idx].val)
            self.nodes[idx].val = max((self.nodes[idx].val + delta[idx]) * self.flag_list[idx], 0.01)

    def sample_all(self):
        """
        Sample all nodes according to their causal relations
        :return: sampled_vals (np.ndarray) array of sampled values
        """
        sampled_vals = np.zeros(N)
        for idx in range(len(self.nodes)):
            sampled_vals[idx] = copy.deepcopy(self.nodes[idx].val)

        self.obe_state = self.obersevation()
        self.last_state = self.obe_last_state()

        return sampled_vals, self.obe_state, self.last_state

    def get_value(self, node_idx):
        """
        Get the value at index node_idx
        :param node_idx: (int)
        :return: val (float)
        """
        # print("self.nodes[node_idx].val:", self.nodes[node_idx].val)
        return self.nodes[node_idx].val

    def get_last_value(self, node_idx):
        """
        Get the value at index node_idx
        :param node_idx: (int)
        :return: val (float)
        """
        return self.nodes[node_idx].last_val

    def obersevation(self):
        """
        返回可观察的状态
        """
        obe = []
        for i in range(N):
            if self.flag_list[i] == 1:
                obe.append(self.nodes[i].val)
        return obe

    def obe_last_state(self):
        """
        返回上一层干预节点的状态,需要除一个单位
        """
        obe_last = []
        if 4 in self.last_vertex:
            j = self.nodes[4].val
            obe_last.append(j)
        else:
            for i in range(13):
                if i in self.last_vertex:
                    j = self.nodes[i].val
                    obe_last.append(j)
        return obe_last
        # obe_last = []
        # for i in range(N):
        #     if i in self.last_vertex:
        #             j = self.nodes[i].val
        #             obe_last.append(j)
        # return obe_last


class CausalNode:
    def __init__(self, idx):
        """
        Create data structure for node which knows its parents
        :param idx: index of node in graph
        :param adj_mat: upper triangular matrix for graph
        """
        self.id = idx
        self.val = None
        self.last_val = None
        self.intervened = False

    def increase(self, val):
        self.last_val = copy.deepcopy(self.val)
        self.val = copy.deepcopy(val + self.val)
        self.intervened = True


    def intervene(self, val):
        """
        Intervene on this node. Valid for one call to sample()
        :param val: (float) value to set this node to
        """
        self.last_val = copy.deepcopy(self.val)
        self.val = copy.deepcopy(val)
        self.intervened = True
