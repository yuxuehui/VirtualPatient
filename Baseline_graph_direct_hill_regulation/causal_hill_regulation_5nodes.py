import copy

import numpy as np
import itertools

# paper here: https://arxiv.org/pdf/1901.08162.pdf
# N = 5 nodes, edges in upper triangular matrix from {-1, 0, 1}
global N
N = 5
ALL_ADJ_LISTS = [tuple(l) for l in
                 itertools.product([-1, 0, 1], repeat=int(N * (N - 1) / 2))]

adj_list1 = [0, 1, 0, 0,
             -1, 0, 0,
             -1, 0,
             -1]


def _get_random_adj_list(train):
    idx = np.random.randint(0, len(ALL_ADJ_LISTS))
    return ALL_ADJ_LISTS[idx]


def _swap_rows_and_cols(arr_original, permutation):
    """
    根据permutation的节点遍历顺序，调整 arr 的节点遍历顺序
    :param arr_original:
    :param permutation:
    :return:
    """
    if not isinstance(permutation, list):
        permutation = list(permutation)
    arr = arr_original.copy()
    arr[:] = arr[permutation]  # 按照 permutation 的顺序重新排列
    arr[:, :] = arr[:, permutation]  # 按照 permutation 的顺序重新排列
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
        print(len(test))
        if len(test) > 408:
            break


class CausalGraph:
    def __init__(self,
                 adj_list=adj_list1,
                 train=True,
                 permute=True,
                 intervene_idx=None):
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
        self.time = 0
        print("Graph init:", self.print_graph())

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

    def increase(self, node_idx, val):
        self.nodes[node_idx].increase(val)

    def calculate(self):
        """
        干预完了之后计算没干预点的值
        :return:
        """
        # print("self.time:", self.time)
        self.time = self.time + 1
        delta = np.zeros(N)
        delta[1] = self.nodes[0].val * 0.5 + (
                0.5 + 100 / (1 + pow(self.nodes[2].val, 2) * max((self.time - 2), 0)) - self.nodes[1].val) * 0.1
        delta[0] = - self.nodes[0].val * 0.1 - 1.0  # 自衰减
        delta[2] = (0.5 + 100 / (1 + pow(self.nodes[3].val, -2) * max((self.time - 2), 0)) - self.nodes[2].val) * 0.1
        delta[3] = (0.5 + 100 / (1 + pow(max(self.nodes[4].val, 0.01), -2) * max((self.time - 2), 0)) - self.nodes[
            3].val) * 0.1
        for idx in range(len(self.nodes)):
            if not self.nodes[idx].intervened:
                self.nodes[idx].last_val = copy.deepcopy(self.nodes[idx].val)
            self.nodes[idx].val = max(self.nodes[idx].val + delta[idx], 0.01)

    def sample_all(self):
        """
        Sample all nodes according to their causal relations
        :return: sampled_vals (np.ndarray) array of sampled values
        """
        sampled_vals = np.zeros(N)
        for idx in range(len(self.nodes)):
            sampled_vals[idx] = copy.deepcopy(self.nodes[idx].val)
        return sampled_vals

    def get_value(self, node_idx):
        """
        Get the value at index node_idx
        :param node_idx: (int)
        :return: val (float)
        """
        return self.nodes[node_idx].val

    def get_last_value(self, node_idx):
        """
        Get the value at index node_idx
        :param node_idx: (int)
        :return: val (float)
        """
        return self.nodes[node_idx].last_val


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
        self.val = val + self.val
        self.intervened = True

    def intervene(self, val):
        """
        Intervene on this node. Valid for one call to sample()
        :param val: (float) value to set this node to
        """
        self.last_val = copy.deepcopy(self.val)
        self.val = val
        self.intervened = True
