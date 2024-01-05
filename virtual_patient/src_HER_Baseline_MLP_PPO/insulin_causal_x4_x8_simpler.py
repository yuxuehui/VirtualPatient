import os
import sys

import numpy as np
import itertools
import copy
import pandas as pd
import torch

# paper here: https://arxiv.org/pdf/1901.08162.pdf
# N = 5 nodes, edges in upper triangular matrix from {-1, 0, 1}

N = 13


# ALL_ADJ_LISTS = [tuple(l) for l in
#                  itertools.product([-1, 0, 1], repeat=int(N * (N - 1) / 2))]
#
# adj_list1 = [0, -1, 0, 0,
# 	            -1, -1, 0,
# 	                0, -1,
#                       -1]


# def _get_random_adj_list(train):
#     idx = np.random.randint(0, len(ALL_ADJ_LISTS))
#     return ALL_ADJ_LISTS[idx]


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


def getParams(PATIENT_PARA_FILE, name):
    patient_params = pd.read_csv(PATIENT_PARA_FILE)
    params = patient_params.loc[patient_params.Name == name].squeeze()
    return params


def rerankVector(state):
    return np.hstack((state[0], state[1], state[4], state[7], state[8], state[10], state[11], state[12], state[2],
                      state[3], state[5], state[6], state[9], state[13]))


class MLP(torch.nn.Module):
    def __init__(self, num):
        super().__init__()
        self.layer1 = torch.nn.Linear(1, num)
        self.layer2 = torch.nn.Linear(num, num)
        self.layer3 = torch.nn.Linear(num, 1)

    def forward(self, x):
        x = self.layer1(x)
        x = torch.nn.functional.relu(x)

        x = self.layer2(x)
        x = torch.nn.functional.relu(x)

        x = self.layer3(x)

        return x


class RecNN(torch.nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.rnn = torch.nn.LSTM(input_size=1, hidden_size=self.hidden_size, num_layers=1, batch_first=True)
        # 至于这个线性层为什么是2维度接收，要看最后网络输出的维度是否匹配label的维度
        self.linear = torch.nn.Linear(self.hidden_size, 1)

    def forward(self, x):
        # print("x shape: {}".format(x.shape))
        # x [batch_size, seq_len, input_size]
        output, hn = self.rnn(x)
        # print("output shape: {}".format(output.shape))
        # out [seq_len, batch_size, hidden_size]
        x = output.reshape(-1, self.hidden_size)

        # print("after change shape: {}".format(x.shape))
        x = self.linear(x)

        # print("after linear shape: {}".format(x.shape))

        return x


class SimGlucose():
    EAT_RATE = 5  # g/min CHO
    DETA_TIME = 1  # min

    def __init__(self, params, flag_list, vertex, last_vertex, init_state=None):
        self.params = params
        self.init_state = init_state
        self.flag_list = flag_list
        self.vertex = vertex
        self.last_vertex = last_vertex
        self.reset()

    def step(self, intervene, carb):
        # 将吃一餐转化为实际饮食量
        CHO = self._announce_meal(carb)
        # print(ins, carb, CHO)
        # Detect eating or not and update last digestion amount
        if CHO > 0 and self.last_CHO <= 0:
            self._last_Qsto = self.state[0] + self.state[1]
            self._last_foodtaken = 0
            self.is_eating = True

        if self.is_eating:
            self._last_foodtaken += CHO  # g

        # Detect eating ended
        if CHO <= 0 and self.last_CHO > 0:
            self.is_eating = False

        self.last_CHO = CHO
        # self.last_ins = ins
        self.last_state = self.state
        self.last_all_state = copy.deepcopy(self.state)
        self.state = self.model(self.state, CHO, intervene, self.params, self._last_Qsto, self._last_foodtaken)
        # print(self.state)
        self.obe_state = self.obersevation()
        # print(self.obe_state)
        self.last_state = self.obe_last_state()

        # print('next state::', self.state)

    def model(self, x, CHO, intervene, params, last_Qsto, last_foodtaken):
        dxdt = np.zeros(13)

        d = CHO * 1000  # g -> mg
        # insulin = ins * 6000 / params.BW  # U/min -> pmol/kg/min
        # Glucose in the stomach
        qsto = x[0] + x[1]  # x[0]：Q_sto1(胃中固体葡萄糖质量)；x[1]：Q_sto2(胃中液体葡萄糖质量)
        Dbar = last_Qsto + last_foodtaken

        # Stomach solid
        dxdt[0] = (-params.kmax * x[0] + d) * self.flag_list[0]

        if Dbar > 0:
            aa = 5 / 2 / (1 - params.b) / Dbar
            cc = 5 / 2 / params.d / Dbar
            kgut = params.kmin + (params.kmax - params.kmin) / 2 * (np.tanh(
                aa * (qsto - params.b * Dbar)) - np.tanh(cc * (qsto - params.d * Dbar)) + 2)
        else:
            kgut = params.kmax

        # stomach liquid
        dxdt[1] = (params.kmax * x[0] - x[1] * kgut) * self.flag_list[1]

        # intestine
        dxdt[2] = (kgut * x[1] - params.kabs * x[2]) * self.flag_list[2]  # x[2]：Q_gut(肠中葡萄糖质量)

        # Rate of appearance
        Rat = params.f * params.kabs * x[2] / params.BW
        # Glucose Production
        EGPt = params.kp1 - params.kp2 * x[3] - params.kp3 * x[8]  # x[3]：G_p(t)(血浆中葡萄糖质量)
        # Glucose Utilization
        Uiit = params.Fsnc  # 1

        # renal excretion   肾脏排泄，大于一定阈值339，就进行排泄
        if x[3] > params.ke2:  # params.ke2=339
            Et = params.ke1 * (x[3] - params.ke2)  # params.ke1=0.0005
        else:
            Et = 0

        # glucose kinetics
        # plus dextrose IV injection input u[2] if needed
        dxdt[3] = max(EGPt, 0) + Rat - Uiit - Et - \
                  params.k1 * x[3] + params.k2 * x[4]  # x[4]：G_t(t)(慢速平衡组织中葡萄糖质量)
        dxdt[3] = ((x[3] >= 0) * dxdt[3]) * self.flag_list[3]  # 可以确保(x[3] >= 0)为TRUE的时候，相乘为dxdt[3]； 否则为0。

        Vmt = params.Vm0 + params.Vmx * x[6]  # x[6]：X(t)(葡萄糖利用中胰岛素的参与量)
        Kmt = params.Km0  # 246.8819
        Uidt = Vmt * x[4] / (Kmt + x[4])
        dxdt[4] = -Uidt + params.k1 * x[3] - params.k2 * x[4]
        dxdt[4] = ((x[4] >= 0) * dxdt[4]) * self.flag_list[4]

        # insulin kinetics
        dxdt[5] = -(params.m2 + params.m4) * x[5] + params.m1 * x[9] + params.ka1 * \
                  x[10] + params.ka2 * x[11]  # plus insulin IV injection u[3] if needed
        # x[5]：I_p(t)(血浆中胰岛素质量)；x[9]：I_l(t)(肾脏中胰岛素质量)；x[10]：I_ev(t)(血管外胰岛素质量)；x[11]：？
        It = x[5] / params.Vi
        dxdt[5] = ((x[5] >= 0) * dxdt[5]) * self.flag_list[5]

        # insulin action on glucose utilization
        dxdt[6] = (-params.p2u * x[6] + params.p2u * (It - params.Ib)) * self.flag_list[6]

        # insulin action on production
        dxdt[7] = (-params.ki * (x[7] - It)) * self.flag_list[7]  # x[7]:I'(t)(中间室胰岛素浓度)

        dxdt[8] = (-params.ki * (x[8] - x[7])) * self.flag_list[8]  # x[8]:XL(t)(延迟的胰岛素信号)

        # insulin in the liver (pmol/kg)
        dxdt[9] = -(params.m1 + params.m30) * x[9] + params.m2 * x[5]
        dxdt[9] = ((x[9] >= 0) * dxdt[9]) * self.flag_list[9]

        # subcutaneous insulin kinetics
        # dxdt[10] = insulin - (params.ka1 + params.kd) * x[10]
        # dxdt[10] = (x[10] >= 0) * dxdt[10]
        dxdt[10] = 0

        dxdt[11] = params.kd * x[10] - params.ka2 * x[11]
        dxdt[11] = ((x[11] >= 0) * dxdt[11]) * self.flag_list[11]

        # subcutaneous glcuose
        dxdt[12] = (-params.ksc * x[12] + params.ksc * x[3])  # x[3]：血管外葡萄糖质量
        dxdt[12] = ((x[12] >= 0) * dxdt[12]) * self.flag_list[12]

        for i in range(13):
            if i in self.vertex:
                x[i] = intervene[0]
            else:
                x[i] = x[i] + dxdt[i]
                x[i] *= self.flag_list[i]

        return x

    def obersevation(self):
        """
        返回可观察的状态
        """
        obe = []
        for i in range(13):
            if self.flag_list[i] == 1:
                obe.append(self.state[i])
        return obe

    def obe_last_state(self):
        """
        返回上一层干预节点的状态,需要除一个单位
        """
        obe_last = []
        for i in range(13):
            if i in self.last_vertex:
                j = self.state[i]
                obe_last.append(j)
        return obe_last

    def _announce_meal(self, meal):
        '''
        patient announces meal.
        The announced meal will be added to self.planned_meal
        The meal is consumed in self.EAT_RATE
        The function will return the amount to eat at current time
        '''
        self.planned_meal += meal
        if self.planned_meal > 0:
            to_eat = min(self.EAT_RATE, self.planned_meal)
            self.planned_meal -= to_eat
            self.planned_meal = max(0, self.planned_meal)
        else:
            to_eat = 0
        return to_eat

    def reset(self):
        '''
        Reset the patient state to default intial state
        '''
        self.state = copy.deepcopy(self.init_state)
        self.last_all_state = copy.deepcopy(self.init_state)
        self.obe_state = self.obersevation()
        self.last_state = self.obe_last_state()

        self._last_Qsto = self.state[0] + self.state[1]
        self._last_foodtaken = 0

        self.last_CHO = 0
        self.last_ins = 0

        self.is_eating = False
        self.planned_meal = 0


class CausalGraph:
    def __init__(self,
                 train=True,
                 permute=True,
                 vertex=[3],
                 last_vertex=[12],
                 patient_ID='adult#004',
                 flag=0,
                 meal_time=[]):
        """
        Create the causal graph structure.
        :param adj_list: 10 ints in {-1, 0, 1} which form the upper-tri adjacency matrix上三角邻接矩阵
        """
        # if adj_list is None:
        #     adj_list = _get_random_adj_list(train)
        self.vertex = vertex
        self.last_vertex = last_vertex
        self.adj_mat = self.handle(self.vertex, self.last_vertex)
        self.basal = 0.01393558889998341  # 基础率，这里需要根据RL给出
        self.patient_ID = patient_ID
        self.flag = flag
        self.meal_time = meal_time
        current_path = os.path.abspath(__file__)
        # 获取当前文件的父目录
        father_path = os.path.abspath(os.path.dirname(current_path) + os.path.sep + ".")
        print(father_path)
        sys.path.append(father_path)  # /home/yuxuehui/yxhfile/causal-metarl-master/src
        father_path = os.path.abspath(os.path.dirname(father_path) + os.path.sep + ".")
        sys.path.append(father_path)  # /home/yuxuehui/yxhfile/causal-metarl-master

        self.params = getParams(father_path + '/' + 'Data/vpatient_params.csv', self.patient_ID)
        self.init_state = []
        self.flag_list, self.len_obe = self.flag_vertex()

        print("self.flag_list:", self.flag_list)

        for item in self.params[2:15]:
            self.init_state.append(item)
        # self.init_state.append(self.params[14]) # 加入x12向量

        self.simulator = SimGlucose(params=self.params, init_state=self.init_state, vertex=self.vertex,
                                    last_vertex=self.last_vertex, flag_list=self.flag_list)

        self.time = 0
        self.meal = 50  # 设置一日三餐： 6:00  11:00  18:00，每餐吃50g碳水
        self.CR = 16  # 大剂量，直接根据meal计算， meal/CR
        self.CF = 42.65337551
        self.target = 140
        self.reset_graph()
        print("Graph初始化:", self.simulator.state[8] / self.params.Vg, self.simulator.state)
        print('patient_ID: ' + self.patient_ID)

        # TODO: get equivalence classes for graphs

    def handle(self, vertex, last_vertex):
        """
        将干预节点的父节点切掉，上一层干预节点的子节点切掉
        """
        adj_mat = np.array(
            [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 0
             [-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 1
             [-1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 2
             [0, 0, -1, 0, -1, 0, 0, 0, -1, 0, 0, 0, 0],  # 3
             [0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0],  # 4
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, 0],  # 5
             [0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0],  # 6
             [0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0],  # 7
             [0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0],  # 8
             [0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0],  # 9
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 10
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0],  # 11
             [0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0]]  # 12
        )
        for i in vertex:
            for j in range(13):
                adj_mat[i][j] = 0
        for i in last_vertex:
            for j in range(13):
                adj_mat[j][i] = 0
        return adj_mat

    def flag_vertex(self):
        """
        顶点是否在该层强化学习中，如果是返回1，不是返回0，本质上是一个有向图的遍历,或许return一个列表
        然后传入SimGlucose比较合适(0,1)列表
        """
        flag_list = [1] * 13
        num_ = 13

        return flag_list, num_

    def reset_graph(self):
        """
        设置一开始初始化graph，先运行1天（1440min）
        return:
        """
        carb = 0
        state = self.init_state

        self.simulator.reset()

    def intervene(self, action):
        """
        干预 insulin 为 RL 的 action （这里只干预基础胰岛素）
        Intervene on the node at node_idx by setting its value to val.
        :param node_idx: (int) node to intervene on
        :param val: (float) value to set
        """
        breakfast = 0
        lunch = 0
        dinner = 0
        empty = []
        if self.meal_time == empty:
            breakfast = 360
            lunch = 660
            dinner = 1080
        else:
            breakfast = self.meal_time[0]
            lunch = self.meal_time[1]
            dinner = self.meal_time[2]
        carb = 0
        intervene = 0
        if self.flag == 0:
            print(self.time)
            intervene = action
            if self.time % breakfast == 0 and self.time % (breakfast * 2) != 0 and self.time != 0:
                carb = 50
                ins = action + self.meal / self.CR + (
                        self.simulator.state[8] / self.params.Vg - self.target) / self.CF  # 计算餐前大剂量胰岛素
                intervene = ins
            if self.time % lunch == 0 and self.time % (lunch * 2) != 0 and self.time != 0:
                carb = 50
                ins = action + self.meal / self.CR + (
                        self.simulator.state[8] / self.params.Vg - self.target) / self.CF  # 计算餐前大剂量胰岛素
                intervene = ins
            if self.time % dinner == 0 and self.time % (dinner * 2) != 0 and self.time != 0:
                carb = 50
                ins = action + self.meal / self.CR + (
                        self.simulator.state[8] / self.params.Vg - self.target) / self.CF  # 计算餐前大剂量胰岛素
                intervene = ins
            self.simulator.step(intervene, carb)
            self.time += 1
        if self.flag == 1:
            if self.time == 0:
                for i in range(breakfast):
                    intervene = np.array([0.01])
                    self.simulator.step(intervene, 0)
                    self.time += 1
                return
            print(self.time)
            carb = action
            intervene = 0.01
            if (self.time % breakfast == 0 and self.time % (breakfast * 2) != 0 and self.time != 0) \
                    or (self.time % lunch == 0 and self.time % (lunch * 2) != 0 and self.time != 0) \
                    or (self.time % dinner == 0 and self.time % (dinner * 2) != 0 and self.time != 0):
                ins = 0.01 + self.meal / self.CR + (
                        self.simulator.state[8] / self.params.Vg - self.target) / self.CF  # 计算餐前大剂量胰岛素
                intervene = ins
            intervene = np.array([intervene])
            self.simulator.step(intervene, carb)
            self.time += 1
            if (self.time - 1) % breakfast == 0 and (self.time - 1) % (breakfast * 2) != 0 and (self.time - 1) != 0:
                for i in range(lunch - breakfast - 1):
                    intervene = np.array([0.01])
                    self.simulator.step(intervene, 0)
                    self.time += 1
            if (self.time - 1) % lunch == 0 and (self.time - 1) % (lunch * 2) != 0 and (self.time - 1) != 0:
                for i in range(dinner - lunch - 1):
                    intervene = np.array([0.01])
                    self.simulator.step(intervene, 0)
                    self.time += 1
            if (self.time - 1) % dinner == 0 and (self.time - 1) % (dinner * 2) != 0 and (self.time - 1) != 0:
                for i in range(1440 - dinner - 1):
                    intervene = np.array([0.01])
                    self.simulator.step(intervene, 0)
                    self.time += 1
        if self.flag == 2:
            intervene = action[0]
            carb = action[1]
            if (self.time % breakfast == 0 and self.time % (breakfast * 2) != 0 and self.time != 0) \
                    or (self.time % lunch == 0 and self.time % (lunch * 2) != 0 and self.time != 0) \
                    or (self.time % dinner == 0 and self.time % (dinner * 2) != 0 and self.time != 0):
                ins = intervene + self.meal / self.CR + (
                        self.simulator.state[8] / self.params.Vg - self.target) / self.CF  # 计算餐前大剂量胰岛素
                intervene = ins
            intervene = np.array([intervene])
            self.simulator.step(intervene, carb)
            self.time += 1
    def sample_all(self):
        """
        返回当前13个变量状态 和 当前血糖值
        :return: sampled_vals (np.ndarray) array of sampled values
        """

        return self.simulator.obe_state, self.simulator.last_state

    def get_last_value(self, node_idx):
        """
        Get the value at index node_idx
        :param node_idx: (int)
        :return: val (float)
        """
        return self.simulator.last_all_state[node_idx]

    def get_value(self, node_idx):
        """
        Get the value at index node_idx
        :param node_idx: (int)
        :return: val (float)
        """
        # print("get_value:", self.simulator.state[node_idx])
        return self.simulator.state[node_idx]
